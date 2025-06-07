const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;
let pythonProcess = null;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      webviewTag: true,
      webSecurity: false  // Required for PDF.js to work with local files
    }
  });

  mainWindow.loadFile('index.html');
  
  // Start Python backend
  startPythonBackend();
}

function startPythonBackend() {
    console.log('[Main] Starting Python backend...');
    
    // Just use the current environment - don't try to detect anything
    const currentEnv = { ...process.env };
    console.log('[Main] VIRTUAL_ENV:', currentEnv.VIRTUAL_ENV || 'Not set');
    console.log('[Main] CONDA_DEFAULT_ENV:', currentEnv.CONDA_DEFAULT_ENV || 'Not set');
    
    // Try different Python executables and file names
    const pythonExecutables = ['python', 'python3', 'py'];
    const pythonFiles = ['app.py', 'improved_backend.py'];
    
    // Look in these directories (in order of preference)
    const searchDirectories = [
        path.join(__dirname, 'ai_backend'),  // First check ai_backend subdirectory
        __dirname,                          // Then check main directory
        path.join(__dirname, 'backend'),    // Also check common backend folder names
        path.join(__dirname, 'python')      // And python folder
    ];
    
    let started = false;
    let foundPythonFile = null;
    let workingDirectory = null;
    
    // First, find where the Python file actually exists
    for (const dir of searchDirectories) {
        for (const file of pythonFiles) {
            const filePath = path.join(dir, file);
            if (require('fs').existsSync(filePath)) {
                foundPythonFile = file;
                workingDirectory = dir;
                console.log(`[Main] Found Python file: ${foundPythonFile} in ${workingDirectory}`);
                break;
            }
        }
        if (foundPythonFile) break;
    }
    
    if (!foundPythonFile) {
        console.error('[Main] Could not find app.py or improved_backend.py in any of these directories:');
        searchDirectories.forEach(dir => console.error(`  - ${dir}`));
        
        dialog.showErrorBox('Python Script Missing', 
            'Could not find app.py or improved_backend.py.\n\n' +
            'Searched in:\n' +
            searchDirectories.map(dir => `• ${dir}`).join('\n') + '\n\n' +
            'Please ensure your Python backend file exists in one of these locations.'
        );
        return;
    }
    
    // Now try to start the Python process
    for (const pythonExe of pythonExecutables) {
        if (started) break;
        
        try {
            console.log(`[Main] Trying to start: ${pythonExe} ${foundPythonFile} in ${workingDirectory}`);
            
            pythonProcess = spawn(pythonExe, ['-u', foundPythonFile], {
                stdio: ['pipe', 'pipe', 'pipe'],
                cwd: workingDirectory,  // Use the directory where we found the Python file
                env: currentEnv         // Pass the current environment
            });
            
            // Test if process started successfully
            pythonProcess.on('error', (err) => {
                console.error(`[Main] Failed to start ${pythonExe} ${foundPythonFile}:`, err.message);
                pythonProcess = null;
            });
            
            pythonProcess.stdout.on('data', (data) => {
                console.log('[Main] Raw data from Python stdout:', data.toString());
                const lines = data.toString().split('\n');
                lines.forEach(line => {
                    if (line.trim()) {
                        try {
                            // Try to parse as JSON (backend messages)
                            const message = JSON.parse(line.trim());
                            console.log('[Main] Successfully parsed JSON message:', message);
                            
                            // Forward to all renderer processes
                            const allWindows = BrowserWindow.getAllWindows();
                            allWindows.forEach(window => {
                                if (window && !window.isDestroyed()) {
                                    console.log('[Main] Forwarding message to window:', window.id);
                                    window.webContents.send('python-message', message);
                                }
                            });
                        } catch (e) {
                            // Regular log message
                            console.log('[Python] Non-JSON message:', line.trim());
                        }
                    }
                });
            });
            
            pythonProcess.stderr.on('data', (data) => {
                const errorMsg = data.toString();
                console.error('[Python Error] Raw stderr:', errorMsg);
                
                // Check for common Python errors
                if (errorMsg.includes('ModuleNotFoundError') || errorMsg.includes('ImportError')) {
                    console.error('[Main] Missing Python dependencies. Please run: pip install sentence-transformers scikit-learn flask flask-cors numpy');
                }
            });
            
            pythonProcess.on('close', (code) => {
                console.log(`[Python] Process exited with code ${code}`);
                pythonProcess = null;
                
                if (code !== 0) {
                    console.error('[Main] Python process crashed, attempting restart...');
                    setTimeout(() => {
                        if (!pythonProcess) {
                            startPythonBackend();
                        }
                    }, 5000);
                }
            });
            
            // Wait a moment to see if process starts successfully
            setTimeout(() => {
                if (pythonProcess && !pythonProcess.killed) {
                    started = true;
                    console.log(`[Main] Successfully started Python backend: ${pythonExe} ${foundPythonFile} from ${workingDirectory}`);
                }
            }, 1000);
            
            break;
            
        } catch (err) {
            console.error(`[Main] Error starting ${pythonExe} ${foundPythonFile}:`, err.message);
            pythonProcess = null;
            continue;
        }
    }
    
    // Check if we succeeded after trying all combinations
    setTimeout(() => {
        if (!started || !pythonProcess) {
            console.error('[Main] Failed to start Python backend with any combination');
            
            // Show error to user
            dialog.showErrorBox('Python Backend Error', 
                'Failed to start Python backend.\n\n' +
                'Please ensure:\n' +
                '1. Python is installed and in PATH\n' +
                '2. Required Python packages are installed:\n' +
                '   pip install sentence-transformers scikit-learn flask flask-cors numpy\n\n' +
                `Found Python file: ${foundPythonFile}\n` +
                `In directory: ${workingDirectory}\n` +
                `Environment info:\n` +
                `VIRTUAL_ENV: ${currentEnv.VIRTUAL_ENV || 'Not set'}\n` +
                `CONDA_DEFAULT_ENV: ${currentEnv.CONDA_DEFAULT_ENV || 'Not set'}`
            );
        }
    }, 2000);
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    if (pythonProcess) {
      pythonProcess.kill();
    }
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// IPC Handlers - IMPORTANT: Make sure these pass through all data correctly
ipcMain.on('search-tabs', (event, data) => {
    console.log('[Main] Received search-tabs request:', JSON.stringify(data));
    console.log('[Main] Python process status:', {
        exists: !!pythonProcess,
        writable: pythonProcess?.stdin?.writable,
        killed: pythonProcess?.killed
    });
    
    if (!pythonProcess || !pythonProcess.stdin.writable) {
        console.error('[Main] Python process not available for search');
        event.reply('python-message', {
            type: 'error',
            message: 'Python backend not available. Please restart the application.'
        });
        return;
    }
    
    // Ensure the message has the required structure
    const message = {
        type: 'search',
        browser_id: data.browser_id || 'default',  // This is crucial!
        query: data.query,
        threshold: data.threshold || 0.5
    };
    
    console.log('[Main] Sending to Python:', JSON.stringify(message));
    
    try {
        pythonProcess.stdin.write(JSON.stringify(message) + '\n');
        console.log('[Main] Successfully wrote message to Python stdin');
    } catch (err) {
        console.error('[Main] Error writing to Python process:', err);
        event.reply('python-message', {
            type: 'error',
            message: 'Failed to communicate with Python backend'
        });
    }
});

ipcMain.on('update-tab', (event, data) => {
    console.log('[Main] Received update-tab request for browser:', data.browser_id);
    
    if (!pythonProcess || !pythonProcess.stdin.writable) {
        console.error('[Main] Python process not available for tab update');
        return;
    }
    
    const message = {
        type: 'update_tab',
        browser_id: data.browser_id || 'default',  // This is crucial!
        tab_id: data.tab_id,
        title: data.title,
        url: data.url,
        content: data.content
    };
    
    console.log('[Main] Sending tab update to Python (content length:', data.content?.length || 0, ')');
    
    try {
        pythonProcess.stdin.write(JSON.stringify(message) + '\n');
    } catch (err) {
        console.error('[Main] Error writing to Python process:', err);
    }
});

ipcMain.on('update-settings', (event, data) => {
    console.log('[Main] Received update-settings request:', JSON.stringify(data));
    
    if (!pythonProcess || !pythonProcess.stdin.writable) {
        console.error('[Main] Python process not available for settings update');
        return;
    }
    
    const message = {
        type: 'update_settings',
        browser_id: data.browser_id || 'default',
        chunk_size: data.chunk_size,
        reprocess_tabs: data.reprocess_tabs || false
    };
    
    try {
        pythonProcess.stdin.write(JSON.stringify(message) + '\n');
    } catch (err) {
        console.error('[Main] Error writing to Python process:', err);
    }
});

ipcMain.on('get-stats', (event, data) => {
    console.log('[Main] Received get-stats request:', JSON.stringify(data));
    
    if (!pythonProcess || !pythonProcess.stdin.writable) {
        console.error('[Main] Python process not available for stats');
        event.reply('python-message', {
            type: 'backend_stats',
            error: 'Python backend not available'
        });
        return;
    }
    
    const message = {
        type: 'get_stats',
        browser_id: data.browser_id || 'default'
    };
    
    try {
        pythonProcess.stdin.write(JSON.stringify(message) + '\n');
    } catch (err) {
        console.error('[Main] Error writing to Python process:', err);
    }
});

ipcMain.on('remove-tab', (event, data) => {
    console.log('[Main] Received remove-tab request:', JSON.stringify(data));
    
    if (!pythonProcess || !pythonProcess.stdin.writable) {
        console.error('[Main] Python process not available for tab removal');
        return;
    }
    
    const message = {
        type: 'remove_tab',
        browser_id: data.browser_id || 'default',
        tab_id: data.tab_id
    };
    
    try {
        pythonProcess.stdin.write(JSON.stringify(message) + '\n');
    } catch (err) {
        console.error('[Main] Error writing to Python process:', err);
    }
});

// Legacy IPC handlers for compatibility
ipcMain.on('new-tab', (event, url) => {
  mainWindow.webContents.send('create-tab', url);
});

ipcMain.on('close-tab', (event, tabId) => {
  mainWindow.webContents.send('remove-tab', tabId);
});

ipcMain.on('pdf-content', (event, data) => {
  mainWindow.webContents.send('pdf-content', data);
});

// Cleanup on app quit
app.on('before-quit', () => {
    if (pythonProcess) {
        console.log('[Main] Killing Python process...');
        pythonProcess.kill('SIGTERM');
        
        // Force kill after 5 seconds if still running
        setTimeout(() => {
            if (pythonProcess && !pythonProcess.killed) {
                console.log('[Main] Force killing Python process...');
                pythonProcess.kill('SIGKILL');
            }
        }, 5000);
    }
});

console.log('[Main] Electron main process started with enhanced Python backend support');