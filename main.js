const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { PythonShell } = require('python-shell');
const Store = require('electron-store');

const store = new Store();

let mainWindow;
let pythonProcess;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      webviewTag: true
    }
  });

  mainWindow.loadFile('index.html');
  
  // Start Python backend
  startPythonBackend();
}

function startPythonBackend() {
  const options = {
    mode: 'text',
    pythonPath: 'python3',
    pythonOptions: ['-u'],
    scriptPath: path.join(__dirname, 'ai_backend'),
  };

  pythonProcess = new PythonShell('app.py', options);

  pythonProcess.on('message', function (message) {
    mainWindow.webContents.send('python-message', message);
  });

  pythonProcess.on('error', function (err) {
    console.error('Python Error:', err);
  });

  // Print all stdout and stderr from Python backend to terminal
  if (pythonProcess && pythonProcess.childProcess) {
    pythonProcess.childProcess.stdout.on('data', (data) => {
      process.stdout.write(`[PYTHON STDOUT] ${data}`);
    });
    pythonProcess.childProcess.stderr.on('data', (data) => {
      process.stderr.write(`[PYTHON STDERR] ${data}`);
    });
  }
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

// IPC handlers for tab management
ipcMain.on('new-tab', (event, url) => {
  mainWindow.webContents.send('create-tab', url);
});

ipcMain.on('close-tab', (event, tabId) => {
  mainWindow.webContents.send('remove-tab', tabId);
});

ipcMain.on('search-tabs', (event, query) => {
  pythonProcess.send(JSON.stringify({
    type: 'search',
    query: query
  }));
});

// Add IPC handler for update-tab
ipcMain.on('update-tab', (event, tabData) => {
  console.log('[Main] Forwarding update-tab to Python:', tabData);
  pythonProcess.send(JSON.stringify({
    type: 'update_tab',
    ...tabData
  }));
}); 