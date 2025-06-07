# Better Browser

A modern web browser built with Electron.js and Python, featuring AI-powered tab management and search capabilities. This browser can efficiently handle 200+ tabs in a single session using semantic search and smart tab organization.

## Features

- Modern, clean user interface
- AI-powered tab search and management
- Semantic search across all open tabs
- Smart tab organization (next todo)
<!-- - Real-time tab updates and synchronization -->

## Project Structure

```
ai-browser/
├── ai_backend/        # Python backend for AI functionality
│   └── app.py        # Main Python application
├── main.js           # Electron main process
├── index.html        # Browser UI
├── package.json      # Node.js dependencies
└── requirements.txt  # Python dependencies
```

## Prerequisites

- Node.js (v14 or higher)
- Python 3.8 or higher
- pip (Python package manager)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd better-browse
```

2. Install Node.js dependencies:
```bash
npm install
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

**No need to run the backend manually!**

The Python backend (`ai_backend/app.py`) is started automatically by Electron when you launch the app.

To start the application:
```bash
npm start
```

This will launch the Electron app and automatically start the backend for AI-powered tab management and search.

## Usage

- **New Tab**: Click the '+' button in the toolbar
- **Search Tabs**: Use the search bar in the toolbar or sidebar to search across all open tabs
- **URL Navigation**: Enter URLs in the address bar and press Enter
- **Tab Management**: 
  - Click on tabs to switch between them
  - Click the '×' button to close tabs
  - Use the search bar or sidebar to find specific tabs

## AI Features

The browser uses advanced AI techniques for:
- Semantic search across tabs (using tab content, title, and URL)
- Smart tab organization
- Content-based tab recommendations
- Efficient memory management for large numbers of tabs

## Development

To run the application in development mode:
```bash
npm run dev
```

## License

MIT License 