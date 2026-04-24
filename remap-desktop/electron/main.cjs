const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const fs = require('fs');
const crypto = require('crypto');
const { spawn } = require('child_process');

const isDev = !app.isPackaged;
const backendState = {
  running: false,
  port: 5000,
  apiKey: '',
  startedAt: '',
  logs: [],
};
let backendProcess = null;

function pushLog(line) {
  if (!line) return;
  backendState.logs.push(`${new Date().toISOString()} ${line}`);
  if (backendState.logs.length > 5000) {
    backendState.logs = backendState.logs.slice(-2000);
  }
}

function repoRoot() {
  if (isDev) {
    return path.resolve(__dirname, '..', '..');
  }
  return path.join(process.resourcesPath, 'backend');
}

function resolvePythonBin(root) {
  const candidates = process.platform === 'win32'
    ? [
        path.join(root, '.venv', 'Scripts', 'python.exe'),
        'py',
        'python',
      ]
    : [
        path.join(root, '.venv', 'bin', 'python3'),
        'python3',
        'python',
      ];

  for (const candidate of candidates) {
    if (candidate.includes(path.sep)) {
      if (fs.existsSync(candidate)) return candidate;
      continue;
    }
    return candidate;
  }
  return process.platform === 'win32' ? 'python' : 'python3';
}

function backendCommand() {
  const root = repoRoot();
  const python = resolvePythonBin(root);
  const script = path.join(root, 'remap_server.py');
  return { python, script, cwd: root };
}

function startBackend() {
  if (backendState.running) return backendStatus();
  const { python, script, cwd } = backendCommand();

  if (!fs.existsSync(script)) {
    throw new Error(`Backend script not found: ${script}`);
  }

  backendState.apiKey = crypto.randomBytes(24).toString('base64url');
  backendState.startedAt = new Date().toISOString();
  backendState.logs = [];

  const args = process.platform === 'win32' && python === 'py'
    ? ['-3', script, '--host', '127.0.0.1', '--port', String(backendState.port), '--api-key', backendState.apiKey]
    : [script, '--host', '127.0.0.1', '--port', String(backendState.port), '--api-key', backendState.apiKey];

  backendProcess = spawn(python, args, {
    cwd,
    env: process.env,
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  backendProcess.stdout.on('data', (data) => {
    pushLog(`[backend] ${data.toString().trimEnd()}`);
  });
  backendProcess.stderr.on('data', (data) => {
    pushLog(`[backend:err] ${data.toString().trimEnd()}`);
  });

  backendProcess.on('exit', (code, signal) => {
    pushLog(`[backend] exited code=${code ?? 'null'} signal=${signal ?? 'null'}`);
    backendState.running = false;
    backendProcess = null;
  });

  backendState.running = true;
  pushLog(`[backend] started with ${python} ${args.join(' ')}`);
  return backendStatus();
}

function stopBackend() {
  if (!backendProcess || !backendState.running) {
    backendState.running = false;
    return backendStatus();
  }

  backendProcess.kill();
  backendState.running = false;
  pushLog('[backend] stop requested');
  return backendStatus();
}

function backendStatus() {
  return {
    running: backendState.running,
    port: backendState.port,
    apiKey: backendState.apiKey,
    startedAt: backendState.startedAt,
    baseUrl: `http://127.0.0.1:${backendState.port}/api/v1`,
  };
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 700,
    backgroundColor: '#0f1220',
    webPreferences: {
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.cjs'),
    },
  });

  if (isDev) {
    win.loadURL('http://localhost:5173');
  } else {
    win.loadFile(path.join(__dirname, '..', 'dist', 'index.html'));
  }
}

ipcMain.handle('runtime-info', async () => ({
  platform: process.platform,
  isDev,
  ...backendStatus(),
}));
ipcMain.handle('backend-start', async () => startBackend());
ipcMain.handle('backend-stop', async () => stopBackend());
ipcMain.handle('backend-status', async () => backendStatus());
ipcMain.handle('backend-logs', async () => ({ logs: backendState.logs.slice(-1000) }));

app.whenReady().then(() => {
  startBackend();
  createWindow();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  stopBackend();
  if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => {
  stopBackend();
});
