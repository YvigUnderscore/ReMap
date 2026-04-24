const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('remapDesktop', {
  getRuntimeInfo: () => ipcRenderer.invoke('runtime-info'),
  startBackend: () => ipcRenderer.invoke('backend-start'),
  stopBackend: () => ipcRenderer.invoke('backend-stop'),
  getBackendStatus: () => ipcRenderer.invoke('backend-status'),
  getBackendLogs: () => ipcRenderer.invoke('backend-logs'),
});
