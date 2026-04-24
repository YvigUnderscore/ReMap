import type { RuntimeInfo } from './types'

declare global {
  interface Window {
    remapDesktop?: {
      getRuntimeInfo: () => Promise<RuntimeInfo>
      startBackend: () => Promise<RuntimeInfo>
      stopBackend: () => Promise<RuntimeInfo>
      getBackendStatus: () => Promise<RuntimeInfo>
      getBackendLogs: () => Promise<{ logs: string[] }>
    }
  }
}

export {}
