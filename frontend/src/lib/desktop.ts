import { invoke } from "@tauri-apps/api/core";
import { getCurrentWebview } from "@tauri-apps/api/webview";

async function safeInvoke<T>(command: string): Promise<T | null> {
  try {
    return await invoke<T>(command);
  } catch {
    return null;
  }
}

export const desktop = {
  pickVideoFiles: () => safeInvoke<string[]>("pick_video_files"),
  pickDirectory: () => safeInvoke<string | null>("pick_directory"),
  pickDirectories: () => safeInvoke<string[]>("pick_directories"),
  pickOcioFile: () => safeInvoke<string | null>("pick_ocio_file"),
  revealPath: async (path: string) => {
    try {
      await invoke("reveal_path", { path });
    } catch {
      return null;
    }
    return null;
  },
  onFileDrop: async (handler: (paths: string[]) => void) => {
    try {
      return await getCurrentWebview().onDragDropEvent((event) => {
        if (event.payload.type === "drop") {
          handler(event.payload.paths);
        }
      });
    } catch {
      return () => undefined;
    }
  },
};
