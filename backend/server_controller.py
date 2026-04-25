from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import json
from threading import RLock

from .models import ServerConfig
from .settings_store import SettingsStore


ROOT_DIR = Path(__file__).resolve().parent.parent
REMOTE_SERVER_SCRIPT = ROOT_DIR / "remap_server.py"


class ServerController:
    def __init__(self, settings_store: SettingsStore):
        self.settings_store = settings_store
        self._lock = RLock()
        self._process: subprocess.Popen | None = None
        self._log_file = Path(tempfile.gettempdir()) / "remap_desktop_remote_server.log"
        self._log_handle = None

    def get_state(self) -> dict:
        settings = self.settings_store.get()
        config = settings.server
        health = self.check_health()
        jobs = self.fetch_remote_jobs(config) if health["reachable"] else []
        return {
            "config": config.to_dict(),
            "running": self.is_running(),
            "log_path": str(self._log_file),
            "health": health,
            "remote_jobs": jobs,
        }

    def update_config(self, payload: dict) -> dict:
        settings = self.settings_store.update_server(payload)
        return settings.server.to_dict()

    def is_running(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None

    def start(self) -> dict:
        settings = self.settings_store.get()
        config = settings.server
        with self._lock:
            if self.is_running():
                return self.get_state()
            self._log_handle = self._log_file.open("w", encoding="utf-8")
            command = [
                sys.executable,
                str(REMOTE_SERVER_SCRIPT),
                "--host",
                config.host,
                "--port",
                str(config.port),
                "--api-key",
                config.api_key,
            ]
            if config.output_dir:
                command.extend(["--output-dir", config.output_dir])
            self._process = subprocess.Popen(
                command,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                cwd=str(ROOT_DIR),
            )
        return self.get_state()

    def stop(self) -> dict:
        with self._lock:
            process = self._process
            self._process = None
            log_handle = self._log_handle
            self._log_handle = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        if log_handle is not None:
            log_handle.close()
        return self.get_state()

    def check_health(self) -> dict:
        settings = self.settings_store.get()
        config = settings.server
        url = f"http://127.0.0.1:{config.port}/api/v1/health"
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                data = json.loads(response.read().decode("utf-8"))
            return {"reachable": True, "url": url, "payload": data}
        except Exception as exc:
            return {"reachable": False, "url": url, "error": str(exc)}

    def fetch_remote_jobs(self, config: ServerConfig | None = None) -> list[dict]:
        config = config or self.settings_store.get().server
        request = urllib.request.Request(
            f"http://127.0.0.1:{config.port}/api/v1/jobs",
            headers={"Authorization": f"Bearer {config.api_key}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=2) as response:
                data = json.loads(response.read().decode("utf-8"))
            return data.get("jobs", [])
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
            return []
