from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from threading import RLock
from typing import Any

from .models import (
    SETTINGS_SCHEMA_VERSION,
    AppSettings,
    ProcessingJobRequest,
    ServerConfig,
    default_mapper_type,
    default_worker_count,
)


ROOT_DIR = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT_DIR / "backend_state"
SETTINGS_PATH = STATE_DIR / "settings.json"


class SettingsStore:
    def __init__(self, path: Path | None = None):
        self.path = path or SETTINGS_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._settings = self._load()

    def _load(self) -> AppSettings:
        if not self.path.exists():
            settings = AppSettings()
            self._save_unlocked(settings)
            return settings
        with self.path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        settings = AppSettings.from_payload(raw)
        if self._migrate_unlocked(settings, raw):
            self._save_unlocked(settings)
        return settings

    def _migrate_unlocked(self, settings: AppSettings, raw: dict[str, Any]) -> bool:
        if settings.schema_version >= SETTINGS_SCHEMA_VERSION:
            return False

        raw_defaults = raw.get("defaults") or {}
        if raw_defaults.get("num_workers") in (None, 16):
            settings.defaults.num_workers = default_worker_count()

        preferred_mapper = default_mapper_type()
        if preferred_mapper == "GLOMAP" and raw_defaults.get("mapper_type") in (None, "COLMAP"):
            settings.defaults.mapper_type = "GLOMAP"

        settings.schema_version = SETTINGS_SCHEMA_VERSION
        return True

    def _save_unlocked(self, settings: AppSettings) -> None:
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump(settings.to_dict(), fh, indent=2)

    def get(self) -> AppSettings:
        with self._lock:
            return AppSettings.from_payload(self._settings.to_dict())

    def update(self, payload: dict[str, Any]) -> AppSettings:
        with self._lock:
            current = self._settings
            if "theme" in payload:
                current.theme = str(payload["theme"]).strip() or current.theme
            if "defaults" in payload:
                current.defaults = ProcessingJobRequest.from_payload(payload["defaults"])
            if "server" in payload:
                merged_server = current.server.to_dict()
                merged_server.update(payload["server"] or {})
                current.server = ServerConfig.from_payload(merged_server)
            self._save_unlocked(current)
            return AppSettings.from_payload(current.to_dict())

    def update_server(self, payload: dict[str, Any]) -> AppSettings:
        return self.update({"server": payload})

    def reset(self) -> AppSettings:
        with self._lock:
            self._settings = AppSettings()
            self._save_unlocked(self._settings)
            return AppSettings.from_payload(self._settings.to_dict())
