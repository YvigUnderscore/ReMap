from __future__ import annotations

from .internal_api import create_internal_app
from .job_service import JobService
from .server_controller import ServerController
from .settings_store import SettingsStore


def build_services():
    settings_store = SettingsStore()
    job_service = JobService(settings_store)
    server_controller = ServerController(settings_store)
    return settings_store, job_service, server_controller


def create_app():
    settings_store, job_service, server_controller = build_services()
    return create_internal_app(settings_store, job_service, server_controller)

