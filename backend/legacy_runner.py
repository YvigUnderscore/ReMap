from __future__ import annotations

from pathlib import Path
from threading import Event
from types import MethodType, ModuleType, SimpleNamespace
from typing import Callable
import importlib.util
import shutil
import sys
import time

from .models import ProcessingJobRequest, legacy_input_mode_label
from .probe_service import probe_rescan_dataset, probe_video


ROOT_DIR = Path(__file__).resolve().parent.parent
LEGACY_GUI_PATH = ROOT_DIR / "ReMap-GUI.py"
_LEGACY_MODULE: ModuleType | None = None


def _load_legacy_module() -> ModuleType:
    global _LEGACY_MODULE
    if _LEGACY_MODULE is not None:
        return _LEGACY_MODULE
    spec = importlib.util.spec_from_file_location("remap_legacy_gui", LEGACY_GUI_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load legacy GUI module from {LEGACY_GUI_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _LEGACY_MODULE = module
    return module


class _Var:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


class _Widget:
    def __init__(self):
        self.state = {}

    def configure(self, **kwargs):
        self.state.update(kwargs)


class _ConsoleWidget(_Widget):
    def __init__(self, callback: Callable[[str], None]):
        super().__init__()
        self.callback = callback
        self.lines: list[str] = []

    def after(self, delay_ms, callback, *args):
        callback(*args)

    def insert(self, _where, message):
        text = str(message).rstrip("\n")
        if not text:
            return
        self.lines.append(text)
        self.callback(text)

    def see(self, _where):
        return None


class _StepLabel(_Widget):
    def __init__(self, callback: Callable[[str], None] | None = None):
        super().__init__()
        self.callback = callback

    def configure(self, **kwargs):
        super().configure(**kwargs)
        text = kwargs.get("text")
        if text is not None and self.callback:
            self.callback(str(text))


class _ProgressBar(_Widget):
    def __init__(self, callback: Callable[[float], None] | None = None):
        super().__init__()
        self.callback = callback
        self.value = 0.0

    def set(self, value):
        self.value = value
        if self.callback:
            self.callback(float(value))


def _proxy_after(proxy, _delay_ms, callback, *args):
    callback(*args)


def _proxy_log_safe(proxy, message):
    proxy._runner._emit_log(str(message))


def _proxy_log(proxy, message):
    proxy._runner._emit_log(str(message))


def _proxy_log_tagged(proxy, tag, message):
    proxy._runner._emit_log(f"{tag} {message}")


def _proxy_set_step(proxy, index, total=None):
    total = total or len(proxy.STEPS) or 1
    name = proxy.STEPS[index] if index < len(proxy.STEPS) else "Done"
    proxy._runner._on_step(index=index, total=total, name=name)


def _proxy_check_cancelled(proxy):
    if proxy._cancelled:
        raise proxy._module.CancelledError("Processing cancelled by user")
    while not proxy._pause_event.is_set():
        if proxy._cancelled:
            raise proxy._module.CancelledError("Processing cancelled by user")
        proxy._pause_event.wait(timeout=0.25)


def _proxy_finish(proxy, success=True, cancelled=False):
    proxy._processing = False
    proxy._runner._on_finish(success=success, cancelled=cancelled)


class LegacyGuiPipelineRunner:
    def __init__(
        self,
        request: ProcessingJobRequest,
        log_callback: Callable[[str], None],
        step_callback: Callable[[int, int, str], None],
        detail_callback: Callable[[str, int | None], None],
        finish_callback: Callable[[bool, bool], None],
    ):
        self.request = request
        self.log_callback = log_callback
        self.step_callback = step_callback
        self.detail_callback = detail_callback
        self.finish_callback = finish_callback
        self._last_detail_text = ""
        self._last_detail_time = 0.0
        self._last_progress = 0
        self._module = _load_legacy_module()
        self._proxy = self._build_proxy()

    def _probe_video_infos(self) -> list[dict]:
        try:
            return [probe_video(path) for path in self.request.input_paths]
        except Exception as exc:
            self._emit_log(f"[WARN] Could not probe video metadata for execution: {exc}")
            return []

    def _probe_rescan_infos(self) -> list[dict]:
        try:
            return [probe_rescan_dataset(path) for path in self.request.input_paths]
        except Exception as exc:
            self._emit_log(f"[WARN] Could not probe Rescan metadata for execution: {exc}")
            return []

    def _build_proxy(self):
        proxy = SimpleNamespace()
        proxy._runner = self
        proxy._module = self._module
        proxy.STEPS = list(self._module.SfMApp.STEPS)
        proxy.STEPS_STRAY_A = list(self._module.SfMApp.STEPS_STRAY_A)
        proxy.STEPS_STRAY_B = list(self._module.SfMApp.STEPS_STRAY_B)
        proxy._cancelled = False
        proxy._pause_event = Event()
        proxy._pause_event.set()
        proxy._processing = True
        proxy.has_ocio_lib = bool(getattr(self._module, "HAS_OCIO", False))
        proxy.has_glomap = shutil.which("glomap") is not None
        proxy.after = MethodType(_proxy_after, proxy)
        proxy._log_safe = MethodType(_proxy_log_safe, proxy)
        proxy._log = MethodType(_proxy_log, proxy)
        proxy._log_tagged = MethodType(_proxy_log_tagged, proxy)
        proxy._set_step = MethodType(_proxy_set_step, proxy)
        proxy._check_cancelled = MethodType(_proxy_check_cancelled, proxy)
        proxy._finish = MethodType(_proxy_finish, proxy)
        proxy._run_ffmpeg_streamed = MethodType(self._module.SfMApp._run_ffmpeg_streamed, proxy)
        proxy._run_process = MethodType(self._module.SfMApp._run_process, proxy)

        proxy.console = _ConsoleWidget(self._emit_log)
        proxy.step_label = _StepLabel(self._on_status_text)
        proxy.progress_bar = _ProgressBar(self._on_progress_value)
        proxy.btn_run = _Widget()
        proxy.btn_cancel = _Widget()

        input_mode = legacy_input_mode_label(self.request.input_mode)
        proxy.input_mode = _Var(input_mode)
        proxy.video_paths = [Path(item) for item in self.request.input_paths] if self.request.input_mode == "video" else []
        proxy.stray_paths = [Path(item) for item in self.request.input_paths] if self.request.input_mode == "rescan" else []

        primary_input = self.request.input_paths[0] if self.request.input_paths else ""
        proxy.video_path = _Var(primary_input)
        proxy.output_path = _Var(self.request.output_path)
        proxy.fps_extract = _Var(self.request.fps_extract)
        proxy.force_16bit = _Var(self.request.force_16bit)
        proxy.camera_model = _Var(self.request.camera_model)
        proxy.feature_type = _Var(self.request.feature_type)
        proxy.matcher_type = _Var(self.request.matcher_type)
        proxy.max_keypoints = _Var(str(self.request.max_keypoints))
        proxy.pairing_mode = _Var(self.request.pairing_mode)
        proxy.mapper_type = _Var(self.request.mapper_type)
        proxy.stray_approach = _Var(self.request.stray_approach)
        proxy.stray_confidence = _Var(self.request.stray_confidence)
        proxy.stray_depth_subsample = _Var(self.request.stray_depth_subsample)
        proxy.stray_gen_pointcloud = _Var(self.request.stray_gen_pointcloud)
        proxy.color_enabled = _Var(self.request.color_enabled)
        proxy.color_source = _Var(self.request.color_source)
        proxy.color_dest = _Var(self.request.color_dest)
        proxy.detected_color_profile = _Var(self.request.detected_color_profile)
        proxy.ocio_path = _Var(self.request.ocio_path)
        proxy.ocio_in_cs = _Var(self.request.ocio_in_cs)
        proxy.ocio_out_cs = _Var(self.request.ocio_out_cs)
        proxy.use_acescg_exr = _Var(self.request.use_acescg_exr)
        proxy.num_workers = _Var(self.request.num_workers)
        proxy.server_port = _Var(self.request.server_port)
        proxy.server_key_var = _Var(self.request.server_api_key)
        proxy._video_infos = self._probe_video_infos() if self.request.input_mode == "video" else []
        proxy._rescan_infos = self._probe_rescan_infos() if self.request.input_mode == "rescan" else []
        return proxy

    def _emit_log(self, message: str) -> None:
        cleaned = str(message).strip("\n")
        if cleaned:
            self.log_callback(cleaned)

    def _on_step(self, index: int, total: int, name: str) -> None:
        self.step_callback(index, total, name)

    def _on_status_text(self, text: str) -> None:
        cleaned = str(text).strip()
        if not cleaned:
            return
        now = time.monotonic()
        if cleaned == self._last_detail_text and now - self._last_detail_time < 1.0:
            return
        if now - self._last_detail_time < 0.5 and cleaned != "Done":
            return
        self._last_detail_text = cleaned
        self._last_detail_time = now
        self.detail_callback(cleaned, self._last_progress or None)

    def _on_progress_value(self, value: float) -> None:
        progress = int(max(0.0, min(1.0, float(value))) * 100)
        if progress == self._last_progress:
            return
        self._last_progress = progress
        if self._last_detail_text:
            self.detail_callback(self._last_detail_text, progress)

    def _on_finish(self, success: bool, cancelled: bool) -> None:
        self.finish_callback(success, cancelled)

    def cancel(self) -> None:
        self._proxy._cancelled = True
        self._proxy._pause_event.set()

    def pause(self) -> None:
        self._proxy._pause_event.clear()

    def resume(self) -> None:
        self._proxy._pause_event.set()

    def run(self) -> None:
        self._proxy._run_process()
