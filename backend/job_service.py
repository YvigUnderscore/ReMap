from __future__ import annotations

from collections import deque
import json
from pathlib import Path
from threading import Condition, RLock, Thread
import time
from typing import Any, Iterator
from uuid import uuid4

from .bundle_postprocess import normalize_final_bundle
from .estimate_service import estimate_request
from .legacy_runner import LegacyGuiPipelineRunner
from .models import JobDetail, JobLogEvent, ProcessingJobRequest, utc_now_iso
from .pipeline_state import (
    CheckpointStore,
    cache_status,
    clear_cache,
    enforce_cache_limit,
    request_fingerprint,
    restore_cache,
    store_cache,
    touch_manifest,
)
from .reconstruction_preview import build_reconstruction_preview
from .settings_store import STATE_DIR, SettingsStore


TERMINAL_STATUSES = {"completed", "failed", "cancelled", "interrupted"}
JOB_STORE_PATH = STATE_DIR / "jobs.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}
EXR_EXTENSIONS = {".exr"}
RECONSTRUCTION_EXTENSIONS = {".bin", ".txt", ".ply", ".json", ".nvm", ".glb", ".obj"}


class JobService:
    def __init__(self, settings_store: SettingsStore, store_path: Path | None = None):
        self.settings_store = settings_store
        self.store_path = store_path or JOB_STORE_PATH
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, JobDetail] = {}
        self._runners: dict[str, LegacyGuiPipelineRunner] = {}
        self._queue: deque[str] = deque()
        self._lock = RLock()
        self._condition = Condition(self._lock)
        self._next_log_id = 1
        self._load_jobs()
        self._worker = Thread(target=self._queue_worker, daemon=True)
        self._worker.start()

    def _load_jobs(self) -> None:
        if not self.store_path.exists():
            return
        dirty = False
        try:
            with self.store_path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except Exception:
            return

        raw_jobs = raw.get("jobs", raw if isinstance(raw, list) else [])
        max_log_id = 0
        for item in raw_jobs:
            try:
                logs = [
                    JobLogEvent(
                        id=int(event.get("id", 0)),
                        timestamp=str(event.get("timestamp") or utc_now_iso()),
                        level=str(event.get("level") or "info"),
                        message=str(event.get("message") or ""),
                    )
                    for event in item.get("logs", [])
                ]
                max_log_id = max(max_log_id, *(event.id for event in logs), 0)
                job = JobDetail(
                    job_id=str(item["job_id"]),
                    status=str(item.get("status") or "failed"),
                    progress=int(item.get("progress") or 0),
                    current_step=str(item.get("current_step") or ""),
                    created_at=str(item.get("created_at") or utc_now_iso()),
                    updated_at=str(item.get("updated_at") or utc_now_iso()),
                    label=str(item.get("label") or ""),
                    output_path=str(item.get("output_path") or ""),
                    input_mode=str(item.get("input_mode") or "video"),
                    error=item.get("error"),
                    queue_position=item.get("queue_position"),
                    progress_note=str(item.get("progress_note") or ""),
                    eta_seconds=item.get("eta_seconds"),
                    request=dict(item.get("request") or {}),
                    logs=logs,
                )
            except Exception:
                continue

            if job.status == "processing":
                job.status = "interrupted"
                job.current_step = "Interrupted by app shutdown"
                job.error = job.error or "The backend stopped while this job was running."
                job.updated_at = utc_now_iso()
                dirty = True
            self._jobs[job.job_id] = job

        self._next_log_id = max_log_id + 1
        for job in sorted(self._jobs.values(), key=lambda item: item.created_at):
            if job.status in {"queued", "paused"}:
                self._queue.append(job.job_id)
        self._recompute_queue_positions()
        if dirty:
            self._save_jobs_unlocked()

    def _save_jobs_unlocked(self) -> None:
        payload = {
            "schema_version": 1,
            "updated_at": utc_now_iso(),
            "jobs": [job.to_dict() for job in sorted(self._jobs.values(), key=lambda item: item.created_at)],
        }
        tmp_path = self.store_path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        tmp_path.replace(self.store_path)

    def create_job(self, payload: dict[str, Any]) -> JobDetail:
        settings = self.settings_store.get()
        merged = settings.defaults.to_dict()
        merged.update(payload or {})
        request = ProcessingJobRequest.from_payload(merged)
        if not request.output_path:
            raise ValueError("Missing output_path")
        if not request.input_paths:
            raise ValueError("Missing input_paths")
        self._remember_paths(request)

        job_id = uuid4().hex[:12]
        now = utc_now_iso()
        detail = JobDetail(
            job_id=job_id,
            status="queued",
            progress=0,
            current_step="Queued",
            created_at=now,
            updated_at=now,
            label=request.label or f"{request.input_mode.capitalize()} job",
            output_path=request.output_path,
            input_mode=request.input_mode,
            error=None,
            progress_note="Waiting for the queue",
            eta_seconds=None,
            request=request.to_dict(),
            logs=[],
        )
        with self._lock:
            self._jobs[job_id] = detail
            self._queue.append(job_id)
            self._recompute_queue_positions()
            self._save_jobs_unlocked()
            self._condition.notify_all()
        return self.get_job(job_id)

    def create_jobs_batch(self, payloads: list[dict[str, Any]]) -> list[JobDetail]:
        created = []
        for payload in payloads:
            created.append(self.create_job(payload))
        return created

    def _remember_paths(self, request: ProcessingJobRequest) -> None:
        try:
            settings = self.settings_store.get()
            recent_inputs = list(request.input_paths) + [
                item for item in settings.recent_inputs if item not in request.input_paths
            ]
            recent_outputs = [request.output_path] + [
                item for item in settings.recent_outputs if item != request.output_path
            ]
            self.settings_store.update(
                {
                    "recent_inputs": recent_inputs[:20],
                    "recent_outputs": recent_outputs[:20],
                }
            )
        except Exception:
            pass

    def _queue_worker(self) -> None:
        while True:
            with self._condition:
                while True:
                    job_id = self._next_runnable_job_id_unlocked()
                    if job_id is not None:
                        break
                    self._condition.wait(timeout=1.0)
                job = self._jobs[job_id]
                request = ProcessingJobRequest.from_payload(job.request)
            self._run_job(job_id, request)
            with self._condition:
                if job_id in self._queue:
                    self._queue.remove(job_id)
                self._recompute_queue_positions()
                self._save_jobs_unlocked()
                self._condition.notify_all()

    def _next_runnable_job_id_unlocked(self) -> str | None:
        for job_id in list(self._queue):
            job = self._jobs.get(job_id)
            if job is None or job.status == "cancelled":
                self._queue.remove(job_id)
                self._recompute_queue_positions()
                continue
            if job.status == "queued":
                return job_id
        return None

    def _recompute_queue_positions(self) -> None:
        queued_lookup = {job_id: index + 1 for index, job_id in enumerate(self._queue)}
        for job_id, job in self._jobs.items():
            job.queue_position = queued_lookup.get(job_id) if job.status in {"queued", "processing", "paused"} else None

    def _run_job(self, job_id: str, request: ProcessingJobRequest) -> None:
        started_at = time.monotonic()
        settings = self.settings_store.get()
        fingerprint = request_fingerprint(request)
        checkpoint = CheckpointStore(Path(request.output_path), fingerprint, request)
        estimate = estimate_request(request.to_dict(), self.snapshot_jobs())
        cache_result = {"hit": False, "restored": []}
        if not request.skip_existing:
            self._clear_reusable_artifacts(Path(request.output_path))
        if settings.cache_enabled:
            cache_result = restore_cache(request, fingerprint, Path(request.output_path))
        touch_manifest(
            Path(request.output_path),
            "fingerprint",
            {"fingerprint": fingerprint, "cache": cache_result},
        )
        checkpoint.mark("prepare", "completed", detail="Fingerprint and cache preflight complete")
        self._update_job(
            job_id,
            status="processing",
            current_step="Initialising",
            progress=1,
            error=None,
            queue_position=1,
            progress_note="Starting pipeline",
            eta_seconds=estimate.get("estimated_seconds"),
        )
        self._append_log(job_id, "info", "Legacy pipeline bridge started")
        if cache_result.get("hit"):
            self._append_log(
                job_id,
                "info",
                f"Global cache hit: restored {', '.join(cache_result.get('restored', []))}",
            )

        last_checkpoint_step = {"value": "prepare"}

        def on_log(message: str) -> None:
            self._append_log(job_id, "info", message)

        def on_step(index: int, total: int, name: str) -> None:
            progress = min(95, max(1, int((index / max(total, 1)) * 100)))
            step_name = self._checkpoint_step_name(name, index)
            previous = last_checkpoint_step.get("value")
            if previous and previous != step_name:
                checkpoint.mark(previous, "completed", detail=f"Advanced to {name}")
            last_checkpoint_step["value"] = step_name
            checkpoint.mark(step_name, "running", detail=name)
            self._update_job(
                job_id,
                current_step=name,
                progress=progress,
                progress_note=self._format_eta_note("", started_at, progress, estimate),
                eta_seconds=self._eta_seconds(started_at, progress, estimate),
            )

        def on_detail(note: str, progress: int | None = None) -> None:
            next_progress = progress
            changes: dict[str, Any] = {
                "progress_note": self._format_eta_note(note, started_at, next_progress, estimate),
            }
            if progress is not None:
                next_progress = min(99, max(1, int(progress)))
                changes["progress"] = next_progress
                changes["eta_seconds"] = self._eta_seconds(started_at, next_progress, estimate)
            self._update_job(job_id, **changes)

        def on_finish(success: bool, cancelled: bool) -> None:
            if cancelled:
                checkpoint.mark("bundle", "cancelled", detail="Job cancelled")
                self._update_job(job_id, status="cancelled", current_step="Cancelled", progress=100, progress_note="", eta_seconds=None)
                self._append_log(job_id, "warning", "Job cancelled")
            elif success:
                checkpoint.mark(last_checkpoint_step.get("value", "sfm"), "completed", detail="Pipeline step completed")
                normalize_final_bundle(
                    request.output_path,
                    keep_srgb_png=request.keep_srgb_png and request.input_mode != "rescan",
                    use_acescg_exr=request.use_acescg_exr,
                )
                checkpoint.mark("bundle", "completed", detail="Final bundle normalized")
                if settings.cache_enabled:
                    stored = store_cache(request, fingerprint, Path(request.output_path))
                    enforce_cache_limit(settings.cache_max_size_gb)
                    self._append_log(job_id, "info", f"Global cache stored: {', '.join(stored.get('stored', [])) or 'nothing new'}")
                self._update_job(job_id, status="completed", current_step="Done", progress=100, progress_note="", eta_seconds=None)
                self._append_log(job_id, "info", "Job completed successfully")
            else:
                checkpoint.mark("bundle", "failed", detail="Legacy runner returned failure")
                self._update_job(job_id, status="failed", current_step="Error", progress=100, progress_note="", eta_seconds=None)

        runner = LegacyGuiPipelineRunner(
            request=request,
            log_callback=on_log,
            step_callback=on_step,
            detail_callback=on_detail,
            finish_callback=on_finish,
        )
        with self._lock:
            self._runners[job_id] = runner

        try:
            runner.run()
            with self._lock:
                job = self._jobs.get(job_id)
                if job and job.status == "processing":
                    checkpoint.mark(last_checkpoint_step.get("value", "sfm"), "completed", detail="Pipeline step completed")
                    normalize_final_bundle(
                        request.output_path,
                        keep_srgb_png=request.keep_srgb_png and request.input_mode != "rescan",
                        use_acescg_exr=request.use_acescg_exr,
                    )
                    checkpoint.mark("bundle", "completed", detail="Final bundle normalized")
                    if settings.cache_enabled:
                        stored = store_cache(request, fingerprint, Path(request.output_path))
                        enforce_cache_limit(settings.cache_max_size_gb)
                        self._append_log(job_id, "info", f"Global cache stored: {', '.join(stored.get('stored', [])) or 'nothing new'}")
                    self._update_job(job_id, status="completed", current_step="Done", progress=100, progress_note="", eta_seconds=None)
        except Exception as exc:
            self._append_log(job_id, "error", f"Job failed: {exc}")
            checkpoint.mark("bundle", "failed", detail=str(exc))
            self._update_job(job_id, status="failed", current_step="Error", progress=100, error=str(exc), progress_note="", eta_seconds=None)
        finally:
            with self._lock:
                self._runners.pop(job_id, None)
                self._condition.notify_all()

    def _checkpoint_step_name(self, name: str, index: int) -> str:
        lower = name.lower()
        if "color" in lower or "ocio" in lower:
            return "color"
        if "feature" in lower:
            return "features"
        if "pair" in lower:
            return "pairs"
        if "match" in lower:
            return "matching"
        if "sfm" in lower or "triang" in lower or "reconstruct" in lower:
            return "sfm"
        return ["prepare", "color", "features", "pairs", "matching", "sfm", "bundle"][min(index, 6)]

    def _clear_reusable_artifacts(self, output_path: Path) -> None:
        generated_dirs = [output_path / "images", output_path / "hloc_outputs"]
        for directory in generated_dirs:
            if directory.exists() and directory.is_dir():
                for item in directory.iterdir():
                    if item.is_file() and item.suffix.lower() in {".h5", ".txt", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr"}:
                        try:
                            item.unlink()
                        except OSError:
                            pass

    def _eta_seconds(self, started_at: float, progress: int | None, estimate: dict[str, Any]) -> int | None:
        if not progress or progress <= 1:
            return int(estimate.get("estimated_seconds") or 0) or None
        elapsed = time.monotonic() - started_at
        projected = elapsed * (100 - progress) / max(progress, 1)
        history_remaining = max(0.0, float(estimate.get("estimated_seconds") or 0) - elapsed)
        if history_remaining:
            projected = (projected + history_remaining) / 2
        return int(max(0, projected))

    def _format_eta_note(self, note: str, started_at: float, progress: int | None, estimate: dict[str, Any]) -> str:
        eta = self._eta_seconds(started_at, progress, estimate)
        if eta is None:
            return note
        minutes = eta // 60
        seconds = eta % 60
        eta_text = f"ETA {minutes}m {seconds:02d}s" if minutes else f"ETA {seconds}s"
        return f"{note} - {eta_text}" if note else eta_text

    def _append_log(self, job_id: str, level: str, message: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            event = JobLogEvent(
                id=self._next_log_id,
                timestamp=utc_now_iso(),
                level=level,
                message=message,
            )
            self._next_log_id += 1
            job.logs.append(event)
            job.updated_at = utc_now_iso()
            self._recompute_queue_positions()
            self._save_jobs_unlocked()
            self._condition.notify_all()

    def _update_job(self, job_id: str, **changes) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in changes.items():
                setattr(job, key, value)
            job.updated_at = utc_now_iso()
            self._recompute_queue_positions()
            self._save_jobs_unlocked()
            self._condition.notify_all()

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)
            return [job.summary().to_dict() for job in jobs]

    def snapshot_jobs(self) -> list[JobDetail]:
        with self._lock:
            return [
                JobDetail(
                    job_id=job.job_id,
                    status=job.status,
                    progress=job.progress,
                    current_step=job.current_step,
                    created_at=job.created_at,
                    updated_at=job.updated_at,
                    label=job.label,
                    output_path=job.output_path,
                    input_mode=job.input_mode,
                    error=job.error,
                    queue_position=job.queue_position,
                    progress_note=job.progress_note,
                    eta_seconds=job.eta_seconds,
                    request=dict(job.request),
                    logs=list(job.logs),
                )
                for job in self._jobs.values()
            ]

    def get_job(self, job_id: str) -> JobDetail:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            cloned = JobDetail(
                job_id=job.job_id,
                status=job.status,
                progress=job.progress,
                current_step=job.current_step,
                created_at=job.created_at,
                updated_at=job.updated_at,
                label=job.label,
                output_path=job.output_path,
                input_mode=job.input_mode,
                error=job.error,
                queue_position=job.queue_position,
                progress_note=job.progress_note,
                eta_seconds=job.eta_seconds,
                request=dict(job.request),
                logs=list(job.logs),
            )
            return cloned

    def cancel_job(self, job_id: str) -> JobDetail:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.status in {"queued", "paused"} and job_id in self._queue and job_id not in self._runners:
                self._queue.remove(job_id)
                job.status = "cancelled"
                job.current_step = "Cancelled"
                job.progress = 100
                job.progress_note = ""
                job.eta_seconds = None
                job.updated_at = utc_now_iso()
                self._recompute_queue_positions()
                self._save_jobs_unlocked()
                self._condition.notify_all()
                return self.get_job(job_id)
            runner = self._runners.get(job_id)
            if runner is None and job.status not in {"queued", "processing", "paused"}:
                return self.get_job(job_id)
            job.current_step = "Cancelling"
            job.progress_note = "Cancellation requested"
            job.eta_seconds = None
            job.updated_at = utc_now_iso()
            self._save_jobs_unlocked()
            self._condition.notify_all()
        if runner is not None:
            runner.cancel()
        self._append_log(job_id, "warning", "Cancellation requested")
        return self.get_job(job_id)

    def pause_job(self, job_id: str) -> JobDetail:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.status not in {"queued", "processing"}:
                return self.get_job(job_id)
            runner = self._runners.get(job_id)
            if runner is not None:
                runner.pause()
            job.status = "paused"
            if job.current_step not in {"Queued", "Paused"}:
                job.current_step = f"Paused - {job.current_step}"
            else:
                job.current_step = "Paused"
            job.progress_note = "Paused"
            job.eta_seconds = None
            job.updated_at = utc_now_iso()
            self._recompute_queue_positions()
            self._save_jobs_unlocked()
            self._condition.notify_all()
        self._append_log(job_id, "warning", "Pause requested")
        return self.get_job(job_id)

    def resume_job(self, job_id: str) -> JobDetail:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.status != "paused":
                return self.get_job(job_id)
            runner = self._runners.get(job_id)
            if runner is not None:
                runner.resume()
                job.status = "processing"
                job.current_step = job.current_step.removeprefix("Paused - ") or "Processing"
                job.progress_note = "Resumed"
            else:
                job.status = "queued"
                job.current_step = "Queued"
                job.progress_note = "Waiting for the queue"
            job.eta_seconds = None
            job.updated_at = utc_now_iso()
            self._recompute_queue_positions()
            self._save_jobs_unlocked()
            self._condition.notify_all()
        self._append_log(job_id, "info", "Resume requested")
        return self.get_job(job_id)

    def delete_job(self, job_id: str) -> dict[str, Any]:
        runner: LegacyGuiPipelineRunner | None = None
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            runner = self._runners.get(job_id)
            if job_id in self._queue:
                self._queue.remove(job_id)
            self._jobs.pop(job_id, None)
            self._recompute_queue_positions()
            self._save_jobs_unlocked()
            self._condition.notify_all()
        if runner is not None:
            runner.cancel()
        return {"deleted": True, "job_id": job_id}

    def clear_queued_jobs(self) -> dict[str, Any]:
        removed: list[str] = []
        with self._lock:
            for job_id, job in list(self._jobs.items()):
                if job.status != "processing" and job_id not in self._runners:
                    if job_id in self._queue:
                        self._queue.remove(job_id)
                    self._jobs.pop(job_id, None)
                    removed.append(job_id)
            self._recompute_queue_positions()
            self._save_jobs_unlocked()
            self._condition.notify_all()
        return {"removed": removed, "count": len(removed)}

    def get_artifacts(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            request = dict(job.request)
            output_path = job.output_path

        input_paths = [str(item) for item in request.get("input_paths", [])]
        output = Path(output_path) if output_path else Path()
        output_images = output / "images"
        root_final_bundle = output / f"{output.name}_SfM_Dataset_Output"
        legacy_final_bundle = output / "sparse" / "0" / "models" / "0" / "0"
        final_bundle = root_final_bundle if root_final_bundle.exists() else legacy_final_bundle
        final_images = final_bundle / "images"
        preview_images = final_bundle / "images_srgb_png"

        frame_dirs = [output_images, final_images, preview_images]
        frames = []
        exrs = []
        for directory in frame_dirs:
            collected = self._collect_files(directory, IMAGE_EXTENSIONS | EXR_EXTENSIONS, limit=48)
            for item in collected:
                if item["extension"] in EXR_EXTENSIONS:
                    exrs.append(item)
                else:
                    frames.append(item)

        input_media = []
        for value in input_paths:
            path = Path(value)
            if path.is_dir():
                samples = self._collect_files(path, IMAGE_EXTENSIONS | EXR_EXTENSIONS, limit=24)
                input_media.append(self._path_item(path, "folder", extra={"samples": samples}))
            else:
                input_media.append(self._path_item(path, "video" if path.suffix.lower() in VIDEO_EXTENSIONS else "file"))

        reconstruction_dirs = [
            final_bundle,
            output / "sparse",
            output / "sparse" / "0",
            output / "sparse" / "0" / "models",
            output / "live_reconstruction",
            output / "hloc_outputs",
        ]
        reconstruction = []
        for directory in reconstruction_dirs:
            reconstruction.extend(self._collect_files(directory, RECONSTRUCTION_EXTENSIONS, limit=32))

        return {
            "job_id": job_id,
            "input_paths": input_media,
            "output_path": self._path_item(output, "folder"),
            "frame_dirs": [self._path_item(path, "folder") for path in frame_dirs if path.exists()],
            "frames": frames[:48],
            "exrs": exrs[:48],
            "reconstruction": reconstruction[:64],
            "latest_outputs": self._collect_files(output, IMAGE_EXTENSIONS | EXR_EXTENSIONS | RECONSTRUCTION_EXTENSIONS, limit=64, recursive=True),
        }

    def get_reconstruction_preview(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            output_path = job.output_path
        return build_reconstruction_preview(output_path)

    def cache_status(self) -> dict[str, Any]:
        return cache_status()

    def clear_cache(self) -> dict[str, Any]:
        return clear_cache()

    def _path_item(self, path: Path, kind: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        exists = path.exists()
        stat = path.stat() if exists else None
        item = {
            "path": str(path),
            "name": path.name or str(path),
            "kind": kind,
            "exists": exists,
            "size": stat.st_size if stat and path.is_file() else None,
            "modified_at": stat.st_mtime if stat else None,
            "extension": path.suffix.lower(),
            "previewable": path.is_file() and path.suffix.lower() in (IMAGE_EXTENSIONS | VIDEO_EXTENSIONS),
        }
        if extra:
            item.update(extra)
        return item

    def _collect_files(
        self,
        directory: Path,
        extensions: set[str],
        limit: int,
        recursive: bool = False,
    ) -> list[dict[str, Any]]:
        if not directory.exists() or not directory.is_dir():
            return []
        iterator = directory.rglob("*") if recursive else directory.iterdir()
        files = [
            self._path_item(path, "file")
            for path in iterator
            if path.is_file() and path.suffix.lower() in extensions
        ]
        files.sort(key=lambda item: item.get("modified_at") or 0, reverse=True)
        return files[:limit]

    def stream_logs(self, job_id: str, last_event_id: int = 0) -> Iterator[dict[str, Any]]:
        while True:
            with self._condition:
                while True:
                    job = self._jobs.get(job_id)
                    if job is None:
                        raise KeyError(job_id)
                    pending = [event.to_dict() for event in job.logs if event.id > last_event_id]
                    terminal = job.status in TERMINAL_STATUSES
                    if pending or terminal:
                        break
                    self._condition.wait(timeout=1.0)

                for event in pending:
                    last_event_id = event["id"]
                    yield event

                if terminal and not pending:
                    break
