from __future__ import annotations

from collections import deque
import json
from pathlib import Path
from threading import Condition, RLock, Thread
from typing import Any, Iterator
from uuid import uuid4

from .bundle_postprocess import normalize_final_bundle
from .legacy_runner import LegacyGuiPipelineRunner
from .models import JobDetail, JobLogEvent, ProcessingJobRequest, utc_now_iso
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
        self._update_job(
            job_id,
            status="processing",
            current_step="Initialising",
            progress=1,
            error=None,
            queue_position=1,
            progress_note="Starting pipeline",
        )
        self._append_log(job_id, "info", "Legacy pipeline bridge started")

        def on_log(message: str) -> None:
            self._append_log(job_id, "info", message)

        def on_step(index: int, total: int, name: str) -> None:
            progress = min(95, max(1, int((index / max(total, 1)) * 100)))
            self._update_job(job_id, current_step=name, progress=progress, progress_note="")

        def on_detail(note: str, progress: int | None = None) -> None:
            changes: dict[str, Any] = {"progress_note": note}
            if progress is not None:
                changes["progress"] = min(99, max(1, int(progress)))
            self._update_job(job_id, **changes)

        def on_finish(success: bool, cancelled: bool) -> None:
            if cancelled:
                self._update_job(job_id, status="cancelled", current_step="Cancelled", progress=100, progress_note="")
                self._append_log(job_id, "warning", "Job cancelled")
            elif success:
                normalize_final_bundle(
                    request.output_path,
                    keep_srgb_png=request.keep_srgb_png and request.input_mode != "rescan",
                    use_acescg_exr=request.use_acescg_exr,
                )
                self._update_job(job_id, status="completed", current_step="Done", progress=100, progress_note="")
                self._append_log(job_id, "info", "Job completed successfully")
            else:
                self._update_job(job_id, status="failed", current_step="Error", progress=100, progress_note="")

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
                    normalize_final_bundle(
                        request.output_path,
                        keep_srgb_png=request.keep_srgb_png and request.input_mode != "rescan",
                        use_acescg_exr=request.use_acescg_exr,
                    )
                    self._update_job(job_id, status="completed", current_step="Done", progress=100, progress_note="")
        except Exception as exc:
            self._append_log(job_id, "error", f"Job failed: {exc}")
            self._update_job(job_id, status="failed", current_step="Error", progress=100, error=str(exc), progress_note="")
        finally:
            with self._lock:
                self._runners.pop(job_id, None)
                self._condition.notify_all()

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
