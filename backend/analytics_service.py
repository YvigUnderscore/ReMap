from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import os
from pathlib import Path
import struct
import subprocess
import time
from typing import Any

import h5py

from .models import JobDetail


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".exr"}
_H5_CACHE: dict[str, tuple[float, int, dict[str, Any]]] = {}
_OUTPUT_CACHE: dict[str, tuple[float, int, dict[str, Any]]] = {}
_MAXIMA = {
    "gpu_temperature_c": None,
    "gpu_memory_used_mb": None,
    "cpu_percent": None,
    "ram_percent": None,
    "process_rss_mb": None,
}


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _seconds_between(start: str | None, end: str | None = None) -> float | None:
    start_dt = _parse_iso(start)
    end_dt = _parse_iso(end) or datetime.now(timezone.utc)
    if start_dt is None:
        return None
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)
    return max(0.0, (end_dt - start_dt).total_seconds())


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _update_max(name: str, value: float | int | None) -> None:
    if value is None:
        return
    current = _MAXIMA.get(name)
    if current is None or value > current:
        _MAXIMA[name] = value


def _system_snapshot() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "cpu_percent": None,
        "ram_percent": None,
        "ram_used_gb": None,
        "ram_total_gb": None,
        "process_rss_mb": None,
        "disk_used_gb": None,
        "disk_total_gb": None,
        "disk_percent": None,
    }
    try:
        import psutil

        cpu_percent = float(psutil.cpu_percent(interval=None))
        vm = psutil.virtual_memory()
        process = psutil.Process(os.getpid())
        disk = psutil.disk_usage(str(Path.cwd().anchor or Path.cwd()))
        payload.update(
            {
                "cpu_percent": cpu_percent,
                "ram_percent": float(vm.percent),
                "ram_used_gb": vm.used / (1024**3),
                "ram_total_gb": vm.total / (1024**3),
                "process_rss_mb": process.memory_info().rss / (1024**2),
                "disk_used_gb": disk.used / (1024**3),
                "disk_total_gb": disk.total / (1024**3),
                "disk_percent": float(disk.percent),
            }
        )
        _update_max("cpu_percent", cpu_percent)
        _update_max("ram_percent", float(vm.percent))
        _update_max("process_rss_mb", payload["process_rss_mb"])
    except Exception:
        pass
    return payload


def _parse_nvidia_smi() -> dict[str, Any]:
    query = (
        "name,utilization.gpu,memory.used,memory.total,temperature.gpu,"
        "power.draw,power.limit"
    )
    cmd = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1.5)
        if result.returncode != 0 or not result.stdout.strip():
            return {}
        line = result.stdout.strip().splitlines()[0]
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 7:
            return {}
        payload = {
            "name": parts[0],
            "utilization_percent": float(parts[1]) if parts[1] else None,
            "memory_used_mb": float(parts[2]) if parts[2] else None,
            "memory_total_mb": float(parts[3]) if parts[3] else None,
            "temperature_c": float(parts[4]) if parts[4] else None,
            "power_draw_w": float(parts[5]) if parts[5] else None,
            "power_limit_w": float(parts[6]) if parts[6] else None,
        }
        _update_max("gpu_temperature_c", payload["temperature_c"])
        _update_max("gpu_memory_used_mb", payload["memory_used_mb"])
        return payload
    except Exception:
        return {}


def _gpu_snapshot() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "available": False,
        "name": "",
        "capability": "",
        "torch_allocated_mb": None,
        "torch_reserved_mb": None,
        "torch_peak_allocated_mb": None,
        "utilization_percent": None,
        "memory_used_mb": None,
        "memory_total_mb": None,
        "temperature_c": None,
        "power_draw_w": None,
        "power_limit_w": None,
    }
    try:
        import torch

        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(index)
            major, minor = torch.cuda.get_device_capability(index)
            payload.update(
                {
                    "available": True,
                    "name": props.name,
                    "capability": f"sm_{major}{minor}",
                    "torch_allocated_mb": torch.cuda.memory_allocated(index) / (1024**2),
                    "torch_reserved_mb": torch.cuda.memory_reserved(index) / (1024**2),
                    "torch_peak_allocated_mb": torch.cuda.max_memory_allocated(index) / (1024**2),
                }
            )
            _update_max("gpu_memory_used_mb", payload["torch_reserved_mb"])
    except Exception:
        pass
    payload.update({k: v for k, v in _parse_nvidia_smi().items() if v not in ("", None)})
    return payload


def _read_points3d_count(path: Path) -> int | None:
    try:
        with path.open("rb") as fh:
            return int(struct.unpack("<Q", fh.read(8))[0])
    except Exception:
        return None


def _h5_stats(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError:
        return {}
    cache_key = str(path)
    cached = _H5_CACHE.get(cache_key)
    if cached and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
        return dict(cached[2])

    stats = {
        "path": str(path),
        "name": path.name,
        "size_mb": stat.st_size / (1024**2),
        "images": 0,
        "pairs": 0,
        "keypoints": 0,
        "matches": 0,
    }
    try:
        with h5py.File(str(path), "r", libver="latest") as fd:
            for key in fd.keys():
                item = fd[key]
                if not isinstance(item, h5py.Group):
                    continue
                if "keypoints" in item:
                    stats["images"] += 1
                    stats["keypoints"] += int(item["keypoints"].shape[0])
                if "matches0" in item:
                    stats["pairs"] += 1
                    matches = item["matches0"].__array__()
                    stats["matches"] += int((matches > -1).sum())
    except Exception:
        pass
    _H5_CACHE[cache_key] = (stat.st_mtime, stat.st_size, dict(stats))
    return stats


def _collect_output_stats(output_path: str) -> dict[str, Any]:
    root = Path(output_path) if output_path else Path()
    if not root.exists():
        return {
            "frames": 0,
            "h5_files": [],
            "features": 0,
            "matches": 0,
            "points3d": None,
            "output_size_mb": 0,
        }

    try:
        marker = root.stat().st_mtime
    except OSError:
        marker = time.time()
    cache_key = str(root)
    cached = _OUTPUT_CACHE.get(cache_key)
    if cached and cached[0] == marker:
        return dict(cached[2])

    frames = 0
    output_size = 0
    h5_files: list[dict[str, Any]] = []
    points3d = None
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            output_size += path.stat().st_size
        except OSError:
            pass
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            frames += 1
        elif suffix == ".h5" and len(h5_files) < 16:
            h5_files.append(_h5_stats(path))
        elif path.name == "points3D.bin" and points3d is None:
            points3d = _read_points3d_count(path)

    features = sum(int(item.get("keypoints", 0) or 0) for item in h5_files)
    matches = sum(int(item.get("matches", 0) or 0) for item in h5_files)
    payload = {
        "frames": frames,
        "h5_files": h5_files,
        "features": features,
        "matches": matches,
        "points3d": points3d,
        "output_size_mb": output_size / (1024**2),
    }
    _OUTPUT_CACHE[cache_key] = (marker, output_size, dict(payload))
    return payload


def _step_timings(job: JobDetail) -> dict[str, float]:
    markers = {
        "extract": ("extract", "extraction"),
        "features": ("feature", "loma feature"),
        "pairs": ("pair",),
        "matching": ("matching", "matched"),
        "sfm": ("sfm", "reconstruction", "triangulation"),
        "postprocess": ("color", "remapping", "clean"),
    }
    starts: dict[str, datetime] = {}
    ordered = []
    for event in job.logs:
        timestamp = _parse_iso(event.timestamp)
        if timestamp is None:
            continue
        message = event.message.lower()
        for name, needles in markers.items():
            if name not in starts and any(needle in message for needle in needles):
                starts[name] = timestamp
                ordered.append((name, timestamp))

    if not ordered:
        return {}
    end_dt = _parse_iso(job.updated_at) or datetime.now(timezone.utc)
    timings: dict[str, float] = {}
    for index, (name, timestamp) in enumerate(ordered):
        next_timestamp = ordered[index + 1][1] if index + 1 < len(ordered) else end_dt
        timings[name] = max(0.0, (next_timestamp - timestamp).total_seconds())
    return timings


def build_analytics_payload(jobs: list[JobDetail]) -> dict[str, Any]:
    jobs_sorted = sorted(jobs, key=lambda item: item.updated_at, reverse=True)
    status_counts = Counter(job.status for job in jobs)
    matcher_counts = Counter(str(job.request.get("matcher_type", "")) for job in jobs)
    feature_counts = Counter(str(job.request.get("feature_type", "")) for job in jobs)
    input_counts = Counter(job.input_mode for job in jobs)

    recent = []
    total_features = 0
    total_matches = 0
    total_frames = 0
    total_pairs = 0
    feature_ms_samples: list[float] = []
    match_ms_samples: list[float] = []
    total_duration_samples: list[float] = []

    for job in jobs_sorted[:20]:
        artifacts = _collect_output_stats(job.output_path)
        timings = _step_timings(job)
        duration = _seconds_between(job.created_at, job.updated_at)
        if duration and job.status in {"completed", "failed", "cancelled", "interrupted"}:
            total_duration_samples.append(duration)

        h5_pairs = sum(int(item.get("pairs", 0) or 0) for item in artifacts["h5_files"])
        total_features += int(artifacts["features"])
        total_matches += int(artifacts["matches"])
        total_frames += int(artifacts["frames"])
        total_pairs += h5_pairs
        feature_ms = _safe_ratio((timings.get("features") or 0) * 1000, artifacts["frames"])
        match_ms = _safe_ratio((timings.get("matching") or 0) * 1000, h5_pairs)
        if feature_ms:
            feature_ms_samples.append(feature_ms)
        if match_ms:
            match_ms_samples.append(match_ms)

        recent.append(
            {
                "job_id": job.job_id,
                "label": job.label,
                "status": job.status,
                "progress": job.progress,
                "matcher_type": job.request.get("matcher_type"),
                "feature_type": job.request.get("feature_type"),
                "input_mode": job.input_mode,
                "frames": artifacts["frames"],
                "features": artifacts["features"],
                "matches": artifacts["matches"],
                "pairs": h5_pairs,
                "points3d": artifacts["points3d"],
                "output_size_mb": artifacts["output_size_mb"],
                "duration_seconds": duration,
                "step_seconds": timings,
                "feature_ms_per_frame": feature_ms,
                "match_ms_per_pair": match_ms,
                "updated_at": job.updated_at,
            }
        )

    system = _system_snapshot()
    gpu = _gpu_snapshot()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "system": system,
        "gpu": gpu,
        "maxima": dict(_MAXIMA),
        "jobs": {
            "total": len(jobs),
            "active": sum(1 for job in jobs if job.status in {"queued", "processing", "paused"}),
            "completed": status_counts.get("completed", 0),
            "failed": status_counts.get("failed", 0) + status_counts.get("interrupted", 0),
            "cancelled": status_counts.get("cancelled", 0),
            "by_status": dict(status_counts),
            "recent": recent,
        },
        "usage": {
            "matchers": dict(matcher_counts),
            "features": dict(feature_counts),
            "input_modes": dict(input_counts),
        },
        "throughput": {
            "frames_observed": total_frames,
            "features_observed": total_features,
            "matches_observed": total_matches,
            "pairs_observed": total_pairs,
            "avg_feature_ms_per_frame": (
                sum(feature_ms_samples) / len(feature_ms_samples) if feature_ms_samples else None
            ),
            "avg_match_ms_per_pair": (
                sum(match_ms_samples) / len(match_ms_samples) if match_ms_samples else None
            ),
            "avg_job_duration_seconds": (
                sum(total_duration_samples) / len(total_duration_samples) if total_duration_samples else None
            ),
        },
    }
