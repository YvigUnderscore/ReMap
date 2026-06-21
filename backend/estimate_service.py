from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import Any

from .analytics_service import build_analytics_payload
from .models import JobDetail, ProcessingJobRequest
from .probe_service import probe_inputs


def _pair_count(frames: int, mode: str) -> int:
    if frames <= 1:
        return 0
    if "exhaustive" in mode.lower():
        return frames * (frames - 1) // 2
    return sum(min(20, frames - index - 1) for index in range(frames))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _duration_model(jobs: list[JobDetail], request: ProcessingJobRequest) -> dict[str, Any]:
    analytics = build_analytics_payload(jobs)
    similar = [
        item for item in analytics["jobs"]["recent"]
        if item.get("status") == "completed"
        and item.get("input_mode") == request.input_mode
        and item.get("matcher_type") == request.matcher_type
        and item.get("feature_type") == request.feature_type
    ]
    pool = similar or [
        item for item in analytics["jobs"]["recent"]
        if item.get("status") == "completed" and item.get("input_mode") == request.input_mode
    ]
    feature_ms = _median([
        float(item["feature_ms_per_frame"])
        for item in pool
        if item.get("feature_ms_per_frame")
    ])
    match_ms = _median([
        float(item["match_ms_per_pair"])
        for item in pool
        if item.get("match_ms_per_pair")
    ])
    total_per_frame = _median([
        float(item["duration_seconds"]) / max(1, int(item.get("frames") or 0))
        for item in pool
        if item.get("duration_seconds") and item.get("frames")
    ])
    return {
        "confidence": "history" if pool else "fallback",
        "samples": len(pool),
        "feature_ms_per_frame": feature_ms or 65.0,
        "match_ms_per_pair": match_ms or (34.0 if "loma" in request.matcher_type.lower() else 18.0),
        "total_seconds_per_frame": total_per_frame or 0.9,
    }


def _resource_warnings(request: ProcessingJobRequest, estimated_disk_bytes: int, frames: int) -> list[str]:
    warnings: list[str] = []
    try:
        if request.output_path:
            usage = shutil.disk_usage(str(Path(request.output_path).anchor or Path(request.output_path)))
            remaining = usage.free - estimated_disk_bytes
            if remaining < 2 * 1024**3:
                warnings.append("Disk space may be tight after this job.")
    except Exception:
        pass
    try:
        import psutil

        ram = psutil.virtual_memory()
        if ram.percent >= request.ram_limit_percent:
            warnings.append(f"RAM is already above {request.ram_limit_percent}%.")
    except Exception:
        pass
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=1.0,
        )
        if result.returncode == 0 and result.stdout.strip():
            used, total = [float(part.strip()) for part in result.stdout.splitlines()[0].split(",")[:2]]
            if total > 0 and used / total * 100 >= request.gpu_vram_limit_percent:
                warnings.append(f"GPU VRAM is already above {request.gpu_vram_limit_percent}%.")
    except Exception:
        pass
    if "exhaustive" in request.pairing_mode.lower() and frames > 250:
        warnings.append("Exhaustive pairing on more than 250 frames can be very expensive.")
    if frames > 1200:
        warnings.append("Large frame count; consider a lower FPS or quality sweep first.")
    return warnings


def estimate_request(payload: dict[str, Any], jobs: list[JobDetail]) -> dict[str, Any]:
    request = ProcessingJobRequest.from_payload(payload)
    probe = probe_inputs(request.input_mode, request.input_paths, request.fps_extract)
    frames = sum(int(item.get("estimated_frames", 0) or 0) for item in probe["items"] if item.get("valid", True))
    pairs = _pair_count(frames, request.pairing_mode)
    model = _duration_model(jobs, request)
    extraction_seconds = 0 if request.input_mode == "images" else frames * 0.045
    feature_seconds = frames * model["feature_ms_per_frame"] / 1000
    match_seconds = pairs * model["match_ms_per_pair"] / 1000
    sfm_seconds = max(20.0 if frames else 0.0, frames * model["total_seconds_per_frame"] * 0.55)
    color_seconds = frames * (0.08 if request.color_enabled else 0.0)
    total_seconds = extraction_seconds + color_seconds + feature_seconds + match_seconds + sfm_seconds
    bytes_per_frame = 12_000_000 if request.use_acescg_exr or request.color_dest.startswith("ACEScg") else 2_500_000
    estimated_disk_bytes = int(frames * bytes_per_frame + pairs * 220 + 250 * 1024**2)
    return {
        "input_mode": request.input_mode,
        "items": probe["items"],
        "frames": frames,
        "pairs": pairs,
        "estimated_disk_bytes": estimated_disk_bytes,
        "estimated_seconds": int(total_seconds),
        "confidence": model["confidence"],
        "history_samples": model["samples"],
        "breakdown": {
            "extraction_seconds": int(extraction_seconds),
            "color_seconds": int(color_seconds),
            "feature_seconds": int(feature_seconds),
            "match_seconds": int(match_seconds),
            "sfm_seconds": int(sfm_seconds),
        },
        "warnings": _resource_warnings(request, estimated_disk_bytes, frames),
    }


def estimate_payload(payload: dict[str, Any], jobs: list[JobDetail]) -> dict[str, Any]:
    raw_requests = payload.get("requests")
    if isinstance(raw_requests, list):
        requests = [item for item in raw_requests if isinstance(item, dict)]
    else:
        requests = [payload]
    estimates = [estimate_request(item, jobs) for item in requests]
    return {
        "estimates": estimates,
        "total_frames": sum(item["frames"] for item in estimates),
        "total_pairs": sum(item["pairs"] for item in estimates),
        "total_disk_bytes": sum(item["estimated_disk_bytes"] for item in estimates),
        "total_seconds": sum(item["estimated_seconds"] for item in estimates),
        "warnings": [warning for item in estimates for warning in item["warnings"]],
    }
