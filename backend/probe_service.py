from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from .models import global_mapper_available, normalize_input_mode


ROOT_DIR = Path(__file__).resolve().parent.parent
FFPROBE_TO_SOURCE = {
    ("bt2020", "linear"): "Linear BT.2020",
    ("bt2020", "alog"): "Apple Log (BT.2020)",
    ("bt2020", "arib-std-b67"): "HLG (BT.2020)",
    ("bt709", "bt709"): "sRGB (Rec.709)",
    ("bt709", "linear"): "Linear sRGB",
    ("bt709", "iec61966-2-1"): "sRGB (Rec.709)",
    ("bt709", "srgb"): "sRGB (Rec.709)",
    # Sony S-Log 3 / S-Gamut 3 (ffprobe reports bt2020 primaries + unknown/slog3 TRC)
    ("bt2020", "unknown"): "S-Log 3 (S-Gamut 3)",
    ("bt2020", "slog3"): "S-Log 3 (S-Gamut 3)",
}

COLOR_SOURCES = [
    "Auto-detect",
    "Linear BT.2020",
    "Linear ACEScg",
    "Apple Log (BT.2020)",
    "Linear sRGB",
    "sRGB (Rec.709)",
    "HLG (BT.2020)",
    "S-Log 3 (S-Gamut 3)",
]

COLOR_DESTINATIONS = [
    "ACEScg (EXR + sRGB PNG)",
    "Linear sRGB",
    "sRGB (Tone Mapped)",
    "Custom OCIO...",
]

FEATURES = ["superpoint_aachen", "superpoint_max", "disk", "aliked-n16", "sift"]
MATCHERS = ["superpoint+lightglue", "superglue", "disk+lightglue", "adalam", "loma_b", "loma_g"]
PAIRING = ["Sequential (Video)", "Exhaustive (Small dataset < 200)"]
CAMERA_MODELS = ["OPENCV", "PINHOLE", "SIMPLE_RADIAL", "OPENCV_FISHEYE"]
MAPPERS = ["COLMAP", "GLOMAP"]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, "N/A", ""):
            return default
        return int(float(value))
    except Exception:
        return default


def _parse_rate(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value or "").strip()
        if "/" in text:
            num, den = text.split("/", 1)
            den_value = float(den)
            if den_value > 0:
                return float(num) / den_value
        parsed = float(text)
        return parsed if parsed > 0 else default
    except Exception:
        return default


def _sampling_step(native_fps: float, target_fps: float | None) -> int:
    if not target_fps or target_fps <= 0:
        return 1
    if native_fps <= 0:
        return 1
    return max(1, round(native_fps / target_fps))


def _estimate_sampled_frames(
    total_frames: int,
    native_fps: float,
    target_fps: float | None,
    duration: float = 0.0,
) -> tuple[int, int]:
    step = _sampling_step(native_fps, target_fps)
    if total_frames > 0:
        return math.ceil(total_frames / step), step
    if target_fps and duration > 0:
        return max(0, round(duration * target_fps)), step
    return 0, step


def probe_video(path: str | Path) -> dict[str, Any]:
    video_path = Path(path)
    info: dict[str, Any] = {
        "path": str(video_path),
        "name": video_path.name,
        "kind": "video",
        "duration": 0.0,
        "native_fps": 30.0,
        "total_frames": 0,
        "estimated_frames": 0,
        "color_profile": "",
        "valid": video_path.is_file(),
    }
    if not video_path.exists():
        return info
    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return info
        data = json.loads(result.stdout)
        info["duration"] = _safe_float(data.get("format", {}).get("duration"), 0.0)
        for stream in data.get("streams", []):
            if stream.get("codec_type") != "video":
                continue
            native_fps = _parse_rate(stream.get("avg_frame_rate"), 0.0)
            if native_fps <= 0:
                native_fps = _parse_rate(stream.get("r_frame_rate"), 30.0)
            info["native_fps"] = native_fps or 30.0
            if not info["duration"]:
                info["duration"] = _safe_float(stream.get("duration"), 0.0)
            c_prim = stream.get("color_primaries", "").lower()
            c_trc = stream.get("color_transfer", "").lower()
            info["color_profile"] = FFPROBE_TO_SOURCE.get((c_prim, c_trc), "")
            info["color_primaries"] = c_prim
            info["color_transfer"] = c_trc
            total_frames = _safe_int(stream.get("nb_frames"), 0)
            if total_frames <= 0 and info["duration"] > 0 and info["native_fps"] > 0:
                total_frames = max(0, round(info["duration"] * info["native_fps"]))
            info["total_frames"] = total_frames
            break
        info["estimated_frames"] = info["total_frames"]
    except Exception:
        return info
    return info


def _probe_image_folder(path: str | Path) -> dict[str, Any]:
    folder = Path(path)
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".exr"}
    total_frames = 0
    if folder.is_dir():
        total_frames = sum(1 for item in folder.iterdir() if item.is_file() and item.suffix.lower() in exts)
    return {
        "path": str(folder),
        "name": folder.name,
        "kind": "images",
        "total_frames": total_frames,
        "estimated_frames": total_frames,
        "native_fps": 0.0,
        "valid": folder.is_dir() and total_frames > 0,
    }


def probe_rescan_dataset(path: str | Path) -> dict[str, Any]:
    dataset_path = Path(path)
    info: dict[str, Any] = {
        "path": str(dataset_path),
        "name": dataset_path.name,
        "kind": "rescan",
        "total_frames": 0,
        "estimated_frames": 0,
        "native_fps": 60.0,
        "color_profile": "",
        "valid": False,
    }
    if not dataset_path.exists():
        return info

    rgb_video = dataset_path / "rgb.mp4"
    if not rgb_video.exists():
        rgb_video = dataset_path / "rgb.mov"

    has_rgb_seq = (dataset_path / "rgb").is_dir() and any((dataset_path / "rgb").iterdir())
    has_rgb = rgb_video.exists() or has_rgb_seq
    has_odom = (dataset_path / "odometry.csv").exists()
    has_cam = (dataset_path / "camera_matrix.csv").exists()
    info["valid"] = has_rgb and has_odom and has_cam

    if rgb_video.exists():
        video_info = probe_video(rgb_video)
        info["native_fps"] = video_info.get("native_fps", 60.0)
        info["color_profile"] = video_info.get("color_profile", "")

    odom_file = dataset_path / "odometry.csv"
    if odom_file.exists():
        try:
            with odom_file.open("r", encoding="utf-8") as fh:
                reader = csv.reader(fh)
                next(reader, None)
                timestamps: list[float] = []
                for row in reader:
                    if row:
                        timestamps.append(float(row[0].strip()))
            info["total_frames"] = len(timestamps)
            if len(timestamps) > 1:
                duration = timestamps[-1] - timestamps[0]
                if duration > 0:
                    computed = (len(timestamps) - 1) / duration
                    if computed > 0.1:
                        info["native_fps"] = computed
        except Exception:
            pass

    return info


def discover_ocio_configs() -> list[str]:
    candidates: list[Path] = []
    env_path = os.environ.get("OCIO")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(ROOT_DIR.glob("*.ocio"))
    unique: list[str] = []
    for path in candidates:
        try:
            resolved = str(path.resolve())
        except Exception:
            resolved = str(path)
        if Path(resolved).exists() and resolved not in unique:
            unique.append(resolved)
    return unique


def load_ocio_spaces(config_path: str | None) -> list[str]:
    if not config_path:
        return []
    path = Path(config_path)
    if not path.exists():
        return []
    try:
        import OpenImageIO as oiio

        config = oiio.ColorConfig(str(path))
        names = config.getColorSpaceNames()
        return sorted(list(names)) if names else []
    except Exception:
        return []


def probe_inputs(input_mode: str, input_paths: list[str], target_fps: float | None = None) -> dict[str, Any]:
    normalized = normalize_input_mode(input_mode)
    items: list[dict[str, Any]] = []
    detected_profile = ""
    max_native_fps = 0.0
    for item in input_paths:
        if normalized == "video":
            info = probe_video(item)
        elif normalized == "rescan":
            info = probe_rescan_dataset(item)
        else:
            info = _probe_image_folder(item)
        max_native_fps = max(max_native_fps, float(info.get("native_fps", 0.0) or 0.0))
        if not detected_profile and info.get("color_profile"):
            detected_profile = str(info["color_profile"])
        fps = target_fps or 0.0
        if normalized in {"video", "rescan"} and fps > 0:
            native_fps = float(info.get("native_fps", fps) or fps)
            estimated, step = _estimate_sampled_frames(
                int(info.get("total_frames", 0) or 0),
                native_fps,
                fps,
                float(info.get("duration", 0.0) or 0.0),
            )
            info["estimated_frames"] = estimated
            info["step"] = step
        items.append(info)
    return {
        "input_mode": normalized,
        "items": items,
        "detected_color_profile": detected_profile,
        "max_native_fps": max_native_fps,
    }


def build_option_payload() -> dict[str, Any]:
    ocio_configs = discover_ocio_configs()
    selected_ocio = ocio_configs[0] if ocio_configs else ""
    ocio_spaces = load_ocio_spaces(selected_ocio)
    mapper_types = ["GLOMAP", "COLMAP"] if global_mapper_available() else MAPPERS
    return {
        "color_sources": COLOR_SOURCES,
        "color_destinations": COLOR_DESTINATIONS,
        "ocio_configs": ocio_configs,
        "ocio_spaces": ocio_spaces,
        "features": FEATURES,
        "matchers": MATCHERS,
        "pairing_modes": PAIRING,
        "camera_models": CAMERA_MODELS,
        "mapper_types": mapper_types,
        "default_ocio_config": selected_ocio,
    }
