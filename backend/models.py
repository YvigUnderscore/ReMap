from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from multiprocessing import cpu_count
from typing import Any
import secrets
import shutil

SETTINGS_SCHEMA_VERSION = 4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_input_mode(value: str | None) -> str:
    value = (value or "video").strip().lower()
    if value in {"video", "videos", "video (.mp4, .mov)"}:
        return "video"
    if value in {"images", "image", "image folder"}:
        return "images"
    if value in {"rescan", "rescan (lidar)", "lidar", "stray"}:
        return "rescan"
    return "video"


def legacy_input_mode_label(value: str) -> str:
    normalized = normalize_input_mode(value)
    return {
        "video": "Video (.mp4, .mov)",
        "images": "Image Folder",
        "rescan": "Rescan (LiDAR)",
    }[normalized]


def default_worker_count() -> int:
    return max(1, cpu_count())


def global_mapper_available() -> bool:
    try:
        import pycolmap

        if hasattr(pycolmap, "global_mapping"):
            return True
    except Exception:
        pass
    return shutil.which("glomap") is not None


def default_mapper_type() -> str:
    return "GLOMAP" if global_mapper_available() else "COLMAP"


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    return [str(value).strip()]


@dataclass
class ProcessingJobRequest:
    input_mode: str = "video"
    input_paths: list[str] = field(default_factory=list)
    output_path: str = ""
    fps_extract: float = 4.0
    force_16bit: bool = False
    camera_model: str = "OPENCV"
    feature_type: str = "superpoint_aachen"
    matcher_type: str = "superpoint+lightglue"
    max_keypoints: int = 4096
    pairing_mode: str = "Sequential (Video)"
    mapper_type: str = field(default_factory=default_mapper_type)
    stray_approach: str = "full_sfm"
    stray_confidence: int = 2
    stray_depth_subsample: int = 2
    stray_gen_pointcloud: bool = True
    color_enabled: bool = False
    color_source: str = "Auto-detect"
    color_dest: str = "ACEScg (EXR + sRGB PNG)"
    detected_color_profile: str = ""
    ocio_path: str = ""
    ocio_in_cs: str = ""
    ocio_out_cs: str = ""
    keep_srgb_png: bool = True
    use_acescg_exr: bool = True
    num_workers: int = field(default_factory=default_worker_count)
    server_port: int = 5000
    server_api_key: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    label: str = ""
    skip_existing: bool = True
    quality_sweep: bool = False
    sweep_sample_frames: int = 80
    exclude_blurry: bool = False
    exclude_black: bool = False
    blur_threshold: float = 75.0
    black_threshold: float = 0.08
    ram_limit_percent: int = 90
    gpu_vram_limit_percent: int = 92

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "ProcessingJobRequest":
        payload = payload or {}
        default = cls()
        return cls(
            input_mode=normalize_input_mode(payload.get("input_mode", default.input_mode)),
            input_paths=_coerce_list(payload.get("input_paths", default.input_paths)),
            output_path=str(payload.get("output_path", default.output_path)).strip(),
            fps_extract=float(payload.get("fps_extract", default.fps_extract)),
            force_16bit=_coerce_bool(payload.get("force_16bit"), default.force_16bit),
            camera_model=str(payload.get("camera_model", default.camera_model)).strip() or default.camera_model,
            feature_type=str(payload.get("feature_type", default.feature_type)).strip() or default.feature_type,
            matcher_type=str(payload.get("matcher_type", default.matcher_type)).strip() or default.matcher_type,
            max_keypoints=int(payload.get("max_keypoints", default.max_keypoints)),
            pairing_mode=str(payload.get("pairing_mode", default.pairing_mode)).strip() or default.pairing_mode,
            mapper_type=str(payload.get("mapper_type", default.mapper_type)).strip() or default.mapper_type,
            stray_approach=str(payload.get("stray_approach", default.stray_approach)).strip() or default.stray_approach,
            stray_confidence=int(payload.get("stray_confidence", default.stray_confidence)),
            stray_depth_subsample=int(payload.get("stray_depth_subsample", default.stray_depth_subsample)),
            stray_gen_pointcloud=_coerce_bool(payload.get("stray_gen_pointcloud"), default.stray_gen_pointcloud),
            color_enabled=_coerce_bool(payload.get("color_enabled"), default.color_enabled),
            color_source=str(payload.get("color_source", default.color_source)).strip() or default.color_source,
            color_dest=str(payload.get("color_dest", default.color_dest)).strip() or default.color_dest,
            detected_color_profile=str(payload.get("detected_color_profile", default.detected_color_profile)).strip(),
            ocio_path=str(payload.get("ocio_path", default.ocio_path)).strip(),
            ocio_in_cs=str(payload.get("ocio_in_cs", default.ocio_in_cs)).strip(),
            ocio_out_cs=str(payload.get("ocio_out_cs", default.ocio_out_cs)).strip(),
            keep_srgb_png=_coerce_bool(payload.get("keep_srgb_png"), default.keep_srgb_png),
            use_acescg_exr=_coerce_bool(payload.get("use_acescg_exr"), default.use_acescg_exr),
            num_workers=int(payload.get("num_workers", default.num_workers)),
            server_port=int(payload.get("server_port", default.server_port)),
            server_api_key=str(payload.get("server_api_key", default.server_api_key)).strip() or default.server_api_key,
            label=str(payload.get("label", default.label)).strip(),
            skip_existing=_coerce_bool(payload.get("skip_existing"), default.skip_existing),
            quality_sweep=_coerce_bool(payload.get("quality_sweep"), default.quality_sweep),
            sweep_sample_frames=int(payload.get("sweep_sample_frames", default.sweep_sample_frames)),
            exclude_blurry=_coerce_bool(payload.get("exclude_blurry"), default.exclude_blurry),
            exclude_black=_coerce_bool(payload.get("exclude_black"), default.exclude_black),
            blur_threshold=float(payload.get("blur_threshold", default.blur_threshold)),
            black_threshold=float(payload.get("black_threshold", default.black_threshold)),
            ram_limit_percent=int(payload.get("ram_limit_percent", default.ram_limit_percent)),
            gpu_vram_limit_percent=int(payload.get("gpu_vram_limit_percent", default.gpu_vram_limit_percent)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JobLogEvent:
    id: int
    timestamp: str
    level: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JobSummary:
    job_id: str
    status: str
    progress: int
    current_step: str
    created_at: str
    updated_at: str
    label: str = ""
    output_path: str = ""
    input_mode: str = "video"
    error: str | None = None
    queue_position: int | None = None
    progress_note: str = ""
    eta_seconds: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JobDetail(JobSummary):
    request: dict[str, Any] = field(default_factory=dict)
    logs: list[JobLogEvent] = field(default_factory=list)

    def summary(self) -> JobSummary:
        return JobSummary(
            job_id=self.job_id,
            status=self.status,
            progress=self.progress,
            current_step=self.current_step,
            created_at=self.created_at,
            updated_at=self.updated_at,
            label=self.label,
            output_path=self.output_path,
            input_mode=self.input_mode,
            error=self.error,
            queue_position=self.queue_position,
            progress_note=self.progress_note,
            eta_seconds=self.eta_seconds,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["logs"] = [event.to_dict() for event in self.logs]
        return data


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 5000
    api_key: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    auto_start: bool = False
    output_dir: str = ""

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "ServerConfig":
        payload = payload or {}
        default = cls()
        return cls(
            host=str(payload.get("host", default.host)).strip() or default.host,
            port=int(payload.get("port", default.port)),
            api_key=str(payload.get("api_key", default.api_key)).strip() or default.api_key,
            auto_start=_coerce_bool(payload.get("auto_start"), default.auto_start),
            output_dir=str(payload.get("output_dir", default.output_dir)).strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SystemCapabilities:
    cpu_count: int
    glomap_available: bool
    ffmpeg_available: bool
    ffprobe_available: bool
    openimageio_available: bool
    torch_available: bool
    cuda_available: bool
    loma_available: bool
    triton_available: bool
    torch_device_name: str
    torch_compute_capability: str
    python_version: str
    platform: str
    ocio_env: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AppSettings:
    schema_version: int = SETTINGS_SCHEMA_VERSION
    theme: str = "graphite"
    defaults: ProcessingJobRequest = field(default_factory=ProcessingJobRequest)
    server: ServerConfig = field(default_factory=ServerConfig)
    recent_inputs: list[str] = field(default_factory=list)
    recent_outputs: list[str] = field(default_factory=list)
    notifications_enabled: bool = True
    system_notifications: bool = True
    cache_enabled: bool = True
    cache_max_size_gb: float = 25.0
    ram_limit_percent: int = 90
    gpu_vram_limit_percent: int = 92
    blur_threshold: float = 75.0
    black_threshold: float = 0.08

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "AppSettings":
        payload = payload or {}
        default = cls()
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            theme=str(payload.get("theme", "graphite")).strip() or "graphite",
            defaults=ProcessingJobRequest.from_payload(payload.get("defaults")),
            server=ServerConfig.from_payload(payload.get("server")),
            recent_inputs=_coerce_list(payload.get("recent_inputs", default.recent_inputs))[:20],
            recent_outputs=_coerce_list(payload.get("recent_outputs", default.recent_outputs))[:20],
            notifications_enabled=_coerce_bool(payload.get("notifications_enabled"), default.notifications_enabled),
            system_notifications=_coerce_bool(payload.get("system_notifications"), default.system_notifications),
            cache_enabled=_coerce_bool(payload.get("cache_enabled"), default.cache_enabled),
            cache_max_size_gb=float(payload.get("cache_max_size_gb", default.cache_max_size_gb)),
            ram_limit_percent=int(payload.get("ram_limit_percent", default.ram_limit_percent)),
            gpu_vram_limit_percent=int(payload.get("gpu_vram_limit_percent", default.gpu_vram_limit_percent)),
            blur_threshold=float(payload.get("blur_threshold", default.blur_threshold)),
            black_threshold=float(payload.get("black_threshold", default.black_threshold)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "theme": self.theme,
            "defaults": self.defaults.to_dict(),
            "server": self.server.to_dict(),
            "recent_inputs": list(self.recent_inputs),
            "recent_outputs": list(self.recent_outputs),
            "notifications_enabled": self.notifications_enabled,
            "system_notifications": self.system_notifications,
            "cache_enabled": self.cache_enabled,
            "cache_max_size_gb": self.cache_max_size_gb,
            "ram_limit_percent": self.ram_limit_percent,
            "gpu_vram_limit_percent": self.gpu_vram_limit_percent,
            "blur_threshold": self.blur_threshold,
            "black_threshold": self.black_threshold,
        }
