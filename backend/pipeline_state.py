from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any

from .models import ProcessingJobRequest


ROOT_DIR = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT_DIR / "backend_state"
CACHE_DIR = STATE_DIR / "cache"
SMALL_HASH_LIMIT = 64 * 1024 * 1024
SAMPLED_BYTES = 4 * 1024 * 1024
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".exr", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}
CACHE_FILES = ("features.h5", "pairs.txt", "matches.h5", "loma_features.h5", "loma_matches.h5")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    size = path.stat().st_size
    with path.open("rb") as fh:
        if size <= SMALL_HASH_LIMIT:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
            return h.hexdigest()
        h.update(fh.read(SAMPLED_BYTES))
        fh.seek(max(0, size - SAMPLED_BYTES))
        h.update(fh.read(SAMPLED_BYTES))
    h.update(str(size).encode("utf-8"))
    h.update(str(int(path.stat().st_mtime)).encode("utf-8"))
    return f"sampled:{h.hexdigest()}"


def _fingerprint_path(path: Path) -> dict[str, Any]:
    if path.is_file():
        stat = path.stat()
        return {
            "path": str(path),
            "kind": "file",
            "name": path.name,
            "size": stat.st_size,
            "mtime": int(stat.st_mtime),
            "sha256": _sha256_file(path),
        }
    if path.is_dir():
        files = [
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in (IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | {".csv", ".json", ".txt"})
        ]
        files.sort(key=lambda item: str(item.relative_to(path)).replace("\\", "/").lower())
        manifest = []
        for item in files[:5000]:
            stat = item.stat()
            manifest.append(
                {
                    "rel": str(item.relative_to(path)).replace("\\", "/"),
                    "size": stat.st_size,
                    "mtime": int(stat.st_mtime),
                    "sha256": _sha256_file(item),
                }
            )
        digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()
        return {
            "path": str(path),
            "kind": "directory",
            "name": path.name,
            "file_count": len(files),
            "sha256": digest,
            "truncated": len(files) > len(manifest),
        }
    return {"path": str(path), "kind": "missing", "sha256": "missing"}


def request_fingerprint(request: ProcessingJobRequest) -> dict[str, Any]:
    inputs = [_fingerprint_path(Path(item)) for item in request.input_paths]
    payload = {
        "version": 1,
        "input_mode": request.input_mode,
        "inputs": inputs,
        "sampling": {
            "fps_extract": request.fps_extract,
            "stray_confidence": request.stray_confidence,
            "stray_depth_subsample": request.stray_depth_subsample,
        },
        "color": {
            "enabled": request.color_enabled,
            "source": request.color_source,
            "dest": request.color_dest,
            "ocio_path": request.ocio_path,
            "ocio_in_cs": request.ocio_in_cs,
            "ocio_out_cs": request.ocio_out_cs,
            "use_acescg_exr": request.use_acescg_exr,
        },
        "quality": {
            "exclude_blurry": request.exclude_blurry,
            "exclude_black": request.exclude_black,
            "blur_threshold": request.blur_threshold,
            "black_threshold": request.black_threshold,
        },
    }
    payload["digest"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return payload


def cache_key(request: ProcessingJobRequest, fingerprint: dict[str, Any]) -> str:
    payload = {
        "version": 1,
        "fingerprint": fingerprint.get("digest"),
        "feature_type": request.feature_type,
        "matcher_type": request.matcher_type,
        "max_keypoints": request.max_keypoints,
        "pairing_mode": request.pairing_mode,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass
class CheckpointStore:
    output_path: Path
    fingerprint: dict[str, Any]
    request: ProcessingJobRequest

    @property
    def root(self) -> Path:
        return self.output_path / ".remap"

    @property
    def path(self) -> Path:
        return self.root / "checkpoints.json"

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "schema_version": 1,
                "fingerprint": self.fingerprint,
                "request": self.request.to_dict(),
                "steps": {},
            }
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {
                "schema_version": 1,
                "fingerprint": self.fingerprint,
                "request": self.request.to_dict(),
                "steps": {},
            }

    def mark(self, step: str, status: str, outputs: list[str] | None = None, detail: str = "") -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        data = self._load()
        data["fingerprint"] = self.fingerprint
        data["request"] = self.request.to_dict()
        data["updated_at"] = utc_now_iso()
        data.setdefault("steps", {})[step] = {
            "status": status,
            "updated_at": utc_now_iso(),
            "outputs": outputs or [],
            "detail": detail,
        }
        tmp = self.path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        tmp.replace(self.path)


def _cache_entry_dir(key: str) -> Path:
    return CACHE_DIR / key


def restore_cache(request: ProcessingJobRequest, fingerprint: dict[str, Any], output_path: Path) -> dict[str, Any]:
    key = cache_key(request, fingerprint)
    entry_dir = _cache_entry_dir(key)
    restored: list[str] = []
    if not request.skip_existing or not entry_dir.exists():
        return {"key": key, "hit": False, "restored": restored}
    hloc_dir = output_path / "hloc_outputs"
    hloc_dir.mkdir(parents=True, exist_ok=True)
    for name in CACHE_FILES:
        source = entry_dir / name
        if source.exists():
            shutil.copy2(source, hloc_dir / name)
            restored.append(name)
    return {"key": key, "hit": bool(restored), "restored": restored}


def store_cache(request: ProcessingJobRequest, fingerprint: dict[str, Any], output_path: Path) -> dict[str, Any]:
    key = cache_key(request, fingerprint)
    hloc_dir = output_path / "hloc_outputs"
    if not hloc_dir.exists():
        return {"key": key, "stored": [], "size": 0}
    entry_dir = _cache_entry_dir(key)
    entry_dir.mkdir(parents=True, exist_ok=True)
    stored: list[str] = []
    total_size = 0
    for path in hloc_dir.iterdir():
        if path.is_file() and (path.name in CACHE_FILES or path.suffix.lower() in {".h5", ".txt"}):
            destination = entry_dir / path.name
            shutil.copy2(path, destination)
            stored.append(path.name)
            total_size += destination.stat().st_size
    metadata = {
        "key": key,
        "created_at": utc_now_iso(),
        "fingerprint": fingerprint.get("digest"),
        "request": {
            "feature_type": request.feature_type,
            "matcher_type": request.matcher_type,
            "max_keypoints": request.max_keypoints,
            "pairing_mode": request.pairing_mode,
        },
        "files": stored,
        "size": total_size,
    }
    with (entry_dir / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    return {"key": key, "stored": stored, "size": total_size}


def cache_status() -> dict[str, Any]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    total_size = 0
    for entry_dir in CACHE_DIR.iterdir():
        if not entry_dir.is_dir():
            continue
        size = 0
        for item in entry_dir.rglob("*"):
            if item.is_file():
                size += item.stat().st_size
        total_size += size
        metadata_path = entry_dir / "metadata.json"
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                metadata = {}
        entries.append(
            {
                "key": entry_dir.name,
                "path": str(entry_dir),
                "size": size,
                "size_mb": size / (1024**2),
                "modified_at": entry_dir.stat().st_mtime,
                "files": metadata.get("files", []),
                "request": metadata.get("request", {}),
                "created_at": metadata.get("created_at", ""),
            }
        )
    entries.sort(key=lambda item: item["modified_at"], reverse=True)
    return {
        "path": str(CACHE_DIR),
        "entries": entries,
        "total_size": total_size,
        "total_size_mb": total_size / (1024**2),
    }


def clear_cache() -> dict[str, Any]:
    before = cache_status()
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return {"cleared": True, "removed_entries": len(before["entries"]), "removed_size": before["total_size"]}


def output_file_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return 0


def enforce_cache_limit(max_size_gb: float) -> None:
    if max_size_gb <= 0:
        return
    status = cache_status()
    limit = max_size_gb * (1024**3)
    if status["total_size"] <= limit:
        return
    entries = sorted(status["entries"], key=lambda item: item["modified_at"])
    current = status["total_size"]
    for entry in entries:
        if current <= limit:
            break
        shutil.rmtree(entry["path"], ignore_errors=True)
        current -= int(entry["size"])


def touch_manifest(output_path: Path, name: str, payload: dict[str, Any]) -> None:
    root = output_path / ".remap"
    root.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["updated_at"] = utc_now_iso()
    tmp = root / f"{name}.json.tmp"
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    tmp.replace(root / f"{name}.json")
