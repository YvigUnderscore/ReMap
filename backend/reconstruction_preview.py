from __future__ import annotations

from pathlib import Path
import struct
from typing import Any

from .colmap_images import read_images_bin


CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def _read_cameras(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    cameras = []
    with path.open("rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            return []
        (count,) = struct.unpack("<Q", raw)
        for _ in range(count):
            camera_id, model_id = struct.unpack("<ii", fh.read(8))
            width, height = struct.unpack("<QQ", fh.read(16))
            model_name, param_count = CAMERA_MODELS.get(model_id, (f"MODEL_{model_id}", 0))
            params = list(struct.unpack(f"<{param_count}d", fh.read(8 * param_count))) if param_count else []
            cameras.append(
                {
                    "camera_id": camera_id,
                    "model": model_name,
                    "width": width,
                    "height": height,
                    "params": params,
                }
            )
    return cameras


def _read_points(path: Path, max_points: int = 6000) -> dict[str, Any]:
    if not path.exists():
        return {"count": 0, "sample": [], "mean_error": None}
    sample = []
    errors = []
    with path.open("rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            return {"count": 0, "sample": [], "mean_error": None}
        (count,) = struct.unpack("<Q", raw)
        stride = max(1, count // max_points) if count else 1
        for index in range(count):
            point_id = struct.unpack("<Q", fh.read(8))[0]
            x, y, z = struct.unpack("<3d", fh.read(24))
            r, g, b = struct.unpack("<BBB", fh.read(3))
            error = struct.unpack("<d", fh.read(8))[0]
            track_len = struct.unpack("<Q", fh.read(8))[0]
            fh.seek(track_len * 8, 1)
            if len(errors) < 20000:
                errors.append(error)
            if index % stride == 0 and len(sample) < max_points:
                sample.append({"id": point_id, "xyz": [x, y, z], "rgb": [r, g, b], "error": error})
    return {
        "count": count,
        "sample": sample,
        "mean_error": sum(errors) / len(errors) if errors else None,
    }


def _candidate_sparse_dirs(output_path: Path) -> list[Path]:
    return [
        output_path / f"{output_path.name}_SfM_Dataset_Output",
        output_path / "sparse" / "0" / "models" / "0" / "0",
        output_path / "sparse" / "0",
        output_path / "live_reconstruction",
    ]


def build_reconstruction_preview(output_path: str | Path) -> dict[str, Any]:
    root = Path(output_path)
    sparse_dir = next(
        (
            candidate
            for candidate in _candidate_sparse_dirs(root)
            if (candidate / "images.bin").exists() or (candidate / "points3D.bin").exists()
        ),
        None,
    )
    if sparse_dir is None:
        return {
            "available": False,
            "sparse_dir": "",
            "cameras": [],
            "images": [],
            "points": {"count": 0, "sample": [], "mean_error": None},
            "stats": {"camera_count": 0, "image_count": 0, "registered_images": 0, "point_count": 0},
        }
    cameras = _read_cameras(sparse_dir / "cameras.bin")
    images = []
    images_bin = sparse_dir / "images.bin"
    if images_bin.exists():
        try:
            images = read_images_bin(images_bin)
        except Exception:
            images = []
    points = _read_points(sparse_dir / "points3D.bin")
    image_sample = [
        {
            "image_id": item["image_id"],
            "name": item["name"],
            "camera_id": item["camera_id"],
            "t": [item["tx"], item["ty"], item["tz"]],
            "q": [item["qw"], item["qx"], item["qy"], item["qz"]],
            "num_points2d": item["num_points2d"],
        }
        for item in images[:500]
    ]
    return {
        "available": True,
        "sparse_dir": str(sparse_dir),
        "cameras": cameras,
        "images": image_sample,
        "points": points,
        "stats": {
            "camera_count": len(cameras),
            "image_count": len(images),
            "registered_images": len(images),
            "point_count": points["count"],
            "mean_reprojection_error": points["mean_error"],
        },
    }
