from __future__ import annotations

from pathlib import Path
import struct


IMAGE_EXTENSIONS = (".exr", ".png", ".jpg", ".jpeg", ".tif", ".tiff")


def read_images_bin(bin_path: Path) -> list[dict]:
    images = []
    with open(bin_path, "rb") as f:
        (num_images,) = struct.unpack("<Q", f.read(8))
        for _ in range(num_images):
            (image_id,) = struct.unpack("<i", f.read(4))
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            (camera_id,) = struct.unpack("<i", f.read(4))
            name_chars = []
            while True:
                ch = f.read(1)
                if ch == b"\x00" or ch == b"":
                    break
                name_chars.append(ch)
            name = b"".join(name_chars).decode("utf-8")
            (num_points2d,) = struct.unpack("<Q", f.read(8))
            points2d_data = f.read(num_points2d * 24) if num_points2d > 0 else b""
            images.append(
                {
                    "image_id": image_id,
                    "qw": qw,
                    "qx": qx,
                    "qy": qy,
                    "qz": qz,
                    "tx": tx,
                    "ty": ty,
                    "tz": tz,
                    "camera_id": camera_id,
                    "name": name,
                    "num_points2d": num_points2d,
                    "points2d_data": points2d_data,
                }
            )
    return images


def write_images_bin(bin_path: Path, images: list[dict]) -> None:
    with open(bin_path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for img in images:
            f.write(struct.pack("<i", img["image_id"]))
            f.write(struct.pack("<4d", img["qw"], img["qx"], img["qy"], img["qz"]))
            f.write(struct.pack("<3d", img["tx"], img["ty"], img["tz"]))
            f.write(struct.pack("<i", img["camera_id"]))
            f.write(img["name"].encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", img["num_points2d"]))
            if img["points2d_data"]:
                f.write(img["points2d_data"])


def normalize_images_bin_for_image_dir(
    bin_path: Path,
    images_dir: Path,
    prefix: str = "images/",
    logger_fn=None,
) -> int:
    if not bin_path.exists() or not images_dir.exists():
        return 0

    available = {p.name.lower(): p.name for p in images_dir.iterdir() if p.is_file()}
    if not available:
        return 0

    images = read_images_bin(bin_path)
    updated = 0
    for img in images:
        basename = Path(img["name"].replace("\\", "/")).name
        stem = Path(basename).stem.lower()
        final_name = basename
        if basename.lower() not in available:
            for ext in IMAGE_EXTENSIONS:
                candidate = f"{stem}{ext}"
                if candidate in available:
                    final_name = available[candidate]
                    break
        next_name = f"{prefix}{final_name}"
        if img["name"] != next_name:
            img["name"] = next_name
            updated += 1

    if updated:
        write_images_bin(bin_path, images)
        if logger_fn:
            logger_fn(f"  -> images.bin normalized for {updated} image path(s)")
    return updated
