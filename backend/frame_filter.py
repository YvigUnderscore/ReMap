from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any, Callable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def reject_low_quality_frames(
    image_dir: str | Path,
    reject_blurry: bool = False,
    reject_black: bool = False,
    blur_threshold: float = 75.0,
    black_threshold: float = 0.08,
    logger: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    directory = Path(image_dir)
    rejected_dir = directory / "_rejected_frames"
    rejected: list[dict[str, Any]] = []
    checked = 0
    if not directory.exists() or not (reject_blurry or reject_black):
        return {"checked": checked, "rejected": rejected}

    try:
        import cv2
    except Exception as exc:
        if logger:
            logger(f"Frame quality filter unavailable: {exc}")
        return {"checked": checked, "rejected": rejected}

    files = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    for path in files:
        checked += 1
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        mean_value = float(image.mean()) / 255.0
        blur_score = float(cv2.Laplacian(image, cv2.CV_64F).var())
        reasons = []
        if reject_black and mean_value <= black_threshold:
            reasons.append("black")
        if reject_blurry and blur_score <= blur_threshold:
            reasons.append("blurry")
        if not reasons:
            continue
        rejected_dir.mkdir(parents=True, exist_ok=True)
        destination = rejected_dir / path.name
        shutil.move(str(path), destination)
        rejected.append(
            {
                "path": str(destination),
                "name": path.name,
                "reasons": reasons,
                "mean_luma": mean_value,
                "blur_score": blur_score,
            }
        )

    payload = {
        "checked": checked,
        "rejected": rejected,
        "reject_blurry": reject_blurry,
        "reject_black": reject_black,
        "blur_threshold": blur_threshold,
        "black_threshold": black_threshold,
    }
    if rejected:
        with (directory / "rejected_frames.json").open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        if logger:
            logger(f"Frame quality filter rejected {len(rejected)}/{checked} frames")
    elif logger:
        logger(f"Frame quality filter checked {checked} frames; none rejected")
    return payload
