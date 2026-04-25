from __future__ import annotations

import os
import platform
import shutil
import sys

from .models import SystemCapabilities, global_mapper_available
from .probe_service import build_option_payload


def detect_capabilities() -> SystemCapabilities:
    try:
        import torch

        torch_available = True
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        torch_available = False
        cuda_available = False

    try:
        import OpenImageIO  # noqa: F401

        openimageio_available = True
    except Exception:
        openimageio_available = False

    return SystemCapabilities(
        cpu_count=os.cpu_count() or 1,
        glomap_available=global_mapper_available(),
        ffmpeg_available=shutil.which("ffmpeg") is not None,
        ffprobe_available=shutil.which("ffprobe") is not None,
        openimageio_available=openimageio_available,
        torch_available=torch_available,
        cuda_available=cuda_available,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        ocio_env=os.environ.get("OCIO", ""),
    )


def detect_capabilities_payload() -> dict:
    payload = detect_capabilities().to_dict()
    payload.update(build_option_payload())
    return payload
