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
        if cuda_available:
            index = torch.cuda.current_device()
            torch_device_name = torch.cuda.get_device_name(index)
            major, minor = torch.cuda.get_device_capability(index)
            torch_compute_capability = f"sm_{major}{minor}"
        else:
            torch_device_name = ""
            torch_compute_capability = ""
    except Exception:
        torch_available = False
        cuda_available = False
        torch_device_name = ""
        torch_compute_capability = ""

    try:
        import loma  # noqa: F401

        loma_available = True
    except Exception:
        loma_available = False

    try:
        import triton  # noqa: F401

        triton_available = True
    except Exception:
        triton_available = False

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
        loma_available=loma_available,
        triton_available=triton_available,
        torch_device_name=torch_device_name,
        torch_compute_capability=torch_compute_capability,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        ocio_env=os.environ.get("OCIO", ""),
    )


def detect_capabilities_payload() -> dict:
    payload = detect_capabilities().to_dict()
    payload.update(build_option_payload())
    return payload
