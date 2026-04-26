from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from importlib import metadata
import os
from pathlib import Path
import subprocess
import sys
from threading import RLock, Thread
from typing import Any, Callable


ROOT_DIR = Path(__file__).resolve().parent.parent
REQUIREMENTS_PATH = ROOT_DIR / "requirements.txt"
REQUIREMENTS_LOCK_PATH = ROOT_DIR / "requirements.lock.txt"
LOMA_LOCKED_URL = "git+https://github.com/davnords/LoMa.git@9105854833f55d18194d0505d913f0a74b194ef0#egg=lomatch"

TRACKED_PACKAGES = [
    "torch",
    "torchvision",
    "hloc",
    "lightglue",
    "lomatch",
    "pycolmap",
    "kornia",
    "numpy",
    "scipy",
    "h5py",
    "opencv-python-headless",
    "OpenImageIO",
    "Flask",
    "psutil",
]

_TASK_LOCK = RLock()
_TASK_STATE: dict[str, Any] = {
    "running": False,
    "action": "",
    "status": "idle",
    "started_at": None,
    "finished_at": None,
    "returncode": None,
    "message": "",
    "log": [],
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_log(message: str) -> None:
    text = str(message).rstrip()
    if not text:
        return
    with _TASK_LOCK:
        _TASK_STATE["log"].append({"timestamp": _utc_now(), "message": text})
        _TASK_STATE["log"] = _TASK_STATE["log"][-600:]


def _set_task(**changes: Any) -> None:
    with _TASK_LOCK:
        _TASK_STATE.update(changes)


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def _package_direct_url(name: str) -> str:
    try:
        distribution = metadata.distribution(name)
        direct_url = distribution.read_text("direct_url.json")
        return direct_url or ""
    except Exception:
        return ""


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        try:
            total += item.stat().st_size
        except OSError:
            pass
    return total


def _model_cache_status() -> dict[str, Any]:
    torch_hub_dir = ""
    checkpoints: list[dict[str, Any]] = []
    try:
        import torch

        torch_hub_dir = torch.hub.get_dir()
        checkpoint_dir = Path(torch_hub_dir) / "checkpoints"
        for path in sorted(checkpoint_dir.glob("*")):
            if not path.is_file() or path.suffix.lower() not in {".pt", ".pth", ".bin"}:
                continue
            stat = path.stat()
            checkpoints.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "size_mb": stat.st_size / (1024**2),
                    "modified_at": stat.st_mtime,
                }
            )
    except Exception:
        pass

    superglue_dir = ROOT_DIR / "SuperGluePretrainedNetwork" / "models" / "weights"
    superglue_weights = []
    if superglue_dir.exists():
        for path in sorted(superglue_dir.glob("*.pth")):
            stat = path.stat()
            superglue_weights.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "size_mb": stat.st_size / (1024**2),
                    "modified_at": stat.st_mtime,
                }
            )

    cache_size = _dir_size(Path(torch_hub_dir)) if torch_hub_dir else 0
    cache_size += _dir_size(superglue_dir)
    return {
        "torch_hub_dir": torch_hub_dir,
        "torch_hub_checkpoints": checkpoints,
        "superglue_weights": superglue_weights,
        "total_size_mb": cache_size / (1024**2),
    }


def dependency_status() -> dict[str, Any]:
    packages = []
    for name in TRACKED_PACKAGES:
        version = _package_version(name)
        packages.append(
            {
                "name": name,
                "installed": version is not None,
                "version": version or "",
                "direct_url": _package_direct_url(name),
            }
        )

    with _TASK_LOCK:
        task = deepcopy(_TASK_STATE)
    return {
        "python": sys.executable,
        "requirements": str(REQUIREMENTS_PATH),
        "requirements_lock": str(REQUIREMENTS_LOCK_PATH),
        "loma_locked_url": LOMA_LOCKED_URL,
        "packages": packages,
        "models": _model_cache_status(),
        "task": task,
    }


def _run_command(command: list[str], env: dict[str, str] | None = None) -> int:
    _append_log(f"$ {' '.join(command)}")
    process = subprocess.Popen(
        command,
        cwd=str(ROOT_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        _append_log(line)
    return int(process.wait())


def _pip_commands(action: str) -> list[list[str]]:
    python = sys.executable
    base = [
        [python, "-m", "pip", "install", "--upgrade", "pip"],
    ]
    install_args = [python, "-m", "pip", "install"]
    if action == "update_packages":
        install_args.append("--upgrade")
    install_args.extend(["-r", str(REQUIREMENTS_PATH)])
    base.append(install_args)
    base.append([python, "-m", "pip", "install", "--ignore-requires-python", "dataclasses==0.8"])
    base.append([python, "-m", "pip", "install", "--no-deps", "--force-reinstall", LOMA_LOCKED_URL])
    return base


def _run_package_action(action: str) -> int:
    env = os.environ.copy()
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    env.setdefault("CUDA_MODULE_LOADING", "LAZY")
    for command in _pip_commands(action):
        returncode = _run_command(command, env=env)
        if returncode != 0:
            return returncode
    _append_log("Dependency baseline installed. Restart ReMap to load newly installed packages.")
    return 0


def _download_core_models() -> None:
    import torch
    from hloc import extract_features, extractors, match_features, matchers
    from hloc.utils.base_model import dynamic_load

    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_names = ["superpoint_aachen", "disk", "aliked-n16"]
    matcher_names = ["superpoint+lightglue", "disk+lightglue", "superglue"]
    for name in feature_names:
        conf = deepcopy(extract_features.confs[name])
        model_class = dynamic_load(extractors, conf["model"]["name"])
        _append_log(f"Preparing feature model {name}...")
        model = model_class(conf["model"]).eval().to(device)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for name in matcher_names:
        conf = deepcopy(match_features.confs[name])
        model_class = dynamic_load(matchers, conf["model"]["name"])
        _append_log(f"Preparing matcher model {name}...")
        model = model_class(conf["model"]).eval().to(device)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _download_loma_model(variant: str, logger: Callable[[str], None]) -> None:
    from .loma_matcher import LoMaMatcher

    matcher = LoMaMatcher(variant, max_keypoints=1024, compile_model=False, logger=logger)
    matcher._ensure_model()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _run_model_action(action: str) -> int:
    try:
        if action == "download_core_models":
            _download_core_models()
        elif action == "download_loma_b":
            _download_loma_model("loma_b", _append_log)
        elif action == "download_loma_g":
            _download_loma_model("loma_g", _append_log)
        elif action == "download_all_models":
            _download_core_models()
            _download_loma_model("loma_b", _append_log)
            _download_loma_model("loma_g", _append_log)
        else:
            _append_log(f"Unknown model action: {action}")
            return 2
    except Exception as exc:
        _append_log(f"ERROR: {exc}")
        return 1
    _append_log("Model cache is ready.")
    return 0


def _run_action(action: str) -> None:
    _set_task(
        running=True,
        action=action,
        status="running",
        started_at=_utc_now(),
        finished_at=None,
        returncode=None,
        message="",
        log=[],
    )
    try:
        if action in {"install_packages", "update_packages"}:
            returncode = _run_package_action(action)
        else:
            returncode = _run_model_action(action)
        _set_task(
            running=False,
            status="completed" if returncode == 0 else "failed",
            finished_at=_utc_now(),
            returncode=returncode,
            message="Done" if returncode == 0 else f"Task failed with code {returncode}",
        )
    except Exception as exc:
        _append_log(f"ERROR: {exc}")
        _set_task(
            running=False,
            status="failed",
            finished_at=_utc_now(),
            returncode=1,
            message=str(exc),
        )


def start_dependency_action(action: str) -> dict[str, Any]:
    allowed = {
        "install_packages",
        "update_packages",
        "download_core_models",
        "download_loma_b",
        "download_loma_g",
        "download_all_models",
    }
    if action not in allowed:
        raise ValueError(f"Unknown dependency action '{action}'")
    with _TASK_LOCK:
        if _TASK_STATE["running"]:
            raise RuntimeError("A dependency task is already running")
    thread = Thread(target=_run_action, args=(action,), daemon=True)
    thread.start()
    return dependency_status()
