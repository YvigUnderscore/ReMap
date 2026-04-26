from __future__ import annotations

from dataclasses import dataclass
import os
import platform
import time
from pathlib import Path
from typing import Callable

import h5py
import numpy as np


LOMA_MATCHER_TYPES = {"loma_b", "loma_g"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def normalize_loma_matcher(value: str | None) -> str:
    normalized = (value or "").strip().lower().replace("-", "_")
    aliases = {
        "loma": "loma_b",
        "loma_b": "loma_b",
        "lomab": "loma_b",
        "loma_g": "loma_g",
        "lomag": "loma_g",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported LoMa matcher '{value}'. Expected loma_b or loma_g.")
    return aliases[normalized]


def is_loma_matcher(value: str | None) -> bool:
    try:
        return normalize_loma_matcher(value) in LOMA_MATCHER_TYPES
    except ValueError:
        return False


def loma_feature_path(export_dir: Path, matcher_type: str) -> Path:
    return Path(export_dir) / f"features-{normalize_loma_matcher(matcher_type)}.h5"


def loma_matches_path(export_dir: Path, matcher_type: str) -> Path:
    return Path(export_dir) / f"matches-{normalize_loma_matcher(matcher_type)}.h5"


def _names_to_pair(name0: str, name1: str, separator: str = "/") -> str:
    try:
        from hloc.utils.parsers import names_to_pair

        return names_to_pair(name0, name1, separator=separator)
    except Exception:
        return separator.join((name0.replace("/", "-"), name1.replace("/", "-")))


def _has_pair(fd: h5py.File, name0: str, name1: str) -> bool:
    candidates = (
        _names_to_pair(name0, name1),
        _names_to_pair(name1, name0),
        _names_to_pair(name0, name1, separator="_"),
        _names_to_pair(name1, name0, separator="_"),
    )
    return any(candidate in fd for candidate in candidates)


def _list_images(image_dir: Path) -> list[str]:
    image_dir = Path(image_dir)
    images = [
        path.relative_to(image_dir).as_posix()
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images)


def _read_pairs(pairs_path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    with Path(pairs_path).open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                continue
            name0, name1 = parts[0], parts[1]
            dedupe_key = tuple(sorted((name0, name1)))
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            pairs.append((name0, name1))
    return pairs


def _set_if_present(target: object, name: str, value: object) -> None:
    if hasattr(target, name):
        try:
            setattr(target, name, value)
        except Exception:
            pass


def configure_torch_for_loma(force_fp16: bool = True, optimize_triton: bool = True) -> None:
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if optimize_triton:
        cache_root = Path(os.environ.get("REMAP_TORCH_CACHE", Path.home() / ".cache" / "remap-torch"))
        os.environ.setdefault("TRITON_CACHE_DIR", str(cache_root / "triton"))
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_root / "inductor"))
        os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
        os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE", "1")

    try:
        import torch

        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        if torch.cuda.is_available():
            if hasattr(torch.backends, "cuda"):
                cuda_backend = torch.backends.cuda
                for fn_name in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_math_sdp"):
                    fn = getattr(cuda_backend, fn_name, None)
                    if callable(fn):
                        fn(True)
                if hasattr(cuda_backend, "matmul"):
                    cuda_backend.matmul.allow_tf32 = False if force_fp16 else True

        if optimize_triton:
            try:
                import torch._inductor.config as inductor_config

                _set_if_present(inductor_config, "max_autotune", True)
                _set_if_present(inductor_config, "coordinate_descent_tuning", True)
                triton_config = getattr(inductor_config, "triton", None)
                if triton_config is not None:
                    _set_if_present(triton_config, "cudagraphs", True)
                    _set_if_present(triton_config, "unique_kernel_names", True)
                    _set_if_present(triton_config, "persistent_reductions", True)
            except Exception:
                pass
    except Exception:
        pass


@dataclass
class _FeatureRecord:
    keypoints_norm: object
    descriptors: object
    pixel_keypoints: np.ndarray
    width: int
    height: int


class LoMaMatcher:
    """HLoc-compatible wrapper around LoMa-B and LoMa-G."""

    def __init__(
        self,
        matcher_type: str = "loma_b",
        max_keypoints: int = 4096,
        *,
        filter_threshold: float | None = None,
        force_fp16: bool = True,
        optimize_triton: bool = True,
        compile_model: bool | None = None,
        logger: Callable[[str], None] | None = None,
        cancel_check: Callable[[], None] | None = None,
    ) -> None:
        self.matcher_type = normalize_loma_matcher(matcher_type)
        self.max_keypoints = int(max_keypoints)
        self.filter_threshold = filter_threshold
        self.force_fp16 = force_fp16
        self.optimize_triton = optimize_triton
        self.compile_model = self._default_compile_enabled() if compile_model is None else bool(compile_model)
        self.logger = logger or (lambda message: None)
        self.cancel_check = cancel_check or (lambda: None)
        self._model = None
        self._torch = None
        self._loma_impl = None
        self._records: dict[str, _FeatureRecord] = {}
        self.last_stats: dict[str, object] = {}

    @staticmethod
    def _default_compile_enabled() -> bool:
        value = os.environ.get("REMAP_LOMA_COMPILE")
        if value is not None:
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return platform.system().lower() == "linux"

    @property
    def output_name(self) -> str:
        return self.matcher_type

    def _check_cancelled(self) -> None:
        self.cancel_check()

    def _ensure_model(self):
        if self._model is not None:
            return self._model

        configure_torch_for_loma(self.force_fp16, self.optimize_triton)
        try:
            import torch
            from loma import LoMa, LoMaB, LoMaG
            import loma.device as loma_device
            import loma.loma as loma_impl
        except ImportError as exc:
            raise RuntimeError(
                "LoMa is not installed. Run the installer again or install "
                "git+https://github.com/davnords/LoMa.git#egg=lomatch."
            ) from exc

        if self.force_fp16:
            loma_device.amp_dtype = torch.float16
            loma_impl.amp_dtype = torch.float16

        cfg_class = LoMaB if self.matcher_type == "loma_b" else LoMaG
        cfg = cfg_class(
            num_keypoints=self.max_keypoints,
            mp=True,
            compile=self.compile_model,
        )
        if self.filter_threshold is not None:
            cfg = cfg_class(
                num_keypoints=self.max_keypoints,
                filter_threshold=float(self.filter_threshold),
                mp=True,
                compile=self.compile_model,
            )

        self.logger(
            f"LoMa {self.matcher_type}: loading weights, FP16 autocast=on, "
            f"Triton/Inductor tuning={'on' if self.optimize_triton else 'off'}, "
            f"compile={'on' if self.compile_model else 'off'}"
        )
        model = LoMa(cfg).eval()
        self._model = model
        self._torch = torch
        self._loma_impl = loma_impl

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            major, minor = torch.cuda.get_device_capability()
            self.logger(
                f"LoMa CUDA device: {props.name} sm_{major}{minor}, "
                f"{props.total_memory / (1024 ** 3):.1f} GB VRAM"
            )
        return self._model

    def _feature_file_is_compatible(self, feature_path: Path, names: list[str]) -> bool:
        if not feature_path.exists():
            return False
        try:
            with h5py.File(str(feature_path), "r", libver="latest") as fd:
                if fd.attrs.get("remap_matcher") != "loma":
                    return False
                if fd.attrs.get("remap_loma_variant") != self.matcher_type:
                    return False
                for name in names:
                    if name not in fd:
                        return False
                    group = fd[name]
                    for key in ("keypoints", "loma_keypoints", "descriptors", "image_size"):
                        if key not in group:
                            return False
                return True
        except Exception:
            return False

    def extract_features(
        self,
        image_dir: Path,
        feature_path: Path,
        *,
        image_names: list[str] | None = None,
        overwrite: bool = False,
    ) -> Path:
        image_dir = Path(image_dir)
        feature_path = Path(feature_path)
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        names = image_names or _list_images(image_dir)
        if not names:
            raise ValueError(f"No images found in {image_dir}")
        self._check_cancelled()

        if overwrite and feature_path.exists():
            feature_path.unlink()

        if self._feature_file_is_compatible(feature_path, names):
            self._check_cancelled()
            self.logger(f"LoMa {self.matcher_type}: feature cache already complete ({len(names)} images)")
            return feature_path

        model = self._ensure_model()
        self._check_cancelled()
        torch = self._torch
        assert torch is not None
        assert self._loma_impl is not None

        started = time.perf_counter()
        total_keypoints = 0
        with h5py.File(str(feature_path), "a", libver="latest") as fd:
            fd.attrs["remap_matcher"] = "loma"
            fd.attrs["remap_loma_variant"] = self.matcher_type
            fd.attrs["max_keypoints"] = self.max_keypoints
            for index, name in enumerate(names, start=1):
                self._check_cancelled()
                if name in fd:
                    del fd[name]
                image_path = image_dir / name
                with torch.inference_mode():
                    keypoints_norm, descriptors, height, width = model.detect_and_describe(
                        str(image_path),
                        num_keypoints=self.max_keypoints,
                    )
                    pixel_keypoints = self._loma_impl.to_pixel_coords(
                        keypoints_norm[0].float(),
                        height,
                        width,
                    ).detach().cpu().numpy().astype(np.float32)
                self._check_cancelled()

                grp = fd.create_group(name)
                keypoint_set = grp.create_dataset("keypoints", data=pixel_keypoints)
                keypoint_set.attrs["uncertainty"] = np.float32(1.0)
                grp.create_dataset("scores", data=np.ones((pixel_keypoints.shape[0],), dtype=np.float16))
                grp.create_dataset("image_size", data=np.array([width, height], dtype=np.float32))
                grp.create_dataset("loma_keypoints", data=keypoints_norm[0].detach().cpu().numpy().astype(np.float16))
                grp.create_dataset("descriptors", data=descriptors[0].detach().cpu().numpy().astype(np.float16))
                total_keypoints += int(pixel_keypoints.shape[0])

                self._records[name] = _FeatureRecord(
                    keypoints_norm=keypoints_norm.detach().cpu().to(torch.float16),
                    descriptors=descriptors.detach().cpu().to(torch.float16),
                    pixel_keypoints=pixel_keypoints,
                    width=int(width),
                    height=int(height),
                )
                if index == 1 or index % 25 == 0 or index == len(names):
                    self.logger(f"LoMa {self.matcher_type}: described {index}/{len(names)} images")

        elapsed = max(time.perf_counter() - started, 1e-6)
        self.last_stats["features"] = {
            "images": len(names),
            "keypoints": total_keypoints,
            "seconds": elapsed,
            "ms_per_image": 1000.0 * elapsed / max(len(names), 1),
            "keypoints_per_image": total_keypoints / max(len(names), 1),
        }
        self.logger(
            f"LoMa {self.matcher_type}: {total_keypoints:,} keypoints over {len(names)} images "
            f"in {elapsed:.2f}s ({1000.0 * elapsed / max(len(names), 1):.1f} ms/image)"
        )
        return feature_path

    def _load_record(self, feature_path: Path, name: str) -> _FeatureRecord:
        if name in self._records:
            return self._records[name]
        torch = self._torch
        if torch is None:
            self._ensure_model()
            torch = self._torch
        assert torch is not None
        with h5py.File(str(feature_path), "r", libver="latest") as fd:
            grp = fd[name]
            image_size = grp["image_size"].__array__()
            record = _FeatureRecord(
                keypoints_norm=torch.from_numpy(grp["loma_keypoints"].__array__()).to(torch.float16).unsqueeze(0),
                descriptors=torch.from_numpy(grp["descriptors"].__array__()).to(torch.float16).unsqueeze(0),
                pixel_keypoints=grp["keypoints"].__array__().astype(np.float32),
                width=int(image_size[0]),
                height=int(image_size[1]),
            )
        self._records[name] = record
        return record

    def match_pairs(
        self,
        pairs_path: Path,
        feature_path: Path,
        matches_path: Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        pairs_path = Path(pairs_path)
        feature_path = Path(feature_path)
        matches_path = Path(matches_path)
        matches_path.parent.mkdir(parents=True, exist_ok=True)
        self._check_cancelled()
        pairs = _read_pairs(pairs_path)
        if not pairs:
            raise ValueError(f"No pairs found in {pairs_path}")
        self._check_cancelled()

        model = self._ensure_model()
        self._check_cancelled()
        torch = self._torch
        assert torch is not None
        assert self._loma_impl is not None
        device = next(model.parameters()).device
        threshold = self.filter_threshold
        if threshold is None:
            threshold = float(getattr(model.cfg, "filter_threshold", 0.1))

        pending = pairs
        if matches_path.exists() and not overwrite:
            with h5py.File(str(matches_path), "r", libver="latest") as fd:
                pending = [(a, b) for a, b in pairs if not _has_pair(fd, a, b)]
        if not pending:
            self.logger(f"LoMa {self.matcher_type}: matches already complete ({len(pairs)} pairs)")
            return matches_path

        if overwrite and matches_path.exists():
            matches_path.unlink()

        started = time.perf_counter()
        total_matches = 0
        with h5py.File(str(matches_path), "a", libver="latest") as fd:
            fd.attrs["remap_matcher"] = "loma"
            fd.attrs["remap_loma_variant"] = self.matcher_type
            fd.attrs["filter_threshold"] = float(threshold)
            for index, (name0, name1) in enumerate(pending, start=1):
                self._check_cancelled()
                rec0 = self._load_record(feature_path, name0)
                rec1 = self._load_record(feature_path, name1)
                self._check_cancelled()
                kpts0 = rec0.keypoints_norm.to(device=device, dtype=torch.float16, non_blocking=True)
                kpts1 = rec1.keypoints_norm.to(device=device, dtype=torch.float16, non_blocking=True)
                desc0 = rec0.descriptors.to(device=device, dtype=torch.float16, non_blocking=True)
                desc1 = rec1.descriptors.to(device=device, dtype=torch.float16, non_blocking=True)

                with torch.inference_mode():
                    scores = model(kpts0, kpts1, desc0, desc1)["scores"]
                    m0, _, mscores0, _ = self._loma_impl.filter_matches(scores, float(threshold))
                    matches0 = m0[0].detach().cpu().numpy().astype(np.int32)
                    match_scores0 = mscores0[0].detach().cpu().numpy().astype(np.float16)
                self._check_cancelled()

                pair_name = _names_to_pair(name0, name1)
                if pair_name in fd:
                    del fd[pair_name]
                grp = fd.create_group(pair_name)
                grp.create_dataset("matches0", data=matches0)
                grp.create_dataset("matching_scores0", data=match_scores0)
                total_matches += int(np.count_nonzero(matches0 > -1))

                if index == 1 or index % 50 == 0 or index == len(pending):
                    self.logger(f"LoMa {self.matcher_type}: matched {index}/{len(pending)} pairs")

        elapsed = max(time.perf_counter() - started, 1e-6)
        self.last_stats["matches"] = {
            "pairs": len(pending),
            "matches": total_matches,
            "seconds": elapsed,
            "ms_per_pair": 1000.0 * elapsed / max(len(pending), 1),
            "matches_per_pair": total_matches / max(len(pending), 1),
        }
        self.logger(
            f"LoMa {self.matcher_type}: {total_matches:,} matches over {len(pending)} pairs "
            f"in {elapsed:.2f}s ({1000.0 * elapsed / max(len(pending), 1):.1f} ms/pair)"
        )
        return matches_path

    def run(
        self,
        image_dir: Path,
        pairs_path: Path,
        export_dir: Path,
        *,
        overwrite: bool = False,
    ) -> tuple[Path, Path]:
        features = loma_feature_path(Path(export_dir), self.matcher_type)
        matches = loma_matches_path(Path(export_dir), self.matcher_type)
        self.extract_features(image_dir, features, overwrite=overwrite)
        self.match_pairs(pairs_path, features, matches, overwrite=overwrite)
        return features, matches
