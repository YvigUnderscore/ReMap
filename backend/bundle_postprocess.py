from __future__ import annotations

from pathlib import Path
import shutil

from .colmap_images import normalize_images_bin_for_image_dir


MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin", "frames.bin", "rigs.bin")
IMAGE_EXTENSIONS = {".exr", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _copy_sparse_bins(source_dir: Path, sparse_dir: Path, final_dir: Path) -> None:
    for name in MODEL_FILES:
        src = source_dir / name
        if not src.exists():
            src = sparse_dir / name
        if src.exists():
            shutil.copy2(str(src), str(final_dir / name))


def _image_files(directory: Path, extensions: set[str] | None = None) -> list[Path]:
    if not directory.exists() or not directory.is_dir():
        return []
    wanted = extensions or IMAGE_EXTENSIONS
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in wanted
    )


def _first_non_empty(candidates: list[tuple[Path, set[str] | None]]) -> tuple[Path | None, list[Path]]:
    for directory, extensions in candidates:
        files = _image_files(directory, extensions)
        if files:
            return directory, files
    return None, []


def _replace_image_dir(final_images: Path, source_dir: Path | None, files: list[Path]) -> None:
    same_dir = source_dir is not None and final_images.resolve() == source_dir.resolve()
    if same_dir:
        return
    if final_images.exists():
        shutil.rmtree(final_images, ignore_errors=True)
    final_images.mkdir(parents=True, exist_ok=True)
    for src in files:
        shutil.copy2(str(src), str(final_images / src.name))


def normalize_final_bundle(
    base_output: str | Path,
    keep_srgb_png: bool = True,
    use_acescg_exr: bool = True,
) -> Path | None:
    base_path = Path(base_output)
    sparse_dir = base_path / "sparse" / "0"
    if not sparse_dir.exists():
        return None

    legacy_model_dir = sparse_dir / "models" / "0" / "0"
    model_source_dir = legacy_model_dir if (legacy_model_dir / "images.bin").exists() else sparse_dir

    final_dir = base_path / f"{base_path.name}_SfM_Dataset_Output"
    final_images = final_dir / "images"
    final_dir.mkdir(parents=True, exist_ok=True)

    _copy_sparse_bins(model_source_dir, sparse_dir, final_dir)

    if use_acescg_exr:
        source_dir, files = _first_non_empty([
            (final_images, {".exr"}),
            (legacy_model_dir / "images", {".exr"}),
            (legacy_model_dir, {".exr"}),
            (base_path / "images", {".exr"}),
        ])
        if not files:
            source_dir, files = _first_non_empty([
                (base_path / "images", IMAGE_EXTENSIONS - {".exr"}),
                (legacy_model_dir / "images", IMAGE_EXTENSIONS - {".exr"}),
            ])
    else:
        source_dir, files = _first_non_empty([
            (base_path / "images", IMAGE_EXTENSIONS - {".exr"}),
            (legacy_model_dir / "images", IMAGE_EXTENSIONS - {".exr"}),
            (final_images, IMAGE_EXTENSIONS - {".exr"}),
        ])

    if files:
        _replace_image_dir(final_images, source_dir, files)
        if use_acescg_exr and any(path.suffix.lower() == ".exr" for path in files):
            for stale in _image_files(final_images, IMAGE_EXTENSIONS - {".exr"}):
                stale.unlink()

    preview_dir = final_dir / "images_srgb_png"
    if keep_srgb_png and use_acescg_exr:
        preview_source, preview_files = _first_non_empty([
            (base_path / "images", IMAGE_EXTENSIONS - {".exr"}),
            (legacy_model_dir / "images_srgb_png", IMAGE_EXTENSIONS - {".exr"}),
            (legacy_model_dir / "images", IMAGE_EXTENSIONS - {".exr"}),
        ])
        if preview_files and preview_source is not None:
            _replace_image_dir(preview_dir, preview_source, preview_files)
    elif preview_dir.exists():
        shutil.rmtree(preview_dir, ignore_errors=True)

    normalize_images_bin_for_image_dir(final_dir / "images.bin", final_images, "images/")
    return final_dir
