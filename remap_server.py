#!/usr/bin/env python3
"""
remap_server.py — REST API server for ReMap.

Allows the ReScan iOS app (or any authenticated client) to upload datasets
and trigger the ReMap processing pipeline remotely.

Can be started:
  1. Standalone:  python remap_server.py [--host 0.0.0.0] [--port 5000]
  2. From the ReMap GUI via the built-in server toggle.

Security: All endpoints (except /api/v1/health) require a Bearer API key
passed in the Authorization header.
"""

import argparse
import json
import logging
import multiprocessing
import os
import secrets
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, request, jsonify, send_file, abort

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
API_VERSION = "v1"
DEFAULT_PORT = 5000
DEFAULT_HOST = "0.0.0.0"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "remap_server" / "uploads"
JOBS_DIR = Path(tempfile.gettempdir()) / "remap_server" / "jobs"
MAX_CONTENT_LENGTH = 10 * 1024 * 1024 * 1024  # 10 GB
# Default video FPS assumed for ReScan iPhone captures when ffprobe is unavailable
_DEFAULT_NATIVE_FPS = 60.0

logger = logging.getLogger("remap_server")

# ---------------------------------------------------------------------------
#  Colorspace support
# ---------------------------------------------------------------------------

# Internal pipeline working colorspace (linear light sRGB).
_INTERNAL_COLORSPACE = "Linear"

# Supported image file extensions for EXR/image-sequence datasets.
_IMAGE_EXTS = {".exr", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# Accepted colorspace identifiers → canonical OIIO colorspace name.
SUPPORTED_COLORSPACES: dict[str, str] = {
    "linear":      "Linear",
    "srgb":        "sRGB",
    "acescg":      "ACEScg",
    "aces2065-1":  "ACES2065-1",
    "rec709":      "Rec. 709",
    "log":         "Log",
    "raw":         "Raw",
}

# ---------------------------------------------------------------------------
#  Job store  (in-memory, thread-safe)
# ---------------------------------------------------------------------------
_jobs: dict = {}
_jobs_lock = threading.Lock()


def _new_job(dataset_dir: Path, settings: dict) -> dict:
    job_id = uuid.uuid4().hex[:12]
    job = {
        "job_id": job_id,
        "status": "queued",          # queued | processing | completed | failed | cancelled
        "progress": 0,
        "current_step": "",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(JOBS_DIR / job_id / "output"),
        "settings": settings,
        "error": None,
        "log": [],
    }
    with _jobs_lock:
        _jobs[job_id] = job
    return job


def _update_job(job_id: str, **kwargs):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)
            _jobs[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()


def _get_job(job_id: str) -> dict | None:
    with _jobs_lock:
        return _jobs.get(job_id, None)


def _append_log(job_id: str, message: str):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]["log"].append(
                {"time": datetime.now(timezone.utc).isoformat(), "message": message}
            )


def _rename_images_in_reconstruction(sfm_dir: Path, old_ext: str, new_ext: str,
                                      logger_fn=None) -> None:
    """Rename image file extensions in a COLMAP sparse reconstruction."""
    try:
        import pycolmap
        recon = pycolmap.Reconstruction(str(sfm_dir))
        renamed = 0
        for image_id, image in recon.images.items():
            if image.name.endswith(old_ext):
                image.name = image.name[:-len(old_ext)] + new_ext
                renamed += 1
        if renamed > 0:
            recon.write(str(sfm_dir))
            if logger_fn:
                logger_fn(f"  → Sparse model updated: {renamed} image reference(s) renamed "
                          f"({old_ext} → {new_ext})")
    except Exception as exc:
        if logger_fn:
            logger_fn(f"  ⚠ Could not update sparse model image names: {exc}")


def _prefix_images_in_reconstruction(sfm_dir: Path, prefix: str,
                                      logger_fn=None) -> None:
    """Prepend a path prefix to image names in a COLMAP sparse reconstruction."""
    try:
        import pycolmap
        recon = pycolmap.Reconstruction(str(sfm_dir))
        updated = 0
        for image_id, image in recon.images.items():
            if not image.name.startswith(prefix):
                image.name = prefix + image.name
                updated += 1
        if updated > 0:
            recon.write(str(sfm_dir))
            if logger_fn:
                logger_fn(f"  → Sparse model updated: {updated} image path(s) prefixed "
                          f"with '{prefix}'")
    except Exception as exc:
        if logger_fn:
            logger_fn(f"  ⚠ Could not update sparse model image paths: {exc}")


def _remap_exr_sources(stray_result: dict, sfm_dir: Path, images_dir: Path,
                       output_colorspace: str | None, logger_fn=None) -> None:
    """Post-processing: replace intermediate PNGs with colorspace-converted EXR files.

    When the source dataset contained EXR images (``rgb/`` directory), SfM was
    run on PNG copies.  After reconstruction this step:
    1. Copies the original EXR files from the source ``rgb/`` directory into
       ``images/``.
    2. Applies the requested output OCIO colorspace conversion to the EXR
       files (if any).
    3. Rewrites the sparse model so image names reference ``.exr`` instead of
       ``.png``.
    4. Deletes the intermediate PNG files from ``images/``.
    """
    if not stray_result.get("has_exr_source", False):
        return

    frame_to_filename = stray_result.get("frame_to_filename", {})
    input_dir = stray_result.get("input_dir")
    if not input_dir or not frame_to_filename:
        return

    input_dir = Path(input_dir)
    rgb_dir = input_dir / "rgb"

    if logger_fn:
        logger_fn("  EXR remapping: copying original EXR files to images/")

    # 1. Copy original EXR files
    copied = 0
    for frame_idx, png_name in frame_to_filename.items():
        exr_src = rgb_dir / f"{frame_idx:06d}.exr"
        if not exr_src.exists():
            if logger_fn:
                logger_fn(f"  ⚠ Missing source EXR: {exr_src.name}")
            continue
        exr_dst_name = png_name.removesuffix(".png") + ".exr"
        exr_dst = images_dir / exr_dst_name
        shutil.copy2(exr_src, exr_dst)
        copied += 1

    if logger_fn:
        logger_fn(f"  → {copied} EXR file(s) copied to images/")

    # 2. Apply output colorspace conversion to EXR files
    if output_colorspace:
        import OpenImageIO as oiio
        to_cs = SUPPORTED_COLORSPACES[output_colorspace]
        if to_cs != _INTERNAL_COLORSPACE:
            converted = 0
            for frame_idx, png_name in frame_to_filename.items():
                exr_dst_name = png_name.removesuffix(".png") + ".exr"
                exr_dst = images_dir / exr_dst_name
                if not exr_dst.exists():
                    continue
                buf = oiio.ImageBuf(str(exr_dst))
                if buf.has_error:
                    if logger_fn:
                        logger_fn(f"  ⚠ Could not read {exr_dst.name} for colorspace conversion")
                    continue
                result = oiio.ImageBufAlgo.colorconvert(buf, _INTERNAL_COLORSPACE, to_cs)
                if result.has_error:
                    if logger_fn:
                        logger_fn(f"  ⚠ Colorspace conversion failed for {exr_dst.name}: "
                                  f"{result.geterror()}")
                    continue
                result.write(str(exr_dst))
                converted += 1
            if logger_fn:
                logger_fn(f"  → {converted} EXR file(s) converted "
                          f"from '{_INTERNAL_COLORSPACE}' to '{to_cs}'")

    # 3. Rename image entries in the sparse model (png → exr)
    _rename_images_in_reconstruction(sfm_dir, ".png", ".exr", logger_fn)

    # 4. Delete intermediate PNG files
    deleted = 0
    for _, png_name in frame_to_filename.items():
        png_path = images_dir / png_name
        if png_path.exists():
            png_path.unlink()
            deleted += 1
    if logger_fn:
        logger_fn(f"  → {deleted} intermediate PNG file(s) removed")
        logger_fn("  ✓ EXR remapping complete")


def _convert_images_colorspace(images_dir: Path, from_space: str, to_space: str,
                                logger_fn=None) -> int:
    """Convert all images in *images_dir* from *from_space* to *to_space*.

    Uses OpenImageIO for the colorspace transformation.
    Returns the number of images converted.
    """
    if from_space == to_space:
        return 0

    import OpenImageIO as oiio

    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr"}
    image_files = [p for p in images_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in extensions]
    converted = 0
    for img_path in image_files:
        buf = oiio.ImageBuf(str(img_path))
        if buf.has_error:
            if logger_fn:
                logger_fn(f"  ⚠ Could not read {img_path.name}, skipping colorspace conversion")
            continue
        result = oiio.ImageBufAlgo.colorconvert(buf, from_space, to_space)
        if result.has_error:
            if logger_fn:
                logger_fn(f"  ⚠ Colorspace conversion failed for {img_path.name}: {result.geterror()}")
            continue
        result.write(str(img_path))
        converted += 1
    return converted

# ---------------------------------------------------------------------------
#  Processing thread  (imports heavy libs lazily)
# ---------------------------------------------------------------------------

def _run_job(job_id: str):
    """Run the ReMap pipeline for a single ReScan dataset."""
    job = _get_job(job_id)
    if job is None:
        return

    _update_job(job_id, status="processing", progress=0, current_step="Initialising")
    _append_log(job_id, "Pipeline started")

    dataset_dir = Path(job["dataset_dir"])
    output_dir = Path(job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = job["settings"]

    try:
        # ── Lazy imports (heavy ML / SfM libraries) ──
        import torch
        from hloc import extract_features, match_features, reconstruction, pairs_from_exhaustive
        import pycolmap
        from sfm_runner import run_sfm_with_live_export
        from stray_to_colmap import convert_stray_to_colmap

        def job_logger(msg):
            _append_log(job_id, msg)
            logger.info("[job %s] %s", job_id, msg)

        cancelled = threading.Event()

        def cancel_check():
            if cancelled.is_set():
                raise RuntimeError("Job cancelled")

        # ── Settings with defaults ──
        try:
            fps = float(settings.get("fps", 4.0))
            max_keypoints = int(settings.get("max_keypoints", 4096))
            num_threads = int(settings.get("num_threads", min(multiprocessing.cpu_count(), 16)))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid numeric setting: {exc}") from exc

        feature_type = settings.get("feature_type", "superpoint_aachen")
        matcher_type = settings.get("matcher_type", "superpoint+lightglue")
        camera_model = settings.get("camera_model", "OPENCV")
        mapper_type = settings.get("mapper_type", "COLMAP")
        stray_approach = settings.get("stray_approach", "full_sfm")
        pairing_mode = settings.get("pairing_mode", "sequential")
        color_pipeline = settings.get("color_pipeline", "None")
        input_colorspace = settings.get("input_colorspace")
        output_colorspace = settings.get("output_colorspace")

        # Map approach label
        if "known" in stray_approach.lower() or "arkit" in stray_approach.lower():
            stray_mode = "known_poses"
        else:
            stray_mode = "full_sfm"

        images_dir = output_dir / "images"
        hloc_dir = output_dir / "hloc_outputs"
        sfm_dir = output_dir / "sparse" / "0"
        images_dir.mkdir(parents=True, exist_ok=True)
        hloc_dir.mkdir(parents=True, exist_ok=True)
        sfm_dir.mkdir(parents=True, exist_ok=True)

        # ── Step 1: Stray → COLMAP conversion ──
        _update_job(job_id, progress=10, current_step="ReScan → COLMAP")
        job_logger("Step 1/5 — ReScan → COLMAP conversion")

        # Detect native FPS from the video.
        # Default to 60 FPS (common for iPhone ReScan captures) when ffprobe
        # is unavailable or the probe fails.
        import subprocess
        import csv as _csv
        video_candidates = list(dataset_dir.glob("rgb.*"))
        native_fps = _DEFAULT_NATIVE_FPS
        video_found = False
        for vc in video_candidates:
            if not vc.is_file():
                continue
            try:
                probe = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "stream=r_frame_rate",
                     "-of", "csv=p=0", str(vc)],
                    capture_output=True, text=True, timeout=10,
                )
                num, den = probe.stdout.strip().split("/")
                native_fps = float(num) / float(den)
                video_found = True
                break
            except Exception:
                job_logger(f"  ⚠ Could not probe video FPS, using default ({_DEFAULT_NATIVE_FPS} FPS)")

        # For image-sequence datasets (no video file), derive native_fps from
        # odometry.csv timestamps — same logic as the GUI's _probe_rescan_dataset.
        n_poses = None
        if not video_found:
            odom_path = dataset_dir / "odometry.csv"
            try:
                with open(odom_path) as _f:
                    _reader = _csv.reader(_f)
                    next(_reader, None)  # skip header
                    first_ts: float | None = None
                    last_ts: float | None = None
                    count = 0
                    for row in _reader:
                        if row:
                            ts = float(row[0].strip())
                            if first_ts is None:
                                first_ts = ts
                            last_ts = ts
                            count += 1
                n_poses = count
                if count > 1 and first_ts is not None and last_ts is not None:
                    duration = last_ts - first_ts
                    if duration > 0:
                        odometry_fps = count / duration
                        if odometry_fps > 0.1:
                            native_fps = odometry_fps
                            job_logger(f"  → Image sequence: native FPS computed from "
                                       f"odometry.csv = {native_fps:.2f}")
            except Exception as exc:
                job_logger(f"  ⚠ Could not compute FPS from odometry.csv ({exc}), "
                           f"using default {_DEFAULT_NATIVE_FPS} FPS")

        computed_subsample = max(1, round(native_fps / fps))

        # Safety check: ensure at least 2 frames will be selected so that SfM
        # can form image pairs.  This guards against extreme subsample values
        # that occur when native_fps falls back to the 60 FPS default but the
        # dataset contains very few odometry poses.
        if n_poses is not None and n_poses > 0:
            # Ceiling division: number of elements in all_frame_indices[::computed_subsample]
            n_selected = (n_poses + computed_subsample - 1) // computed_subsample
            if n_selected < 2:
                new_subsample = max(1, n_poses // 2)
                job_logger(f"  ⚠ subsample={computed_subsample} would select only "
                           f"{n_selected} frame(s) from {n_poses} poses; "
                           f"auto-reducing to {new_subsample}")
                computed_subsample = new_subsample

        try:
            stray_confidence = int(settings.get("stray_confidence", 2))
            stray_depth_subsample = int(settings.get("stray_depth_subsample", 2))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid numeric setting: {exc}") from exc

        stray_result = convert_stray_to_colmap(
            input_dir=dataset_dir,
            output_dir=output_dir,
            mode=stray_mode,
            subsample=computed_subsample,
            confidence_threshold=stray_confidence,
            depth_subsample=stray_depth_subsample,
            skip_pointcloud=not bool(settings.get("stray_gen_pointcloud", True)),
            use_cuda=True,
            image_prefix="",
            logger=job_logger,
            cancel_check=cancel_check,
        )
        job_logger(f"  ✓ {stray_result['n_images']} images, {stray_result['n_points']:,} LiDAR points")

        # ── Optional: input colorspace conversion ──
        if input_colorspace:
            from_cs = SUPPORTED_COLORSPACES[input_colorspace]
            job_logger(f"  Colorspace: converting extracted images from '{from_cs}' → '{_INTERNAL_COLORSPACE}'")
            n = _convert_images_colorspace(images_dir, from_cs, _INTERNAL_COLORSPACE, job_logger)
            job_logger(f"  ✓ {n} image(s) converted from '{from_cs}' to '{_INTERNAL_COLORSPACE}'")

        # ── Step 2: Feature extraction ──
        _update_job(job_id, progress=30, current_step="Features")
        job_logger(f"Step 2/5 — Feature extraction ({feature_type})")

        conf_feature = extract_features.confs[feature_type]
        conf_feature["model"]["max_keypoints"] = max_keypoints
        features_path = extract_features.main(
            conf_feature, images_dir, feature_path=hloc_dir / "features.h5"
        )
        job_logger("  ✓ Features extracted")

        # ── Step 3: Pair generation ──
        _update_job(job_id, progress=50, current_step="Pairs")
        job_logger("Step 3/5 — Pair generation")

        pairs_path = hloc_dir / "pairs.txt"
        if "sequential" in pairing_mode.lower():
            # Inline sequential pair generation (same logic as GUI)
            all_imgs = sorted(
                f.name for f in images_dir.iterdir()
                if f.suffix.lower() in _IMAGE_EXTS
            )
            overlap = 20
            pairs = []
            for i in range(len(all_imgs)):
                for j in range(i + 1, min(i + 1 + overlap, len(all_imgs))):
                    pairs.append((all_imgs[i], all_imgs[j]))
            with open(pairs_path, "w") as f:
                f.writelines(" ".join(p) + "\n" for p in pairs)
        else:
            pairs_from_exhaustive.main(pairs_path, image_list=None, features=features_path)
        job_logger("  ✓ Pairs generated")

        # ── Step 4: Matching ──
        _update_job(job_id, progress=65, current_step="Matching")
        job_logger(f"Step 4/5 — Matching ({matcher_type})")

        conf_match = match_features.confs[matcher_type]
        matches_path = match_features.main(
            conf_match, pairs_path, features=features_path,
            matches=hloc_dir / "matches.h5",
        )
        job_logger("  ✓ Matching complete")

        # ── Step 5: SfM / Triangulation ──
        _update_job(job_id, progress=80, current_step="SfM")

        if stray_result and stray_result["mode"] == "known_poses":
            job_logger("Step 5/5 — Triangulation (ARKit known poses)")
            from hloc.reconstruction import (
                create_empty_db, import_images, get_image_ids,
                import_features as hloc_import_features,
                import_matches as hloc_import_matches,
                estimation_and_geometric_verification,
            )
            db_path = hloc_dir / "database.db"
            create_empty_db(db_path)
            import_images(images_dir, db_path, pycolmap.CameraMode.SINGLE, "PINHOLE")
            image_ids = get_image_ids(db_path)
            hloc_import_features(image_ids, db_path, features_path)
            hloc_import_matches(image_ids, db_path, pairs_path, matches_path)
            estimation_and_geometric_verification(db_path, pairs_path)
            tri_model = pycolmap.triangulate_points(
                reconstruction=pycolmap.Reconstruction(str(sfm_dir)),
                database_path=str(db_path),
                image_path=str(images_dir),
                output_path=str(sfm_dir),
            )
        else:
            job_logger(f"Step 5/5 — SfM reconstruction ({mapper_type})")
            run_sfm_with_live_export(
                sfm_dir=sfm_dir,
                image_dir=images_dir,
                pairs=pairs_path,
                features=features_path,
                matches=matches_path,
                camera_mode=pycolmap.CameraMode.AUTO,
                camera_model=camera_model,
                mapper_type=mapper_type,
                shared_dir=None,
                export_every=5,
                cancel_check=cancel_check,
                logger=job_logger,
                num_threads=num_threads,
            )
        job_logger("  ✓ Reconstruction complete")

        # ── Optional: output colorspace conversion ──
        if output_colorspace:
            to_cs = SUPPORTED_COLORSPACES[output_colorspace]
            job_logger(f"  Colorspace: converting output images from '{_INTERNAL_COLORSPACE}' → '{to_cs}'")
            n = _convert_images_colorspace(images_dir, _INTERNAL_COLORSPACE, to_cs, job_logger)
            job_logger(f"  ✓ {n} image(s) converted from '{_INTERNAL_COLORSPACE}' to '{to_cs}'")

        # ── Optional: EXR remapping (when source was EXR image sequence) ──
        _remap_exr_sources(
            stray_result=stray_result,
            sfm_dir=sfm_dir,
            images_dir=images_dir,
            output_colorspace=output_colorspace,
            logger_fn=job_logger,
        )

        # ── Post-EXR: move images/ to sparse/0/models/0/0/ and update sparse ──
        if stray_result and stray_result.get("has_exr_source", False):
            models_final = sfm_dir / "models" / "0" / "0"
            models_final.mkdir(parents=True, exist_ok=True)
            shutil.move(str(images_dir), str(models_final))
            job_logger("  → images/ moved to sparse/0/models/0/0/")
            # Update sparse model so image paths point to new location
            _prefix_images_in_reconstruction(sfm_dir, "models/0/0/images/", job_logger)
            # Clean up non-essential intermediate files
            for fname in ["database.db"]:
                f = sfm_dir / fname
                if f.exists():
                    f.unlink()
            for log_f in sfm_dir.glob("colmap.LOG*"):
                log_f.unlink()
            job_logger("  → Intermediate files cleaned up from sparse/0/")

        _update_job(job_id, status="completed", progress=100, current_step="Done")
        _append_log(job_id, "Pipeline finished successfully")

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        logger.error("[job %s] FAILED: %s\n%s", job_id, exc, tb)
        _update_job(job_id, status="failed", error=str(exc), current_step="Error")
        _append_log(job_id, f"ERROR: {exc}")

# ---------------------------------------------------------------------------
#  Flask application factory
# ---------------------------------------------------------------------------

def create_app(api_key: str | None = None, output_root: Path | None = None) -> Flask:
    """Create and configure the Flask application.

    Parameters
    ----------
    api_key : str or None
        If *None* a random 32-char key is generated and printed to the console.
    output_root : Path or None
        Override the default temporary directory for jobs output.
    """
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    # API key
    if api_key is None:
        api_key = secrets.token_urlsafe(32)
        logger.info("Generated API key: %s", api_key)
        print(f"\n🔑  API Key (save this): {api_key}\n")

    app.config["API_KEY"] = api_key

    if output_root:
        global JOBS_DIR, UPLOAD_DIR
        JOBS_DIR = output_root / "jobs"
        UPLOAD_DIR = output_root / "uploads"

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Auth helper ──
    def _require_auth():
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            abort(401, description="Missing or malformed Authorization header. Use: Bearer <API_KEY>")
        token = auth[7:]
        if not secrets.compare_digest(token, app.config["API_KEY"]):
            abort(403, description="Invalid API key")

    # ── Routes ──

    @app.route(f"/api/{API_VERSION}/health", methods=["GET"])
    def health():
        """Public health-check endpoint."""
        return jsonify({
            "status": "ok",
            "version": API_VERSION,
            "server": "ReMap",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    @app.route(f"/api/{API_VERSION}/upload", methods=["POST"])
    def upload_dataset():
        """Receive a ReScan dataset as a ZIP archive.

        The ZIP must contain at minimum: ``rgb.mp4`` (or ``rgb.mov``),
        ``odometry.csv``, and ``camera_matrix.csv``.

        Returns the ``dataset_id`` used to reference the upload in
        subsequent calls.
        """
        _require_auth()

        if "file" not in request.files:
            abort(400, description="No 'file' part in the request. Send a ZIP as multipart/form-data with field name 'file'.")

        file = request.files["file"]
        if file.filename == "" or file.filename is None:
            abort(400, description="Empty filename")

        dataset_id = uuid.uuid4().hex[:12]
        dataset_dir = UPLOAD_DIR / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)

        zip_path = dataset_dir / "dataset.zip"
        file.save(str(zip_path))

        # Validate & extract
        if not zipfile.is_zipfile(str(zip_path)):
            shutil.rmtree(dataset_dir, ignore_errors=True)
            abort(400, description="Uploaded file is not a valid ZIP archive")

        with zipfile.ZipFile(str(zip_path), "r") as zf:
            # Security: reject paths with .. or absolute paths
            for member in zf.namelist():
                if member.startswith("/") or ".." in member:
                    shutil.rmtree(dataset_dir, ignore_errors=True)
                    abort(400, description=f"Invalid path in ZIP: {member}")
            zf.extractall(dataset_dir / "data")

        zip_path.unlink()  # Remove ZIP after extraction

        # Resolve: the dataset might be inside a sub-folder
        data_root = dataset_dir / "data"
        candidates = [data_root]
        for child in data_root.iterdir():
            if child.is_dir():
                candidates.append(child)

        actual_dir = None
        for c in candidates:
            has_video = any(c.glob("rgb.mp4")) or any(c.glob("rgb.mov"))
            has_image_sequence = (c / "rgb").is_dir() and any(
                f.is_file() and f.suffix.lower() in _IMAGE_EXTS
                for f in (c / "rgb").iterdir()
            )
            has_odometry = (c / "odometry.csv").exists()
            has_camera = (c / "camera_matrix.csv").exists()
            if (has_video or has_image_sequence) and has_odometry and has_camera:
                actual_dir = c
                break

        if actual_dir is None:
            shutil.rmtree(dataset_dir, ignore_errors=True)
            abort(400, description="ZIP does not contain a valid ReScan dataset. "
                  "Expected: rgb.mp4 (or rgb.mov) or a rgb/ image directory, odometry.csv, camera_matrix.csv")

        return jsonify({
            "dataset_id": dataset_id,
            "files": sorted(str(p.relative_to(actual_dir)) for p in actual_dir.rglob("*") if p.is_file()),
            "message": "Dataset uploaded successfully",
        }), 201

    @app.route(f"/api/{API_VERSION}/process", methods=["POST"])
    def start_processing():
        """Start processing a previously uploaded dataset.

        Expects a JSON body with ``dataset_id`` (from /upload) and
        optional ``settings`` dict.
        """
        _require_auth()

        data = request.get_json(silent=True)
        if data is None:
            abort(400, description="Request body must be JSON")

        dataset_id = data.get("dataset_id")
        if not dataset_id:
            abort(400, description="Missing 'dataset_id'")

        dataset_dir = UPLOAD_DIR / dataset_id / "data"
        if not dataset_dir.exists():
            abort(404, description=f"Dataset '{dataset_id}' not found. Upload it first via /upload.")

        # Resolve actual dataset root (may be in a subfolder)
        actual_dir = None
        candidates = [dataset_dir]
        for child in dataset_dir.iterdir():
            if child.is_dir():
                candidates.append(child)
        for c in candidates:
            has_video = any(c.glob("rgb.mp4")) or any(c.glob("rgb.mov"))
            has_image_sequence = (c / "rgb").is_dir() and any(
                f.is_file() and f.suffix.lower() in _IMAGE_EXTS
                for f in (c / "rgb").iterdir()
            )
            has_odometry = (c / "odometry.csv").exists()
            if (has_video or has_image_sequence) and has_odometry:
                actual_dir = c
                break
        if actual_dir is None:
            abort(400, description="Cannot locate ReScan files in uploaded dataset")

        settings = data.get("settings", {})

        # Validate optional colorspace fields and merge them into settings.
        for cs_field in ("input_colorspace", "output_colorspace"):
            cs_value = data.get(cs_field)
            if cs_value is not None:
                if not isinstance(cs_value, str):
                    abort(400, description=f"'{cs_field}' must be a string")
                cs_key = cs_value.lower()
                if cs_key not in SUPPORTED_COLORSPACES:
                    accepted = ", ".join(sorted(SUPPORTED_COLORSPACES))
                    abort(400, description=(
                        f"Unsupported value '{cs_value}' for '{cs_field}'. "
                        f"Accepted values: {accepted}"
                    ))
                settings[cs_field] = cs_key

        job = _new_job(actual_dir, settings)
        job_id = job["job_id"]

        # Launch in background thread
        t = threading.Thread(target=_run_job, args=(job_id,), daemon=True)
        t.start()

        return jsonify({
            "job_id": job_id,
            "status": job["status"],
            "message": "Processing started",
        }), 202

    @app.route(f"/api/{API_VERSION}/jobs/<job_id>/status", methods=["GET"])
    def job_status(job_id):
        """Get the current status and progress of a processing job."""
        _require_auth()

        job = _get_job(job_id)
        if job is None:
            abort(404, description=f"Job '{job_id}' not found")

        return jsonify({
            "job_id": job["job_id"],
            "status": job["status"],
            "progress": job["progress"],
            "current_step": job["current_step"],
            "created_at": job["created_at"],
            "updated_at": job["updated_at"],
            "error": job["error"],
            "settings": job["settings"],
        })

    @app.route(f"/api/{API_VERSION}/jobs/<job_id>/logs", methods=["GET"])
    def job_logs(job_id):
        """Retrieve the processing log for a job."""
        _require_auth()

        job = _get_job(job_id)
        if job is None:
            abort(404, description=f"Job '{job_id}' not found")

        return jsonify({
            "job_id": job["job_id"],
            "log": job["log"],
        })

    @app.route(f"/api/{API_VERSION}/jobs/<job_id>/result", methods=["GET"])
    def job_result(job_id):
        """Download the result of a completed job as a ZIP archive.

        The ZIP contains the standard COLMAP output structure:
        ``images/``, ``sparse/0/``, and ``hloc_outputs/``.
        """
        _require_auth()

        job = _get_job(job_id)
        if job is None:
            abort(404, description=f"Job '{job_id}' not found")

        if job["status"] != "completed":
            abort(409, description=f"Job is not completed yet (status: {job['status']}). "
                  "Poll /jobs/{job_id}/status until status is 'completed'.")

        output_dir = Path(job["output_dir"])
        if not output_dir.exists():
            abort(500, description="Output directory missing")

        # Create ZIP on-the-fly
        zip_path = JOBS_DIR / job_id / "result.zip"
        if not zip_path.exists():
            with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
                for file_path in output_dir.rglob("*"):
                    if file_path.is_file():
                        arcname = file_path.relative_to(output_dir)
                        zf.write(file_path, arcname)

        return send_file(
            str(zip_path),
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"remap_result_{job_id}.zip",
        )

    @app.route(f"/api/{API_VERSION}/jobs", methods=["GET"])
    def list_jobs():
        """List all jobs (most recent first)."""
        _require_auth()

        with _jobs_lock:
            jobs_summary = [
                {
                    "job_id": j["job_id"],
                    "status": j["status"],
                    "progress": j["progress"],
                    "current_step": j["current_step"],
                    "created_at": j["created_at"],
                    "updated_at": j["updated_at"],
                }
                for j in sorted(_jobs.values(), key=lambda x: x["created_at"], reverse=True)
            ]

        return jsonify({"jobs": jobs_summary})

    @app.route(f"/api/{API_VERSION}/jobs/<job_id>/cancel", methods=["POST"])
    def cancel_job(job_id):
        """Cancel a running job."""
        _require_auth()

        job = _get_job(job_id)
        if job is None:
            abort(404, description=f"Job '{job_id}' not found")

        if job["status"] not in ("queued", "processing"):
            abort(409, description=f"Job cannot be cancelled (status: {job['status']})")

        _update_job(job_id, status="cancelled", current_step="Cancelled")
        _append_log(job_id, "Job cancelled by user")

        return jsonify({"job_id": job_id, "status": "cancelled"})

    # ── Error handlers ──
    @app.errorhandler(400)
    @app.errorhandler(401)
    @app.errorhandler(403)
    @app.errorhandler(404)
    @app.errorhandler(409)
    @app.errorhandler(413)
    @app.errorhandler(500)
    def handle_error(error):
        return jsonify({
            "error": error.name if hasattr(error, "name") else "Error",
            "message": error.description if hasattr(error, "description") else str(error),
        }), error.code if hasattr(error, "code") else 500

    return app

# ---------------------------------------------------------------------------
#  Standalone entry-point
# ---------------------------------------------------------------------------

def start_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 api_key: str | None = None, output_root: Path | None = None,
                 debug: bool = False):
    """Start the ReMap API server (blocking call)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    app = create_app(api_key=api_key, output_root=output_root)
    logger.info("Starting ReMap server on http://%s:%d", host, port)
    app.run(host=host, port=port, debug=debug, threaded=True)


def start_server_background(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                            api_key: str | None = None) -> tuple[threading.Thread, str]:
    """Start the server in a background daemon thread (non-blocking).

    Returns (thread, api_key).
    """
    if api_key is None:
        api_key = secrets.token_urlsafe(32)

    app = create_app(api_key=api_key)
    t = threading.Thread(
        target=lambda: app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False),
        daemon=True,
    )
    t.start()
    return t, api_key


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReMap API Server")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Bind address (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port (default: {DEFAULT_PORT})")
    parser.add_argument("--api-key", default=None, help="API key (auto-generated if omitted)")
    parser.add_argument("--output-dir", default=None, help="Root directory for job outputs")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args()

    output_root = Path(args.output_dir) if args.output_dir else None
    start_server(host=args.host, port=args.port, api_key=args.api_key,
                 output_root=output_root, debug=args.debug)
