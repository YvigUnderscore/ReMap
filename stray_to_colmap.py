#!/usr/bin/env python3
"""
stray_to_colmap.py — Convert a Rescan dataset to a full COLMAP workspace.

Supports two approaches:
  A) "known_poses"  — Import ARKit odometry as ground-truth, then triangulate with COLMAP.
  B) "full_sfm"     — Extract RGB frames only, let COLMAP estimate everything from scratch.

Usage:
    python stray_to_colmap.py --input Rescan_dataset --output ./colmap_out
    python stray_to_colmap.py --input Rescan_dataset --output ./colmap_out --mode known_poses
"""
import argparse
import csv
import math
import struct
import subprocess
import shutil
import sys
import concurrent.futures
from pathlib import Path

import numpy as np
from PIL import Image

# Supported image file extensions for pre-extracted image sequences
_IMAGE_EXTS = {".exr", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


# ─────────────────────────────────────────────────────────────────────────────
#  PARSING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def parse_camera_matrix(csv_path: Path) -> dict:
    """Parse camera_matrix.csv → dict with fx, fy, cx, cy."""
    rows = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append([float(x) for x in row])
    # 3×3 intrinsic matrix
    K = np.array(rows)
    return {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2]}


def parse_odometry(csv_path: Path) -> list[dict]:
    """Parse odometry.csv → list of pose dicts with timestamp, frame, position (xyz), quaternion (qxqyqzqw)."""
    poses = []
    with open(csv_path) as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        # Strip whitespace from headers
        reader.fieldnames = [h.strip() for h in reader.fieldnames]
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            poses.append({
                "timestamp": float(row["timestamp"]),
                "frame": int(row["frame"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": float(row["z"]),
                "qx": float(row["qx"]),
                "qy": float(row["qy"]),
                "qz": float(row["qz"]),
                "qw": float(row["qw"]),
            })
    return poses


# ─────────────────────────────────────────────────────────────────────────────
#  QUATERNION / POSE MATH
# ─────────────────────────────────────────────────────────────────────────────

def quat_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion (w,x,y,z) to 3×3 rotation matrix."""
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),       2*(x*z + y*w)],
        [    2*(x*y + z*w),     1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [    2*(x*z - y*w),         2*(y*z + x*w),   1 - 2*(x*x + y*y)]
    ])


def world_to_camera(position, qw, qx, qy, qz):
    """
    Convert ARKit world-to-device pose (camera-to-world) to COLMAP convention (world-to-camera).
    ARKit odometry: position = camera center in world, quaternion = camera orientation in world.
    COLMAP images.txt: R, t such that X_cam = R * X_world + t
    """
    # Rotation camera-to-world
    R_cw = quat_to_rotation_matrix(qw, qx, qy, qz)
    # World-to-camera
    R_wc = R_cw.T
    t_wc = -R_wc @ np.array(position)
    return R_wc, t_wc


def rotation_matrix_to_quat(R):
    """Convert 3×3 rotation matrix to quaternion (w, x, y, z)."""
    tr = np.trace(R)
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return w, x, y, z


# ─────────────────────────────────────────────────────────────────────────────
#  RGB FRAME EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _remove_hwaccel(cmd: list[str]) -> list[str]:
    """Remove -hwaccel <value> args from an FFmpeg command."""
    clean = []
    skip_next = False
    for c in cmd:
        if c == "-hwaccel":
            skip_next = True
            continue
        if skip_next:
            skip_next = False
            continue
        clean.append(c)
    return clean


def _build_ffmpeg_cmd(video_path: Path, output_dir: Path, image_prefix: str,
                      subsample: int, hwaccel_args: list[str],
                      raw_format: str | None = None) -> list[str]:
    """Build an FFmpeg frame-extraction command.

    Args:
        raw_format: If set (e.g. "h264" or "hevc"), force input format to read
                    the file as a raw bitstream — used for corrupted MP4 files
                    whose moov atom is missing.
    """
    # Use select filter for subsampling: keep every Nth frame
    if subsample > 1:
        vf = f"select=not(mod(n\\,{subsample})),setpts=N/FRAME_RATE/TB"
    else:
        vf = None

    cmd = ["ffmpeg", "-y"]

    # For raw bitstream recovery, force input format before -i
    if raw_format:
        cmd.extend(["-f", raw_format])
    else:
        cmd.extend(hwaccel_args)

    cmd.extend(["-i", str(video_path)])

    if vf:
        cmd.extend(["-vf", vf])
    
    cmd.extend(["-vsync", "0"])

    cmd.extend([
        "-qscale:v", "2",
        str(output_dir / f"{image_prefix}%06d.png")
    ])
    return cmd


def _cleanup_partial(output_dir: Path, image_prefix: str):
    """Remove any partially extracted frames."""
    for f in output_dir.glob(f"{image_prefix}*.png"):
        f.unlink()


def extract_rgb_frames(video_path: Path, output_dir: Path, subsample: int = 1,
                       use_cuda: bool = False, image_prefix: str = "", logger=print):
    """Extract RGB frames from mp4 video using FFmpeg.

    Handles corrupted MP4 files (missing moov atom) by falling back to
    raw H.264/HEVC bitstream extraction.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing frames (only this dataset's prefix)
    existing = sorted(output_dir.glob(f"{image_prefix}*.png"))
    if existing:
        logger(f"  → {len(existing)} frames déjà extraites ({image_prefix or 'sans préfixe'}), skip FFmpeg")
        return len(existing)

    # Detect CUDA
    if use_cuda:
        try:
            probe = subprocess.run(["ffmpeg", "-hwaccels"], capture_output=True, text=True, timeout=5)
            if "cuda" not in probe.stdout.lower():
                use_cuda = False
        except Exception:
            use_cuda = False

    hwaccel_args = ["-hwaccel", "cuda"] if use_cuda else []
    accel = "GPU CUDA" if use_cuda else "CPU"
    logger(f"  → Extraction FFmpeg ({accel}), subsample={subsample}, préfixe='{image_prefix or 'aucun'}'...")

    # ── Attempt 1: normal MP4 container ──────────────────────────────────
    cmd = _build_ffmpeg_cmd(video_path, output_dir, image_prefix, subsample, hwaccel_args)
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

    # Fallback CPU if CUDA failed
    if proc.returncode != 0 and use_cuda:
        logger("  → CUDA échoué, fallback CPU...")
        _cleanup_partial(output_dir, image_prefix)
        cmd = _remove_hwaccel(cmd)
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

    # ── Attempt 2: corrupted MP4 recovery (raw bitstream) ────────────────
    if proc.returncode != 0 and "moov atom not found" in (proc.stderr or ""):
        logger("  ⚠ MP4 corrompu (moov atom manquant) — tentative de récupération du flux brut...")
        _cleanup_partial(output_dir, image_prefix)

        # Try H.264 first, then HEVC (H.265) — Stray Scanner typically uses H.264
        for raw_fmt in ("h264", "hevc"):
            _cleanup_partial(output_dir, image_prefix)
            cmd_raw = _build_ffmpeg_cmd(
                video_path, output_dir, image_prefix, subsample,
                hwaccel_args=[], raw_format=raw_fmt,
            )
            logger(f"  → Essai format brut: {raw_fmt}...")
            proc = subprocess.run(cmd_raw, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            recovered = sorted(output_dir.glob(f"{image_prefix}*.png"))
            if proc.returncode == 0 and recovered:
                logger(f"  ✓ Récupération réussie via {raw_fmt} — {len(recovered)} frames")
                break
            # If this format gave 0 frames despite exit-code 0, try next
            if recovered:
                logger(f"  ✓ Récupération partielle via {raw_fmt} — {len(recovered)} frames")
                break
        else:
            # Both raw formats failed
            raise RuntimeError(
                f"FFmpeg a échoué (MP4 corrompu, récupération impossible):\n"
                f"{proc.stderr[-1000:] if proc.stderr else 'inconnu'}"
            )
    elif proc.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{proc.stderr[-1000:]}")

    frames = sorted(output_dir.glob(f"{image_prefix}*.png"))
    if not frames:
        raise RuntimeError("FFmpeg n'a extrait aucune frame !")
    logger(f"  → {len(frames)} frames extraites")
    return len(frames)


# ─────────────────────────────────────────────────────────────────────────────
#  DEPTH → POINT CLOUD
# ─────────────────────────────────────────────────────────────────────────────

def generate_pointcloud_from_depth(
    dataset_path: Path,
    poses: list[dict],
    intrinsics: dict,
    images_dir: Path,
    selected_frames: list[int],
    frame_to_filename: dict[int, str],
    img_w: int,
    img_h: int,
    confidence_threshold: int = 2,
    depth_subsample: int = 2,
    logger=print
) -> tuple[np.ndarray, np.ndarray]:
    """
    Back-project depth maps into a colored 3D point cloud.

    Returns (points_xyz [N×3], colors_rgb [N×3] uint8).
    """
    depth_dir = dataset_path / "depth"
    conf_dir = dataset_path / "confidence"

    if not depth_dir.exists():
        logger("  ⚠ Pas de dossier depth/ — skip point cloud")
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)

    # Determine depth resolution by reading one image
    sample_depth = Image.open(sorted(depth_dir.glob("*.png"))[0])
    dw, dh = sample_depth.size  # 256×192

    # Use the passed RGB resolution instead of guessing from a sample image
    rw, rh = img_w, img_h
    scale_x = dw / rw
    scale_y = dh / rh
    fx_d = intrinsics["fx"] * scale_x
    fy_d = intrinsics["fy"] * scale_y
    cx_d = intrinsics["cx"] * scale_x
    cy_d = intrinsics["cy"] * scale_y

    logger(f"  → Depth intrinsics: fx={fx_d:.2f}, fy={fy_d:.2f}, cx={cx_d:.2f}, cy={cy_d:.2f}")
    logger(f"  → Depth: {dw}×{dh}, RGB: {rw}×{rh}, scale: {scale_x:.4f}")

    # Build pixel grid (subsampled)
    vs = np.arange(0, dh, depth_subsample)
    us = np.arange(0, dw, depth_subsample)
    uu, vv = np.meshgrid(us, vs)
    uu_flat = uu.flatten()
    vv_flat = vv.flatten()

    all_points = []
    all_colors = []

    # Build pose lookup: frame_index → pose
    pose_by_frame = {p["frame"]: p for p in poses}

    def process_frame(frame_idx):
        depth_file = depth_dir / f"{frame_idx:06d}.png"
        conf_file = conf_dir / f"{frame_idx:06d}.png"

        if not depth_file.exists():
            return None

        # Load depth (mode I = 32-bit integer, values in mm typically)
        depth_img = np.array(Image.open(depth_file))
        depth_values = depth_img[vv_flat, uu_flat].astype(np.float64)

        # Convert to meters (Rescan stores depth in mm)
        depth_m = depth_values / 1000.0

        # Confidence filter
        if conf_file.exists():
            conf_img = np.array(Image.open(conf_file))
            conf_values = conf_img[vv_flat, uu_flat]
            mask = (conf_values >= confidence_threshold) & (depth_m > 0.01) & (depth_m < 10.0)
        else:
            mask = (depth_m > 0.01) & (depth_m < 10.0)

        if mask.sum() == 0:
            return None

        # Back-project to camera coordinates
        u_masked = uu_flat[mask]
        v_masked = vv_flat[mask]
        d_masked = depth_m[mask]

        x_cam = (u_masked - cx_d) * d_masked / fx_d
        y_cam = (v_masked - cy_d) * d_masked / fy_d
        z_cam = d_masked

        pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

        # Transform to world coordinates using pose
        pose = pose_by_frame.get(frame_idx)
        if pose is None:
            return None

        R_cw = quat_to_rotation_matrix(pose["qw"], pose["qx"], pose["qy"], pose["qz"])
        t_world = np.array([pose["x"], pose["y"], pose["z"]])

        # pts_world = R_cw @ pts_cam.T + t_world (camera-to-world)
        pts_world = (R_cw @ pts_cam.T).T + t_world

        # Get colors from RGB frame
        rgb_filename = frame_to_filename.get(frame_idx)
        rgb_file = images_dir / rgb_filename if rgb_filename else None
        if rgb_file is None or not rgb_file.exists():
            # Fallback if RGB frame is missing
            colors = np.full((pts_world.shape[0], 3), 128, dtype=np.uint8)
        else:
            try:
                rgb_img = np.array(Image.open(rgb_file))
            except Exception:
                # Fallback for formats PIL cannot read (e.g. EXR)
                import OpenImageIO as oiio
                buf = oiio.ImageBuf(str(rgb_file))
                if buf.has_error():
                    raise RuntimeError(f"OpenImageIO could not read {rgb_file}: {buf.geterror()}")
                pixels = buf.get_pixels(oiio.FLOAT)
                pixels = np.clip(pixels, 0.0, 1.0)
                rgb_img = (pixels[..., :3] * 255).astype(np.uint8)
            # Map depth pixel coords to RGB pixel coords
            u_rgb = (u_masked / scale_x).astype(int).clip(0, rw - 1)
            v_rgb = (v_masked / scale_y).astype(int).clip(0, rh - 1)
            colors = rgb_img[v_rgb, u_rgb]

        return pts_world, colors

    n_processed = 0
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(process_frame, selected_frames)

    for result in results:
        if result is not None:
            pts, cols = result
            all_points.append(pts)
            all_colors.append(cols)
            n_processed += 1

    if not all_points:
        logger("  ⚠ Aucun point généré")
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)

    points = np.concatenate(all_points)
    colors = np.concatenate(all_colors)
    logger(f"  → {len(points):,} points 3D générés depuis {n_processed} depth maps")
    return points, colors


# ─────────────────────────────────────────────────────────────────────────────
#  COLMAP TEXT EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def write_cameras_txt(path: Path, intrinsics: dict, width: int, height: int):
    """Write COLMAP cameras.txt with a single PINHOLE camera."""
    with open(path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: 1\n")
        f.write(f"1 PINHOLE {width} {height} "
                f"{intrinsics['fx']:.6f} {intrinsics['fy']:.6f} "
                f"{intrinsics['cx']:.6f} {intrinsics['cy']:.6f}\n")


def write_images_txt(path: Path, poses: list[dict], selected_frames: list[int],
                     frame_to_filename: dict[int, str]):
    """Write COLMAP images.txt with camera poses (world-to-camera convention)."""
    with open(path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(selected_frames)}\n")

        pose_by_frame = {p["frame"]: p for p in poses}
        image_id = 1

        for frame_idx in selected_frames:
            pose = pose_by_frame.get(frame_idx)
            if pose is None:
                continue

            filename = frame_to_filename.get(frame_idx)
            if filename is None:
                continue

            # Convert to COLMAP world-to-camera
            position = [pose["x"], pose["y"], pose["z"]]
            R_wc, t_wc = world_to_camera(position, pose["qw"], pose["qx"], pose["qy"], pose["qz"])
            qw, qx, qy, qz = rotation_matrix_to_quat(R_wc)

            f.write(f"{image_id} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} "
                    f"{t_wc[0]:.8f} {t_wc[1]:.8f} {t_wc[2]:.8f} 1 {filename}\n")
            f.write("\n")  # Empty POINTS2D line
            image_id += 1


def write_points3d_txt(path: Path, points: np.ndarray, colors: np.ndarray):
    """Write COLMAP points3D.txt. Each point has a dummy track."""
    with open(path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points)}\n")
        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = colors[i]
            # Dummy error=1.0, empty track (COLMAP will re-triangulate)
            f.write(f"{i + 1} {x:.8f} {y:.8f} {z:.8f} {int(r)} {int(g)} {int(b)} 1.0\n")


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray):
    """Write a simple PLY point cloud."""
    n = len(points)
    with open(path, "wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        f.write(header.encode())
        for i in range(n):
            f.write(struct.pack("<fff", *points[i]))
            f.write(struct.pack("<BBB", *colors[i]))


# ─────────────────────────────────────────────────────────────────────────────
#  COLMAP BINARY EXPORT (for import into COLMAP database)
# ─────────────────────────────────────────────────────────────────────────────

def write_cameras_bin(path: Path, intrinsics: dict, width: int, height: int):
    """Write COLMAP cameras.bin (single PINHOLE camera)."""
    with open(path, "wb") as f:
        # Number of cameras
        f.write(struct.pack("<Q", 1))
        # Camera: id=1, model=PINHOLE(1), width, height, params (fx, fy, cx, cy)
        f.write(struct.pack("<i", 1))   # camera_id
        f.write(struct.pack("<i", 1))   # model_id (PINHOLE)
        f.write(struct.pack("<Q", width))
        f.write(struct.pack("<Q", height))
        f.write(struct.pack("<4d", intrinsics["fx"], intrinsics["fy"],
                            intrinsics["cx"], intrinsics["cy"]))


def write_images_bin(path: Path, poses: list[dict], selected_frames: list[int],
                     frame_to_filename: dict[int, str]):
    """Write COLMAP images.bin."""
    pose_by_frame = {p["frame"]: p for p in poses}

    # Collect valid entries
    entries = []
    image_id = 1
    for frame_idx in selected_frames:
        pose = pose_by_frame.get(frame_idx)
        if pose is None:
            continue
        filename = frame_to_filename.get(frame_idx)
        if filename is None:
            continue
        position = [pose["x"], pose["y"], pose["z"]]
        R_wc, t_wc = world_to_camera(position, pose["qw"], pose["qx"], pose["qy"], pose["qz"])
        qw, qx, qy, qz = rotation_matrix_to_quat(R_wc)
        entries.append((image_id, qw, qx, qy, qz, t_wc, 1, filename))
        image_id += 1

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(entries)))
        for image_id, qw, qx, qy, qz, t, cam_id, name in entries:
            f.write(struct.pack("<i", image_id))
            f.write(struct.pack("<4d", qw, qx, qy, qz))
            f.write(struct.pack("<3d", *t))
            f.write(struct.pack("<i", cam_id))
            # Name as null-terminated string
            f.write(name.encode("utf-8") + b"\x00")
            # Number of 2D points (0 — COLMAP will triangulate)
            f.write(struct.pack("<Q", 0))


def write_points3d_bin(path: Path, points: np.ndarray, colors: np.ndarray):
    """Write COLMAP points3D.bin."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(points)))
        for i in range(len(points)):
            f.write(struct.pack("<Q", i + 1))  # point3d_id
            f.write(struct.pack("<3d", *points[i]))  # xyz
            f.write(struct.pack("<3B", *colors[i]))  # rgb
            f.write(struct.pack("<d", 1.0))  # error
            # Track length = 0
            f.write(struct.pack("<Q", 0))


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def convert_stray_to_colmap(
    input_dir: Path,
    output_dir: Path,
    mode: str = "full_sfm",
    subsample: int = 3,
    confidence_threshold: int = 2,
    depth_subsample: int = 2,
    skip_pointcloud: bool = False,
    use_cuda: bool = True,
    image_prefix: str = "",
    logger=print,
    cancel_check=None,
):
    """
    Main conversion pipeline.

    Args:
        input_dir:    Path to Rescan dataset folder
        output_dir:   Path to output COLMAP workspace
        mode:         "full_sfm" (approach B, default) or "known_poses" (approach A)
        subsample:    Keep every Nth frame (1 = all, 3 = ~20fps→~7fps)
        confidence_threshold: Min confidence for depth points (0-2, 2=highest)
        depth_subsample: Spatial subsampling of depth maps
        skip_pointcloud: Skip depth → point cloud generation
        use_cuda:     Try CUDA for FFmpeg
        image_prefix: Prefix for extracted image filenames (e.g. "ds01_") to avoid collisions
        logger:       Logging callback
        cancel_check: Cancellation callback (raises on cancel)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Validate dataset structure
    required_csv = ["camera_matrix.csv", "odometry.csv"]
    for f in required_csv:
        if not (input_dir / f).exists():
            raise FileNotFoundError(f"Fichier manquant : {input_dir / f}")

    # Find RGB video: accept both .mp4 (H.264) and .mov (Apple ProRes Log)
    rgb_video = None
    for vname in ["rgb.mp4", "rgb.mov"]:
        if (input_dir / vname).exists():
            rgb_video = input_dir / vname
            break

    # Check for pre-extracted image sequence in rgb/ directory
    rgb_dir = input_dir / "rgb"
    has_image_sequence = rgb_dir.is_dir() and any(
        f.is_file() and f.suffix.lower() in _IMAGE_EXTS for f in rgb_dir.iterdir()
    )

    if rgb_video is None and not has_image_sequence:
        raise FileNotFoundError(
            f"Fichier vidéo manquant : rgb.mp4 ou rgb.mov dans {input_dir}, "
            f"et aucune séquence d'images dans {rgb_dir}"
        )

    logger("═" * 60)
    logger(f"  RESCAN → COLMAP  (mode: {mode})")
    logger("═" * 60)

    # ── 1. Parse intrinsics ──────────────────────────────────────────────
    logger("\n[1/5] Parsing des intrinsèques caméra...")
    intrinsics = parse_camera_matrix(input_dir / "camera_matrix.csv")
    logger(f"  → fx={intrinsics['fx']:.2f}, fy={intrinsics['fy']:.2f}, "
           f"cx={intrinsics['cx']:.2f}, cy={intrinsics['cy']:.2f}")

    if cancel_check:
        cancel_check()

    # ── 2. Parse odometry ────────────────────────────────────────────────
    logger("\n[2/5] Parsing de l'odométrie ARKit...")
    all_poses = parse_odometry(input_dir / "odometry.csv")
    logger(f"  → {len(all_poses)} poses chargées")

    # Select frames with subsampling
    all_frame_indices = [p["frame"] for p in all_poses]
    selected_frames = all_frame_indices[::subsample]
    logger(f"  → {len(selected_frames)} frames sélectionnées (subsample={subsample})")

    if cancel_check:
        cancel_check()

    # ── 3. Extract RGB frames ────────────────────────────────────────────
    logger("\n[3/5] Extraction des frames RGB...")
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    frame_to_filename = {}

    if rgb_video is not None:
        n_frames = extract_rgb_frames(
            rgb_video, images_dir,
            subsample=subsample, use_cuda=use_cuda,
            image_prefix=image_prefix, logger=logger
        )

        if cancel_check:
            cancel_check()

        # Build frame → filename mapping (only this dataset's files via prefix glob)
        extracted_files = sorted(images_dir.glob(f"{image_prefix}*.png"))
        # FFmpeg numbers from 000001.png, our frames are 0-indexed
        # With select filter: output frame i corresponds to selected_frames[i]
        for i, frame_idx in enumerate(selected_frames):
            if i < len(extracted_files):
                frame_to_filename[frame_idx] = extracted_files[i].name

        # Delete any extracted file that wasn't mapped (e.g. video recorded longer than odometry)
        mapped_files = set(frame_to_filename.values())
        deleted_count = 0
        for f in extracted_files:
            if f.name not in mapped_files:
                f.unlink()
                deleted_count += 1

        if deleted_count > 0:
            logger(f"  → {deleted_count} frames excédentaires supprimées (vidéo plus longue que l'odométrie).")
    else:
        # Use pre-extracted image sequence from rgb/ directory
        logger(f"  → Séquence d'images dans rgb/ détectée — copie vers images/ (subsample={subsample})")
        src_files = sorted(
            f for f in rgb_dir.iterdir()
            if f.is_file() and f.suffix.lower() in _IMAGE_EXTS
        )
        # Map frame index (parsed from stem) to source file
        src_by_frame = {}
        for f in src_files:
            try:
                src_by_frame[int(f.stem)] = f
            except ValueError:
                pass

        copied = 0
        for frame_idx in selected_frames:
            if frame_idx in src_by_frame:
                src = src_by_frame[frame_idx]
                dst_name = f"{image_prefix}{frame_idx:06d}{src.suffix}"
                dst = images_dir / dst_name
                if not dst.exists():
                    shutil.copy2(src, dst)
                frame_to_filename[frame_idx] = dst_name
                copied += 1

        n_frames = copied
        logger(f"  → {n_frames} frames copiées depuis rgb/")

        if cancel_check:
            cancel_check()

    # Detect image resolution
    if frame_to_filename:
        first_frame_name = next(iter(frame_to_filename.values()))
        first_frame_path = images_dir / first_frame_name
        try:
            sample = Image.open(first_frame_path)
            img_w, img_h = sample.size
        except Exception:
            # Fallback for formats PIL cannot read (e.g. EXR on older Pillow)
            import OpenImageIO as oiio
            buf = oiio.ImageBuf(str(first_frame_path))
            spec = buf.spec()
            img_w, img_h = spec.width, spec.height
    else:
        raise RuntimeError("Aucune frame extraite !")

    logger(f"  → Résolution: {img_w}×{img_h}")

    # ── 4. Generate point cloud from depth ───────────────────────────────
    points = np.zeros((0, 3))
    colors = np.zeros((0, 3), dtype=np.uint8)

    if not skip_pointcloud and (input_dir / "depth").exists():
        logger("\n[4/5] Génération du nuage de points LiDAR...")
        points, colors = generate_pointcloud_from_depth(
            dataset_path=input_dir,
            poses=all_poses,
            intrinsics=intrinsics,
            images_dir=images_dir,
            selected_frames=selected_frames,
            frame_to_filename=frame_to_filename,
            img_w=img_w,
            img_h=img_h,
            confidence_threshold=confidence_threshold,
            depth_subsample=depth_subsample,
            logger=logger,
        )
    else:
        logger("\n[4/5] Point cloud LiDAR — ignoré")

    if cancel_check:
        cancel_check()

    # ── 5. Export COLMAP workspace ────────────────────────────────────────
    logger(f"\n[5/5] Export COLMAP (mode={mode})...")
    sparse_dir = output_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    if mode == "known_poses":
        # ── Approach A: Export poses + point cloud ──
        # Text format (human-readable, required for point_triangulator)
        write_cameras_txt(sparse_dir / "cameras.txt", intrinsics, img_w, img_h)
        write_images_txt(sparse_dir / "images.txt", all_poses, selected_frames, frame_to_filename)

        if len(points) > 0:
            write_points3d_txt(sparse_dir / "points3D.txt", points, colors)
            logger(f"  → points3D.txt: {len(points):,} points")
        else:
            # Write empty points3D.txt — COLMAP will triangulate
            with open(sparse_dir / "points3D.txt", "w") as f:
                f.write("# 3D point list\n# Number of points: 0\n")

        # Also write binary format for compatibility
        write_cameras_bin(sparse_dir / "cameras.bin", intrinsics, img_w, img_h)
        write_images_bin(sparse_dir / "images.bin", all_poses, selected_frames, frame_to_filename)
        if len(points) > 0:
            write_points3d_bin(sparse_dir / "points3D.bin", points, colors)
        else:
            with open(sparse_dir / "points3D.bin", "wb") as f:
                f.write(struct.pack("<Q", 0))

        logger(f"  → cameras.txt: PINHOLE {img_w}×{img_h}")
        logger(f"  → images.txt: {len(frame_to_filename)} images avec poses")
        logger("  → Mode known_poses : prêt pour point_triangulator ou Gaussian Splatting direct")

    else:
        # ── Approach B: Full SfM — just images, no poses ──
        # Export only the images directory — COLMAP will do everything
        # But still export the point cloud + intrinsics info for reference
        logger("  → Mode full_sfm : images prêtes pour le pipeline COLMAP/HLoc complet")

        # Save intrinsics as reference (not imported into COLMAP automatically)
        ref_dir = output_dir / "reference"
        ref_dir.mkdir(exist_ok=True)

        # Copy original data as reference
        shutil.copy2(input_dir / "camera_matrix.csv", ref_dir / "camera_matrix.csv")
        shutil.copy2(input_dir / "odometry.csv", ref_dir / "odometry.csv")

        if len(points) > 0:
            # Save point cloud as reference/bonus
            ply_ref = ref_dir / "lidar_pointcloud.ply"
            write_ply(ply_ref, points, colors)
            logger(f"  → Nuage LiDAR de référence : {ply_ref.name} ({len(points):,} points)")

        logger("  → Intrinsèques et odométrie sauvés dans reference/")
        # Note: for full_sfm, the caller (GUI) will run HLoc pipeline normally

    # ── Export PLY point cloud ────────────────────────────────────────────
    if len(points) > 0:
        ply_path = output_dir / "pointcloud_lidar.ply"
        write_ply(ply_path, points, colors)
        logger(f"\n  ☁ Point cloud PLY: {ply_path} ({len(points):,} points)")

    # ── Summary ──────────────────────────────────────────────────────────
    logger("\n" + "═" * 60)
    logger("  ✓ CONVERSION TERMINÉE")
    logger(f"  → Output: {output_dir}")
    logger(f"  → Images: {len(frame_to_filename)}")
    logger(f"  → Points 3D: {len(points):,}")
    logger(f"  → Mode: {mode}")
    logger("═" * 60)

    return {
        "images_dir": images_dir,
        "sparse_dir": sparse_dir,
        "n_images": len(frame_to_filename),
        "n_points": len(points),
        "intrinsics": intrinsics,
        "img_width": img_w,
        "img_height": img_h,
        "mode": mode,
        "frame_to_filename": frame_to_filename,
        "selected_frames": selected_frames,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert Rescan dataset to COLMAP workspace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  full_sfm     (B, default) Extract RGB only, let COLMAP/HLoc estimate poses from scratch.
                             More robust, better for Gaussian Splatting quality.
  known_poses  (A)          Import ARKit odometry as ground-truth poses.
                             Faster, preserves metric scale, requires COLMAP point_triangulator.

Examples:
  %(prog)s --input Stray_Scanner_dataset --output ./colmap_out
  %(prog)s --input Stray_Scanner_dataset --output ./colmap_out --mode known_poses
  %(prog)s --input Stray_Scanner_dataset --output ./colmap_out --subsample 5 --no-pointcloud
        """)

    parser.add_argument("--input", "-i", required=True, help="Path to Rescan dataset folder")
    parser.add_argument("--output", "-o", required=True, help="Output COLMAP workspace folder")
    parser.add_argument("--mode", "-m", choices=["full_sfm", "known_poses"], default="full_sfm",
                        help="Conversion mode (default: full_sfm)")
    parser.add_argument("--subsample", "-s", type=int, default=3,
                        help="Keep every Nth frame (default: 3)")
    parser.add_argument("--confidence-threshold", "-c", type=int, default=2, choices=[0, 1, 2],
                        help="Min depth confidence level 0-2 (default: 2 = highest)")
    parser.add_argument("--depth-subsample", type=int, default=2,
                        help="Spatial subsampling of depth maps (default: 2)")
    parser.add_argument("--no-pointcloud", action="store_true",
                        help="Skip LiDAR depth → point cloud generation")
    parser.add_argument("--no-cuda", action="store_true",
                        help="Disable CUDA for FFmpeg")

    args = parser.parse_args()

    convert_stray_to_colmap(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        mode=args.mode,
        subsample=args.subsample,
        confidence_threshold=args.confidence_threshold,
        depth_subsample=args.depth_subsample,
        skip_pointcloud=args.no_pointcloud,
        use_cuda=not args.no_cuda,
    )


if __name__ == "__main__":
    main()
