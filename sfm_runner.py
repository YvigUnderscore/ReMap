"""
Custom SfM reconstruction runner with live 3D export support.
Replaces hloc's run_reconstruction() to hook into pycolmap's callbacks.
"""
import json
import shutil
import tempfile
import multiprocessing
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pycolmap

from hloc.reconstruction import (
    create_empty_db,
    import_images,
    get_image_ids,
    import_features,
    import_matches,
    estimation_and_geometric_verification,
)


def _export_reconstruction_state(reconstruction, shared_dir):
    """Export current reconstruction state (PLY + camera JSON) to shared directory."""
    shared_dir = Path(shared_dir)
    shared_dir.mkdir(parents=True, exist_ok=True)

    ply_path = shared_dir / "model.ply"
    cam_path = shared_dir / "cameras.json"

    try:
        # Export point cloud as PLY (atomic write via temp file)
        tmp_ply = shared_dir / "model_tmp.ply"
        reconstruction.export_PLY(str(tmp_ply))
        tmp_ply.replace(ply_path)

        # Export camera poses as JSON
        cam_data = []
        for image_id in reconstruction.reg_image_ids():
            image = reconstruction.image(image_id)
            # cam_from_world → invert to get world_from_cam
            cam_from_world = image.cam_from_world
            R = cam_from_world.rotation.matrix()
            t = cam_from_world.translation
            # world_from_cam = inverse of cam_from_world
            R_inv = R.T
            t_inv = -R_inv @ t
            world_from_cam = np.eye(4)
            world_from_cam[:3, :3] = R_inv
            world_from_cam[:3, 3] = t_inv
            cam_data.append({
                "image_id": int(image_id),
                "name": image.name,
                "world_from_cam": world_from_cam.tolist(),
            })

        tmp_cam = shared_dir / "cameras_tmp.json"
        with open(tmp_cam, 'w') as f:
            json.dump(cam_data, f)
        tmp_cam.replace(cam_path)
    except Exception:
        pass  # Don't crash the pipeline for a visualization issue


def run_sfm_with_live_export(
    sfm_dir,
    image_dir,
    pairs,
    features,
    matches,
    camera_mode=pycolmap.CameraMode.SINGLE,
    camera_model="OPENCV",
    mapper_type="colmap",
    shared_dir=None,
    export_every=5,
    cancel_check=None,
    image_options=None,
    logger=print,
    num_threads=None,
):
    """
    Full SfM pipeline with support for COLMAP (incremental) and GLOMAP (global).
    """
    sfm_dir = Path(sfm_dir)
    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / "database.db"
    
    # Setup logging
    pycolmap.logging.set_log_destination(pycolmap.logging.INFO, sfm_dir / "colmap.LOG.")

    # Import data into COLMAP database
    create_empty_db(database)
    opts = dict(image_options or {})
    opts["camera_model"] = camera_model
    import_images(image_dir, database, camera_mode, image_list=None, options=opts)
    image_ids = get_image_ids(database)

    with pycolmap.Database.open(database) as db:
        import_features(image_ids, db, features)
        import_matches(image_ids, db, pairs, matches, min_match_score=None, skip_geometric_verification=False)

    estimation_and_geometric_verification(database, pairs, verbose=False)

    # --- Mapping ---
    models_path = sfm_dir / "models"
    models_path.mkdir(exist_ok=True, parents=True)
    
    # Check for GLOMAP availability
    use_glomap = (mapper_type.lower() == "glomap")
    glomap_bin = shutil.which("glomap")
    
    if use_glomap and not glomap_bin:
        logger("⚠️ GLOMAP non trouvé (`glomap` introuvable dans le PATH). Bascule sur COLMAP.")
        use_glomap = False

    if use_glomap:
        # --- GLOMAP Execution ---
        # GLOMAP output structure: output_dir/{cameras.bin, images.bin, points3D.bin}
        # We'll output directly to models/0 to mimic COLMAP structure
        glomap_output = models_path / "0"
        glomap_output.mkdir(parents=True, exist_ok=True)

        cmd = [
            glomap_bin, "mapper",
            "--database_path", str(database),
            "--image_path", str(image_dir),
            "--output_path", str(glomap_output)
        ]
        
        if num_threads:
             # Tentative support for num_threads in GLOMAP if it follows COLMAP CLI standards
             # cmd.extend(["--num_threads", str(num_threads)]) 
             # For now, just log that we are using system default or we rely on OMP_NUM_THREADS
             pass

        env = os.environ.copy()
        if num_threads:
            env["OMP_NUM_THREADS"] = str(num_threads)
            
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        
        while proc.poll() is None:
            if cancel_check:
                try:
                    cancel_check()
                except Exception:
                    proc.terminate()
                    raise
            
            # Read line by line if possible or just wait
            # Here we do non-blocking read
            try:
                line = proc.stdout.readline()
                if line:
                    logger(f"[GLOMAP] {line.strip()}")
            except Exception:
                pass

            time.sleep(0.1)
            
        # Read remaining output
        for line in proc.stdout:
            logger(f"[GLOMAP] {line.strip()}")

        if proc.returncode != 0:
            raise RuntimeError(f"GLOMAP a échoué avec le code {proc.returncode}")

        # Post-process: GLOMAP produces standard COLMAP binaries.
        # We need to load it to confirm and export to viewer.
        # FIX: GLOMAP might create a nested '0' directory (e.g. models/0/0)
        # Check if output is directly in glomap_output or in a subdir
        actual_output = glomap_output
        if (glomap_output / "0").exists() and (glomap_output / "0" / "cameras.bin").exists():
            actual_output = glomap_output / "0"
            logger(f"[GLOMAP] Output detected in nested directory: {actual_output}")

        rec = pycolmap.Reconstruction()
        rec.read(str(actual_output))
        
        if shared_dir:
            _export_reconstruction_state(rec, shared_dir)
            
        # Copy to main sfm_dir
        for filename in ["images.bin", "cameras.bin", "points3D.bin"]:
            src = actual_output / filename
            dst = sfm_dir / filename
            if src.exists():
                shutil.copy(str(src), str(dst))

        return rec

    else:
        # --- COLMAP (Incremental) ---
        counter = [0]
        
        def on_initial_pair():
            counter[0] = 2
            if shared_dir:
                _try_export_latest_model(models_path, shared_dir)

        def on_next_image():
            counter[0] += 1
            if cancel_check:
                cancel_check()
            if shared_dir and counter[0] % export_every == 0:
                _try_export_latest_model(models_path, shared_dir)

        if num_threads is None:
            num_threads = min(multiprocessing.cpu_count(), 16)
        
        # Explicitly set threads for both Pipeline and Mapper to be safe
        options = {
            "num_threads": num_threads,
            "mapper": {
                "num_threads": num_threads
            }
        }

        reconstructions = pycolmap.incremental_mapping(
            database_path=str(database),
            image_path=str(image_dir),
            output_path=str(models_path),
            options=options,
            initial_image_pair_callback=on_initial_pair,
            next_image_callback=on_next_image,
        )

        if len(reconstructions) == 0:
            raise RuntimeError("Aucun modèle reconstruit !")

        # Find largest reconstruction
        largest_index = max(reconstructions, key=lambda i: reconstructions[i].num_reg_images())
        rec = reconstructions[largest_index]

        # Final export to viewer
        if shared_dir:
            _export_reconstruction_state(rec, shared_dir)

        # Move result files to sfm_dir
        for filename in ["images.bin", "cameras.bin", "points3D.bin", "frames.bin", "rigs.bin"]:
            src = models_path / str(largest_index) / filename
            dst = sfm_dir / filename
            if dst.exists():
                dst.unlink()
            if src.exists():
                shutil.move(str(src), str(dst))

        return rec


def _try_export_latest_model(models_path, shared_dir):
    """Try to read the latest model from COLMAP's output and export it."""
    try:
        # COLMAP writes models to numbered subdirectories
        model_dirs = sorted(
            [d for d in models_path.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True
        )
        for model_dir in model_dirs:
            bins = list(model_dir.glob("*.bin"))
            if len(bins) >= 3:  # images.bin, cameras.bin, points3D.bin
                rec = pycolmap.Reconstruction()
                rec.read(str(model_dir))
                if rec.num_reg_images() > 0:
                    _export_reconstruction_state(rec, shared_dir)
                    return
    except Exception:
        pass  # Silently skip if model not yet ready
