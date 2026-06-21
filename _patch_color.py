#!/usr/bin/env python3
"""Patch ReMap-GUI.py: replace the color conversion step 1.5 and add post-SfM EXR generation."""
import sys, os
os.environ["PYTHONIOENCODING"] = "utf-8"

gui_path = "ReMap-GUI.py"
content = open(gui_path, "rb").read().decode("utf-8")

# Normalize all line endings to \n for matching, we'll restore \r\n at the end
content = content.replace("\r\n", "\n")

# 1. Replace the color conversion step 1.5
# Use a simpler unique anchor to find and replace the block
marker_start = "            # --- 1.5 COLOR CONVERSION ---\n"
marker_end = "            if not is_stray_mode:\n"

idx_start = content.find(marker_start)
idx_end = content.find(marker_end, idx_start)

if idx_start < 0 or idx_end < 0:
    print(f"[FAIL] Could not find color conversion block markers")
    print(f"  marker_start found: {idx_start >= 0}")
    print(f"  marker_end found: {idx_end >= 0}")
    sys.exit(1)

# Extract the old block
old_block = content[idx_start:idx_end]
print(f"[INFO] Found color conversion block ({len(old_block)} chars)")

NEW_COLOR_STEP = """\
            # --- 1.5 COLOR CONVERSION (SfM proxy) ---
            if self.color_enabled.get() and is_video_mode:
                # Video mode with color conversion: convert extracted frames
                # to tone-mapped sRGB PNGs so SfM operates on display-referred
                # images.  The actual output conversion (EXR) happens after SfM.
                color_step_idx = 1
                GUIProgressTqdm._step_index = color_step_idx
                GUIProgressTqdm._step_name = "Color \u2192 sRGB proxy"
                _source = self.color_source.get()
                if _source == "Auto-detect":
                    _source = self.detected_color_profile.get()
                    if not _source:
                        _source = "Linear BT.2020"
                        self._log_tagged("[CPU]", f"       \u26a0 No color profile detected, assuming {_source}")
                self._log_tagged("[CPU]", f"[{color_step_idx+1}/{len(self.STEPS)}] Converting frames to sRGB proxy for SfM ({_source} \u2192 sRGB)...")
                exts_proxy = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
                proxy_images = sorted([
                    p for p in images_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in exts_proxy
                ])
                if proxy_images:
                    try:
                        configured_threads = int(self.num_workers.get())
                    except Exception:
                        configured_threads = 1
                    threads = max(1, configured_threads)
                    self._log_tagged("[CPU]", f"       \u2192 Converting {len(proxy_images)} frames with {threads} process(es)...")
                    from concurrent.futures import ProcessPoolExecutor, as_completed
                    from concurrent.futures.process import BrokenProcessPool
                    _proxy_ok = 0
                    _proxy_errs = []
                    try:
                        with ProcessPoolExecutor(max_workers=threads) as executor:
                            _futs = [
                                executor.submit(convert_frame_to_srgb_proxy_worker, str(p), _source)
                                for p in proxy_images
                            ]
                            for fut in as_completed(_futs):
                                if self._cancelled:
                                    executor.shutdown(wait=False, cancel_futures=True)
                                    break
                                try:
                                    ok, err = fut.result()
                                except (BrokenProcessPool, Exception) as e:
                                    ok, err = False, str(e)
                                if ok:
                                    _proxy_ok += 1
                                elif err:
                                    _proxy_errs.append(err)
                    except BrokenProcessPool as e:
                        _proxy_errs.append(f"Worker pool crashed: {e}")
                    if _proxy_errs:
                        self._log_tagged("[CPU]", f"       \u274c {len(_proxy_errs)} proxy conversion error(s): {list(set(_proxy_errs))[:3]}")
                    self._log_tagged("[CPU]", f"       \u2713 {_proxy_ok}/{len(proxy_images)} frames converted to sRGB proxy")
            elif self.color_enabled.get() and not is_video_mode and not is_stray_mode:
                # Image mode: apply color conversion in-place (legacy behavior)
                color_step_idx = 1
                GUIProgressTqdm._step_index = color_step_idx
                GUIProgressTqdm._step_name = "Color Conversion"
                global_models_dir = sfm_dir / "models" / "0" / "0"
                global_models_dir.mkdir(parents=True, exist_ok=True)
                apply_color_conversion(images_dir, "[CPU]", f"[{color_step_idx+1}/{len(self.STEPS)}] Applying Color Conversion ({self.color_source.get()} \u2192 {self.color_dest.get()})...", exr_out_dir=global_models_dir)

"""

content = content[:idx_start] + NEW_COLOR_STEP + content[idx_end:]
print("[OK] Replaced color conversion step 1.5")


# 2. Add post-SfM EXR generation for video mode
# Find the anchor: after "Reconstruction complete" and around normalize_final_bundle
anchor = '                self._log_tagged(sfm_tag, "       \u2713 Reconstruction complete")\n'
anchor_idx = content.find(anchor)
if anchor_idx < 0:
    print("[FAIL] Could not find post-SfM anchor")
    sys.exit(1)

# Find the end of the section to replace (up to and including SUCCESS line)
success_marker = '                self._log_tagged("[OK]", f"\\n\u2713 SUCCESS'
success_idx = content.find(success_marker, anchor_idx)
if success_idx < 0:
    print("[FAIL] Could not find SUCCESS marker")
    sys.exit(1)

# Find the end of the SUCCESS line
success_end = content.find("\n", success_idx)
if success_end < 0:
    success_end = len(content)
else:
    success_end += 1  # include the newline

old_post_sfm = content[anchor_idx:success_end]
print(f"[INFO] Found post-SfM block ({len(old_post_sfm)} chars)")

NEW_POST_SFM = '''                self._log_tagged(sfm_tag, "       \u2713 Reconstruction complete")

                # \u2500\u2500 Post-SfM: EXR output conversion (video mode + color enabled) \u2500\u2500
                if self.color_enabled.get() and is_video_mode:
                    self._check_cancelled()
                    _dest = self.color_dest.get()
                    _source = self.color_source.get()
                    if _source == "Auto-detect":
                        _source = self.detected_color_profile.get() or "Linear BT.2020"
                    
                    _cs_in = self.ocio_in_cs.get() if _dest == "Custom OCIO..." else None
                    _cs_out = self.ocio_out_cs.get() if _dest == "Custom OCIO..." else None
                    _cfg_path = self.ocio_path.get() if _dest == "Custom OCIO..." else None

                    self._log_tagged("[CPU]", f"\\n       \u2500\u2500 Post-SfM: Re-extracting frames & converting to output EXR ({_source} \u2192 {_dest}) \u2500\u2500")

                    # Create final output directory for EXR files
                    ds_final = base_out / f"{base_out.name}_SfM_Dataset_Output"
                    ds_final_images = ds_final / "images"
                    ds_final_images.mkdir(parents=True, exist_ok=True)

                    # Re-extract the same sub-sampled frames from video as 16-bit PNGs
                    # into a temporary directory, then convert each to the output EXR.
                    with tempfile.TemporaryDirectory() as _raw_tmpdir:
                        _raw_tmp = Path(_raw_tmpdir)
                        videos = self.video_paths
                        for vid_idx, video in enumerate(videos, start=1):
                            self._check_cancelled()
                            prefix = f"vid{vid_idx:02d}"
                            vid_info = next((info for info in self._video_infos if info["path"] == str(video)), None)
                            native_fps = vid_info["native_fps"] if vid_info else 30.0
                            target_fps = max(float(self.fps_extract.get()), 0.01)
                            step = max(1, round(native_fps / target_fps))
                            vf_filter = f"select='not(mod(n\\\\,{step}))',setpts=N/FRAME_RATE/TB"
                            vf_args = ["-vf", vf_filter, "-vsync", "vfr"] if step > 1 else []
                            cmd_raw = [
                                "ffmpeg", "-y",
                                "-i", str(video),
                                *vf_args,
                                "-pix_fmt", "rgb48be",
                                str(_raw_tmp / f"{prefix}_%04d.png")
                            ]
                            self._log_tagged("[CPU]", f"       Re-extracting {video.name} (16-bit raw, step={step})...")
                            proc, stderr_lines, reader_thread = self._run_ffmpeg_streamed(cmd_raw, "[CPU]")
                            while proc.poll() is None:
                                if self._cancelled:
                                    proc.kill()
                                    proc.wait()
                                    reader_thread.join(timeout=2)
                                    raise CancelledError("Processing cancelled by user")
                                time.sleep(0.25)
                            reader_thread.join(timeout=5)
                            if proc.returncode != 0:
                                self._log_tagged("[ERR]", f"       FFmpeg re-extraction failed for {video.name}")
                                continue

                        # Collect all re-extracted raw frames
                        raw_frames = sorted([
                            p for p in _raw_tmp.iterdir()
                            if p.is_file() and p.suffix.lower() == ".png"
                        ])
                        self._log_tagged("[CPU]", f"       \u2192 {len(raw_frames)} raw frames re-extracted for EXR conversion")

                        # Match raw frames to the SfM proxy frames by sorted order
                        sfm_proxy_frames = sorted([
                            p for p in images_dir.iterdir()
                            if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")
                        ])

                        # Convert each raw frame to the output color space as EXR
                        try:
                            configured_threads = int(self.num_workers.get())
                        except Exception:
                            configured_threads = 1
                        threads = max(1, configured_threads)
                        self._log_tagged("[CPU]", f"       \u2192 Converting {len(raw_frames)} frames to {_dest} EXR with {threads} process(es)...")
                        from concurrent.futures import ProcessPoolExecutor, as_completed
                        from concurrent.futures.process import BrokenProcessPool
                        _exr_ok = 0
                        _exr_errs = []
                        # Build mapping: raw frame -> EXR output path (using SfM proxy name)
                        _exr_tasks = []
                        for idx, raw_path in enumerate(raw_frames):
                            if idx < len(sfm_proxy_frames):
                                exr_name = sfm_proxy_frames[idx].stem + ".exr"
                            else:
                                exr_name = raw_path.stem + ".exr"
                            exr_out = ds_final_images / exr_name
                            _exr_tasks.append((str(raw_path), str(exr_out), _source, _dest, _cs_in, _cs_out, _cfg_path))
                        try:
                            with ProcessPoolExecutor(max_workers=threads) as executor:
                                _futs = [
                                    executor.submit(convert_frame_to_output_exr_worker, *args)
                                    for args in _exr_tasks
                                ]
                                for fut in as_completed(_futs):
                                    if self._cancelled:
                                        executor.shutdown(wait=False, cancel_futures=True)
                                        break
                                    try:
                                        ok, err = fut.result()
                                    except (BrokenProcessPool, Exception) as e:
                                        ok, err = False, str(e)
                                    if ok:
                                        _exr_ok += 1
                                    elif err:
                                        _exr_errs.append(err)
                        except BrokenProcessPool as e:
                            _exr_errs.append(f"Worker pool crashed: {e}")
                        if _exr_errs:
                            self._log_tagged("[CPU]", f"       \u274c {len(_exr_errs)} EXR conversion error(s): {list(set(_exr_errs))[:3]}")
                        self._log_tagged("[CPU]", f"       \u2713 {_exr_ok}/{len(raw_frames)} EXR files written to {ds_final_images}")

                    # Update images.bin in sparse model: rename .png -> .exr
                    try:
                        from backend.colmap_images import read_images_bin as _read_images_bin, write_images_bin as _write_images_bin
                        _bin_path = sfm_dir / "images.bin"
                        if _bin_path.exists():
                            _images_data = _read_images_bin(_bin_path)
                            _renamed = 0
                            for _img in _images_data:
                                for _ext in (".png", ".jpg", ".jpeg"):
                                    if _img["name"].lower().endswith(_ext):
                                        _img["name"] = _img["name"][:-(len(_ext))] + ".exr"
                                        _renamed += 1
                                        break
                            if _renamed > 0:
                                _write_images_bin(_bin_path, _images_data)
                                self._log_tagged("[CPU]", f"       \u2192 Sparse model updated: {_renamed} image reference(s) renamed to .exr")
                        else:
                            self._log_tagged("[CPU]", "       \u26a0 images.bin not found in sparse/0")
                    except Exception as _exc:
                        self._log_tagged("[CPU]", f"       \u26a0 Could not update sparse model image names: {_exc}")

                    self._log_tagged("[CPU]", "       \u2713 Post-SfM EXR conversion complete")

                try:
                    normalize_final_bundle(
                        base_out,
                        keep_srgb_png=self.color_enabled.get() and is_video_mode,
                        use_acescg_exr=self.use_acescg_exr.get(),
                    )
                    self._log_tagged("[CPU]", "       -> Final bundle normalized; images.bin patched")
                except Exception as _bundle_exc:
                    self._log_tagged("[CPU]", f"       \u26a0 Could not normalize final bundle: {_bundle_exc}")

                self._log_tagged("[OK]", f"\\n\u2713 SUCCESS \u2014 Dataset ready: {base_out}")
'''

content = content[:anchor_idx] + NEW_POST_SFM + content[success_end:]
print("[OK] Added post-SfM EXR generation step")

# Restore \r\n line endings (Windows)
content = content.replace("\r\n", "\n").replace("\n", "\r\n")

open(gui_path, "wb").write(content.encode("utf-8"))
print("[OK] ReMap-GUI.py patched successfully")
