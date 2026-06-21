<p align="center">
  <img src="ReMap_logo.png" alt="ReMap" width="240">
</p>

# ReMap

ReMap is a desktop reconstruction pipeline for preparing captures for
[3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).
It turns videos, image folders, or ReScan LiDAR captures into a COLMAP-style
dataset that can be dropped into training tools such as Brush, Nerfstudio, or
gsplat.

ReMap also pairs with [ReScan](https://github.com/YvigUnderscore/ReScan), the
iOS LiDAR capture app. ReScan is optional: the same desktop pipeline works with
regular MP4/MOV footage and image sequences.

<!-- Image slot: docs/images/remap-desktop-dashboard.png
     Suggested: full-width screenshot of the new Tauri dashboard with the job wizard and queue visible. -->

## Current Shape

The project is in the middle of a clean desktop split:

| Layer | What it does |
| --- | --- |
| React + Tauri desktop shell | New main UI, file pickers, notifications, QR quick connect, job monitoring |
| Local desktop backend | Flask service on `127.0.0.1:8765` for jobs, settings, cache, analytics, previews |
| Legacy pipeline bridge | Reuses the proven `ReMap-GUI.py` processing logic without forcing the old UI |
| ReScan API server | Authenticated `/api/v1` server on port `5000` for phone uploads and remote processing |

The previous CustomTkinter interface is still available through
`launch_legacy.bat` or `launch_legacy.sh`.

## Highlights

- New desktop app built with React, Tauri, Tailwind, Framer Motion, and lucide icons.
- Job queue with pause, resume, cancel, delete, requeue, live logs, ETA, and batch creation.
- Drag-and-drop input selection for videos, image folders, and ReScan datasets.
- Input probing with FFprobe/metadata detection, frame estimates, native FPS, and color profile hints.
- Quality sweep presets for fast, balanced, and high-quality test runs.
- Optional frame rejection for blurry or near-black frames.
- Global pipeline cache for reusable HLoc/LoMa feature, pair, and match files.
- Checkpoint manifests in `.remap/` for easier reruns and debugging.
- Live analytics for CPU, RAM, disk, CUDA/GPU memory, temperature, matchers, features, and recent job throughput.
- 3D sparse reconstruction preview in the Jobs view, powered by Three.js.
- ReScan Quick Connect with local URL, API key copy buttons, and QR code.
- Dependency and model manager for pinned packages, LoMa, LightGlue, SuperGlue, and cache warmup.
- LoMa-B and LoMa-G matching support alongside LightGlue, SuperGlue, AdaLAM, DISK, ALIKED, SIFT, and SuperPoint.
- ACEScg/EXR color pipeline with Apple Log, BT.2020, HLG, sRGB, Linear sRGB, and custom OCIO options.

## Pipeline

```text
Capture or images
  -> input probe and frame selection
  -> optional color conversion / EXR proxy generation
  -> HLoc or LoMa features
  -> sequential or exhaustive pairs
  -> matching
  -> COLMAP or GLOMAP reconstruction
  -> bundle normalization
  -> trainer-ready output folder
```

For ReScan datasets, ReMap can use ARKit odometry and LiDAR depth as part of the
conversion step, then either run full SfM or use known poses for faster
triangulation.

<!-- Image slot: docs/images/remap-new-job-wizard.png
     Suggested: screenshot of the Source/Preparation/Color/Reconstruction wizard steps. -->

## Inputs

| Mode | Accepted input | Notes |
| --- | --- | --- |
| Video | `.mp4`, `.mov`, `.avi`, `.webm`, `.mkv`, `.m4v` | FPS extraction controls how many frames enter SfM |
| Images | Folder of `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, `.exr`, `.webp`, `.bmp` | EXR sequences are supported for HDR/color-managed workflows |
| ReScan | Folder or uploaded ZIP with `rgb.mp4`/`rgb.mov` or `rgb/`, plus `odometry.csv` and `camera_matrix.csv` | Optional `depth/` and `confidence/` folders can guide LiDAR processing |

## Output Layout

The new desktop flow creates a normalized final bundle next to the working
pipeline output:

```text
<output_path>/
|-- <output_name>_SfM_Dataset_Output/
|   |-- cameras.bin
|   |-- images.bin
|   |-- points3D.bin
|   |-- images/
|   |   |-- 000001.exr
|   |   `-- ...
|   `-- images_srgb_png/        # optional preview/proxy folder for EXR jobs
|-- hloc_outputs/
|-- sparse/0/
`-- .remap/
    |-- checkpoints.json
    `-- fingerprint.json
```

Use `<output_name>_SfM_Dataset_Output/` as the drag-and-drop folder for most
Gaussian Splatting trainers. The legacy `sparse/0/models/0/0/` bundle may still
exist for compatibility, but the normalized root-level bundle is the preferred
target.

<!-- Image slot: docs/images/remap-output-bundle.png
     Suggested: file explorer screenshot showing the normalized final bundle and images folder. -->

## Installation

### Requirements

| Requirement | Why it matters |
| --- | --- |
| Windows 10/11 or Ubuntu 22.04+ | Main supported desktop targets |
| NVIDIA GPU with CUDA | Strongly recommended for matching and reconstruction speed |
| Python 3.10 or 3.12 | Installer creates `.venv` and installs pinned Python packages |
| Git and FFmpeg | Required for dependency install and frame extraction/probing |
| COLMAP | Required SfM backend |
| GLOMAP | Optional, used when available for faster global mapping |
| Node.js 20+ | Windows installer uses a portable local Node.js without admin rights; Linux installer uses the system package manager |
| Rust toolchain | Installer checks/installs it and fetches Tauri Cargo dependencies |

The Python baseline is pinned in `requirements.txt` and
`requirements.lock.txt`. LoMa is installed separately by the installer and the
Settings dependency manager because the upstream package currently needs a small
metadata workaround.

### Windows

```bat
git clone https://github.com/YvigUnderscore/ReMap.git
cd ReMap
install_all.bat
launch.bat
```

### Linux

```bash
git clone https://github.com/YvigUnderscore/ReMap.git
cd ReMap
sudo ./install_all.sh
./launch.sh
```

If you only need the previous CustomTkinter interface, use `launch_legacy.bat`
or `./launch_legacy.sh` after running the installer.

## Launch Options

| Command | Purpose |
| --- | --- |
| `launch.bat` / `./launch.sh` | Starts the local backend and opens the new Tauri desktop app |
| `launch_legacy.bat` / `./launch_legacy.sh` | Opens the previous CustomTkinter UI |
| `python desktop_backend.py` | Starts only the local desktop backend on `127.0.0.1:8765` |
| `npm run dev` | Runs the React UI in a browser during frontend development |
| `npm run desktop:dev` | Runs the full Tauri desktop app in development mode |
| `python remap_server.py --api-key "secret"` | Starts the ReScan-compatible `/api/v1` server |

## Typical Desktop Workflow

1. Open ReMap with `launch.bat` or `./launch.sh`.
2. Create a job from the New Job wizard.
3. Drop videos, image folders, or ReScan datasets into the source area.
4. Pick an output directory and tune FPS, matching, color, and reconstruction options.
5. Review the frame/pair/disk/time estimates.
6. Start the job or launch a quality sweep.
7. Track progress in Jobs, inspect logs and artifacts, then open the final bundle.

<!-- Image slot: docs/images/remap-jobs-preview.png
     Suggested: screenshot of the Jobs page with logs, artifacts, ETA, and sparse preview visible. -->

## ReScan Remote Capture

The remote API server lets ReScan upload datasets over the local network:

1. Open the Server page in the desktop app.
2. Set the host, port, API key, and output directory.
3. Start the `/api/v1` server.
4. Scan the Quick Connect QR code in ReScan or copy the URL and key manually.
5. Capture on iPhone, upload to ReMap, then monitor processing from desktop or API.

The standalone server can also be launched directly:

```bash
python remap_server.py --host 0.0.0.0 --port 5000 --api-key "your-key"
```

See [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for endpoint details, curl
examples, payload fields, colorspace values, and the internal desktop API.

<!-- Image slot: docs/images/remap-rescan-quick-connect.png
     Suggested: screenshot of the Server page showing QR code, base URL, API key, and health state. -->

## Color Pipeline

ReMap has two goals when color management is enabled: give SfM a stable image
proxy to match against, and keep the final training images in the color space
you actually want.

Built-in sources include:

- Auto-detect
- Apple Log (BT.2020)
- HLG (BT.2020)
- Linear BT.2020
- Linear ACEScg
- Linear sRGB
- sRGB (Rec.709)

Built-in outputs include:

- ACEScg with EXR output plus optional sRGB PNG previews
- Linear sRGB
- sRGB tone mapped
- Custom OCIO transforms

For Apple Log or HDR sources, the recommended path is usually ACEScg EXR output
with `images_srgb_png/` kept for quick inspection.

## Matching And Reconstruction

| Stage | Options |
| --- | --- |
| Features | `superpoint_aachen`, `superpoint_max`, `disk`, `aliked-n16`, `sift` |
| Matchers | `superpoint+lightglue`, `superglue`, `disk+lightglue`, `adalam`, `loma_b`, `loma_g` |
| Pairing | Sequential video pairs or exhaustive pairs for smaller sets |
| SfM | COLMAP or GLOMAP |
| ReScan pose mode | Full SfM or ARKit known poses |

`loma_b` is the faster LoMa option. `loma_g` is heavier and more accurate, and
is best reserved for difficult datasets or quality sweeps.

## Repository Layout

```text
backend/                  Local desktop backend services
frontend/                 React/Tauri UI source
src-tauri/                Tauri shell, file dialogs, notifications
backend_state/            Local settings, job store, cache, fault logs
LUTS/                     Apple Log and HDR conversion LUTs
SuperGluePretrainedNetwork/  External SuperGlue weights/code folder
ReMap-GUI.py              Legacy UI and processing implementation
remap_server.py           ReScan-compatible public API server
desktop_backend.py        Local desktop backend entry point
```

## Troubleshooting

- If the new UI says frontend dependencies are missing, rerun the installer and choose option `10`.
- On Windows, rerun `install_all.bat --node` to refresh the portable local Node.js in `.tools/node`.
- If Tauri fails to start, confirm Rust/Cargo is installed and available in the shell.
- If Python requirements fail to install, rerun installer option `7` and inspect `backend_state/install_logs/pip_install.log`.
- If GLOMAP is missing, ReMap falls back to COLMAP.
- If EXR or OCIO conversion fails, check OpenImageIO in Settings and verify the selected `.ocio` file.
- If a run repeats the same work, check whether `skip_existing` and the global cache are enabled.
- Use `kill_remap.bat`, `kill_remap.ps1`, or `kill_remap.sh` if a backend or server process is stuck.

## License And Acknowledgements

Output data and reconstructions are yours to use commercially. The source code
is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

Thanks to
[HLoc](https://github.com/cvg/Hierarchical-Localization),
[COLMAP](https://colmap.github.io/),
[GLOMAP](https://github.com/colmap/glomap),
[LightGlue](https://github.com/cvg/LightGlue),
[SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork),
[LoMa](https://github.com/davnords/LoMa),
[OpenImageIO](https://openimageio.readthedocs.io/),
[Tauri](https://tauri.app/), and
[ReScan](https://github.com/YvigUnderscore/ReScan).
