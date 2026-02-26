# ReMap

**ReMap** is a desktop pipeline to prepare your 3D captures for [Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). It takes videos or LiDAR scans as input and produces a fully reconstructed COLMAP dataset, ready for training.

It is the **desktop companion** to [**ReScan**](https://github.com/YvigUnderscore/ReScan), our iOS LiDAR capture app. Together, they form a complete end-to-end workflow: **capture on iPhone → process on desktop → train your Gaussian Splat.**

---

## 🔄 The ReMap + ReScan Workflow

```
┌─────────────────────┐         ┌─────────────────────┐         ┌──────────────────────┐
│      📱 ReScan      │   USB   │      🖥️ ReMap       │  output │   🎯 3DGS Training   │
│   (iOS Capture)     │ ──────► │  (Desktop Pipeline) │ ──────► │  (Nerfstudio, etc.)  │
│                     │  copy   │                     │         │                      │
│ • LiDAR depth maps  │         │ • FFmpeg extraction │         │ • images/            │
│ • RGB video (Log)   │         │ • OCIO color mgmt   │         │ • sparse/0/          │
│ • ARKit odometry    │         │ • HLoc features     │         │   cameras.bin        │
│ • Camera intrinsics │         │ • COLMAP / GLOMAP   │         │   images.bin         │
│                     │         │                     │         │   points3D.bin       │
└─────────────────────┘         └─────────────────────┘         └──────────────────────┘
```

> **Don't have an iPhone with LiDAR?** No problem — ReMap also works with plain **MP4/MOV videos** from any camera or with **image folders**. ReScan is optional but unlocks the full LiDAR-guided workflow.

---

## ✨ Features

- **3 Input Modes**: Video files (`.mp4`, `.mov`), Image folders, or [ReScan](https://github.com/YvigUnderscore/ReScan) LiDAR datasets
- **GPU-Accelerated Extraction**: FFmpeg with CUDA hardware decoding when available
- **16-bit Smart Mode**: Automatically enables 16-bit PNG output when converting to linear colorspaces (ACEScg, scene-linear). Can also be forced manually
- **OCIO Color Management**: Full OpenColorIO pipeline — convert from camera Log to any working space before SfM
- **State-of-the-Art SfM**: SuperPoint, DISK, ALIKED features with LightGlue / SuperGlue matching via [HLoc](https://github.com/cvg/Hierarchical-Localization)
- **Dual SfM Engines**: Choose between COLMAP (robust, CPU) or [GLOMAP](https://github.com/colmap/glomap) (fast, GPU)
- **LiDAR-Guided Reconstruction**: When using ReScan data, known ARKit poses can be used for triangulation (Approach A) or full SfM (Approach B)
- **In-App Guidance**: ⓘ tooltips on every section explain settings and provide recommended configurations
- **Auto-Skip**: If images already exist in the output directory, extraction is skipped automatically
- **Modern Dark UI**: Built with [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)

---

## 📁 Output Format

ReMap produces a standard **COLMAP** dataset structure:

```
output_directory/
├── images/               # Extracted / copied images (PNG)
│   ├── vid01_0001.png
│   ├── vid01_0002.png
│   └── ...
├── hloc_outputs/         # Intermediate HLoc files
│   ├── features.h5
│   ├── pairs.txt
│   └── matches.h5
└── sparse/
    └── 0/                # Reconstructed COLMAP model
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

This output is directly compatible with:
- [Brush](https://github.com/ArthurBrussee/brush) (Recommended) — *Huge thanks to the author!*
- [Nerfstudio](https://docs.nerf.studio/)
- [gsplat](https://github.com/nerfstudio-project/gsplat)
- Any COLMAP-based training pipeline

---

## 🚀 Installation

ReMap supports both **Windows** and **Linux** operating systems. We provide scripts to make this as close to a "one-click install" as possible.

### Prerequisites & Dependencies

To run ReMap, your system needs a few core components. Our automated scripts handle downloading and configuring most of these, but here's a breakdown of what happens under the hood:

| Requirement | Details |
|---|---|
| **OS** | Windows 10/11 or Linux (Ubuntu 22.04+ recommended) |
| **GPU** | NVIDIA GPU with CUDA (strongly recommended for HLoc and GLOMAP) |
| **Python** | 3.10 or 3.12 (A Virtual Environment is **critical**, see below) |
| **FFmpeg** | Required for parsing videos and extracting frames |
| **COLMAP** | The core Structure-from-Motion engine |
| **GLOMAP** | (Optional) A faster, GPU-accelerated alternative to COLMAP |

> ⚠️ **The Importance of Virtual Environments (Venv)**
> Due to the complex nature of photogrammetry and Deep Learning dependencies (like PyTorch with specific CUDA versions, HLoc, pycolmap, etc.), it is perfectly normal and **highly recommended** to use an isolated Python Virtual Environment (`.venv`). This keeps ReMap's dependencies strictly separate from your underlying system or other projects, preventing version conflicts, missing packages, and keeping your OS clean. Our installation scripts handle this automatically.

### Method 1: The "One-Click" Automated Install (Recommended)

The `install_all.sh` (Linux) and `install_all.bat` (Windows) scripts handle everything for you: system dependencies, downloading COLMAP/GLOMAP, setting up the Python venv, and installing pip packages. It's essentially a one-click process that sets up the ideal isolated environment.

**For Linux (Ubuntu):**
```bash
# 1. Clone the repository
git clone https://github.com/YvigUnderscore/ReMap.git
cd ReMap

# 2. Run the installer (requires sudo for system packages)
sudo ./install_all.sh

# 3. Launch ReMap
./launch.sh
```

**For Windows:**
```bat
# 1. Clone the repository (or download as ZIP and extract)
git clone https://github.com/YvigUnderscore/ReMap.git
cd ReMap

# 2. Double-click the installer script
install_all.bat

# 3. Launch ReMap
launch.bat
```

> **Details on the automated install:** The script performs these steps sequentially:
> 1. **System packages** — Grabs build tools, CMake, Boost, COLMAP dependencies, FFmpeg.
> 2. **COLMAP** — Installs via system packages (Linux) or downloads pre-built binaries (Windows).
> 3. **GLOMAP** — Clones and builds from source (Linux) or downloads binaries (Windows).
> 4. **Python venv** — Creates the isolated `.venv` environment to protect your system.
> 5. **Python deps** — Installs PyTorch (CUDA), HLoc, pycolmap, CustomTkinter, and all requirements inside the environment.

### Method 2: Manual Install (Linux Example)

If you prefer to install components individually or need a custom setup:

#### Step 1 — System Dependencies

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build git ffmpeg \
    libboost-all-dev libceres-dev libfreeimage-dev libglew-dev \
    qtbase5-dev libflann-dev libeigen3-dev libmetis-dev libsqlite3-dev \
    python3-pip python3-venv colmap
```

#### Step 2 — Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 3 — GLOMAP (Optional)

```bash
git clone --recursive https://github.com/colmap/glomap.git
cd glomap
cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release
ninja -C build
sudo ninja -C build install
cd ..
```

#### Step 4 — Launch

```bash
source .venv/bin/activate
python3 ReMap-GUI.py
```

### OCIO Support

ReMap uses **OpenImageIO** (installed automatically via `requirements.txt`) to process OpenColorIO workflows. If you provide a valid standalone `.ocio` file (e.g., from ACES), ReMap will convert input images from Log/Raw colorspaces into a linear working space before extracting features. This improves matching robustness on flat/log footage.

You can set the `OCIO` environment variable to your config file, or browse for it directly in the app:

```bash
export OCIO=/path/to/your/config.ocio
```

---

## ⚙️ Usage Guide

### Video Mode

1. Select **Video (.mp4, .mov)** as input mode
2. Click **Browse** to select one or more video files
3. Set the **extraction FPS** (2–5 FPS recommended for photogrammetry, higher for fast-moving scenes)
4. Choose your output folder
5. Configure the SfM pipeline (defaults work well for most cases)
6. Click **⚡ START PROCESSING**

### Image Folder Mode

1. Select **Image Folder** as input mode
2. Browse to the folder containing your images (JPG, PNG, TIFF)
3. Images will be copied to the output directory and processed

### ReScan (LiDAR) Mode

> Requires a dataset captured with [ReScan](https://github.com/YvigUnderscore/ReScan) on iPhone.

1. Select **Rescan (LiDAR)** as input mode
2. Browse to the ReScan dataset folder(s) containing `rgb.mp4` or `rgb.mov` (Apple ProRes Log), `odometry.csv`, and `camera_matrix.csv`
3. Choose your reconstruction approach:
   - **Full SfM (Approach B)** — Runs complete SfM pipeline. Most robust
   - **ARKit Poses (Approach A)** — Uses known LiDAR poses for triangulation. Faster
4. Adjust **Extraction FPS** to control the number of frames extracted (the app will perfectly sync Apple Log video and ARKit Odometry based on this FPS)

### Recommended Settings

| Setting | Video (Standard) | Video (Apple Log) | ReScan (LiDAR) |
|---|---|---|---|
| Extraction FPS | 2–5 FPS | 2–5 FPS | 2–5 FPS |
| Features | SuperPoint | SuperPoint | SuperPoint |
| Matcher | LightGlue | LightGlue | LightGlue |
| Pairing | Sequential | Sequential | Sequential |
| SfM Engine | COLMAP | COLMAP | COLMAP |
| 16-bit | Off | Auto (via OCIO) | Off |
| OCIO | Off | On (Log → ACEScg) | Off |

---

## 📄 License

1. **Output Data (Your Reconstructions)**: You own 100% of the data you produce (images, point clouds, COLMAP models, etc.). You are free to use, modify, distribute, and commercialize any output without restrictions.

2. **The Software (Source Code)**: The ReMap source code is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/) License.

    - **Share** — Copy and redistribute in any medium or format
    - **Adapt** — Remix, transform, and build upon the code
    - **NonCommercial** — You may not use the source code for commercial purposes
    - **Commercial Usage** — Requires prior written consent from the author

See the [LICENSE](LICENSE) file for the full legal text.

---

## 🙏 Acknowledgements

- [HLoc](https://github.com/cvg/Hierarchical-Localization) — Hierarchical Localization
- [COLMAP](https://colmap.github.io/) — Structure from Motion
- [GLOMAP](https://github.com/colmap/glomap) — Global Mapper
- [SuperPoint + LightGlue](https://github.com/cvg/LightGlue) — Feature detection & matching
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) — Modern Tkinter UI
- [ReScan](https://github.com/YvigUnderscore/ReScan) — iOS LiDAR capture companion app
