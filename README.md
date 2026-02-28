# ReMap

**ReMap** is a desktop pipeline to prepare your 3D captures for [Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). It takes videos or LiDAR scans as input and produces a fully reconstructed COLMAP dataset, ready for training.

It is the **desktop companion** to [**ReScan**](https://github.com/YvigUnderscore/ReScan), our iOS LiDAR capture app. Together, they form a complete end-to-end workflow: **capture on iPhone → process on desktop → train your Gaussian Splat. (trough another software)**

---

## 🔄 The ReMap + ReScan Workflow

```text
┌─────────────────────┐         ┌─────────────────────┐         ┌──────────────────────┐
│      📱 ReScan      │   API   │      🖥️ ReMap      │  output │   🎯 3DGS Training   │
│   (iOS Capture)     │ ──────► │  (Desktop Pipeline) │ ──────► │  (Nerfstudio, etc.)  │
│                     │         │                     │         │                      │
│ • LiDAR depth maps  │         │ • FFmpeg extraction │         │ • models/0/0/        │
│ • RGB video (Log)   │         │ • OCIO color mgmt   │         │   cameras.bin        │
│ • ARKit odometry    │         │ • HLoc features     │         │   images.bin         │
│ • Camera intrinsics │         │ • COLMAP / GLOMAP   │         │   points3D.bin       │
│                     │         │                     │         │   images/            │
└─────────────────────┘         └─────────────────────┘         └──────────────────────┘
```

> **Don't have an iPhone with LiDAR?** No problem — ReMap also works with plain **MP4/MOV videos** from any camera or with **image folders**. ReScan is optional but unlocks the full LiDAR-guided workflow.

---

## 🌐 Wireless Capture API

The most powerful way to use ReMap is by connecting it directly to **ReScan** over your local network using the built-in REST API (You can also port-forward to open-it remotely).

1. Start the API Server from the **ReMap GUI** or the headless server mode.
2. An **API Key** is instantly generated, you can also customize it (before starting the server).
3. Open **ReScan** on iOS, enter your computer/server IP and API Key.
4. **Capture and send** datasets directly from your phone — ReMap will automatically unpack, process, and spit out the fully solved 3D model!

> 📖 **Developers:** See the comprehensive [API Documentation](API_DOCUMENTATION.md) to integrate ReMap with your own HTTP clients, complete with Swift `URLSession` examples.

---

## ✨ Features

- **🌐 Live API Server**: Remote dataset upload and automated processing from the [ReScan](https://github.com/YvigUnderscore/ReScan) iOS app.
- **🎨 Advanced EXR & Color Pipeline**: Deep OpenColorIO processing. Convert from camera Log to ACEScg before SfM. Automatically exports 16-bit half-float or 32-bit float **EXR** datasets for professional workflows.
- **📸 3 Input Modes**: Video files (`.mp4`, `.mov`), Image folders, or ReScan LiDAR datasets.
- **⚡ GPU-Accelerated Extraction & SfM**: FFmpeg with CUDA hardware decoding, and dual SfM engines (choose robust CPU COLMAP or fast GPU [GLOMAP](https://github.com/colmap/glomap)).
- **📍 LiDAR-Guided Reconstruction**: When using ReScan data, ARKit poses can triangulate the model instantly or guide the full SfM solver.
- **🧠 State-of-the-Art Matching**: SuperPoint, DISK, ALIKED features with LightGlue / SuperGlue matching via [HLoc](https://github.com/cvg/Hierarchical-Localization).
- **🛡️ Secure & Optimized**: Multi-threaded file processing skips existing outputs. Secure installation scripts with pinned submodules ensure a reproducible environment preventing missing dependencies. Windows file-lock handling for ultimate stability. Modern CustomTkinter dark UI.

---

## 📂 Output Format (Drag-and-Drop)

ReMap produces a standard **COLMAP** dataset structure, specifically optimized to be bundled into a **single, self-contained directory** for modern training engines:

```text
output_directory/
├── hloc_outputs/             # Intermediate feature files
└── sparse/
    └── 0/
        └── models/
            └── 0/
                └── 0/        # 🎯 Drag-and-drop this folder to your trainer!
                    ├── cameras.bin
                    ├── images.bin      # Image paths are automatically re-linked!
                    ├── points3D.bin
                    └── images/         # Bundled PNG or EXR frames
                        ├── 000001.exr
                        └── ...
```

> 💡 **Ready for 3DGS**: Because the `images/` directory is bundled inside the sparse model alongside tweaked `.bin` files, you can immediately drag the `0/0/` folder into tools like **Brush**, **Nerfstudio**, or **gsplat**.

---

## 🚀 Installation

ReMap supports both **Windows** and **Linux**. We provide secure scripts to make this a "one-click install."

### Prerequisites

| Requirement | Details |
|---|---|
| **OS** | Windows 10/11 or Linux (Ubuntu 22.04+ recommended) |
| **GPU** | NVIDIA GPU with CUDA (strongly recommended) |
| **Python** | 3.10 or 3.12 (A Virtual Environment is automatically created) |

### The "One-Click" Automated Install (Recommended)

Our installer handles system dependencies, pinned submodule clones (COLMAP/GLOMAP), Python venvs, and pip packages securely.

**For Linux (Ubuntu):**
```bash
git clone https://github.com/YvigUnderscore/ReMap.git
cd ReMap
sudo ./install_all.sh
./launch.sh
```

**For Windows:**
```bat
git clone https://github.com/YvigUnderscore/ReMap.git
cd ReMap
install_all.bat
launch.bat
```

*(For a step-by-step manual setup, refer to older commit histories or open the installer scripts to see the exact system packages required).*

---

## 🎨 Advanced Color Management (OCIO & EXR)

ReMap uses **OpenImageIO** to process OpenColorIO workflows, essential for flat/log footage.

1. Set your `OCIO` environment variable, or browse for it directly in the app.
2. If converting to linear spaces (like **ACEScg**), ReMap enables its **EXR Pipeline**. 
3. Outputs are heavily optimized:
    - **16-bit PNG** (For integer-based linear workflows)
    - **16-bit half-float EXR** (Default for ACES/scene-linear — highly efficient)
    - **32-bit float EXR** (Can be forced manually for maximum precision)

---

## ⚙️ Usage Guide

### Video & Image Folder Modes
1. Select **Video** or **Image Folder** mode.
2. Set extraction FPS (2–5 FPS recommended for video).
3. Configure the SfM pipeline (defaults work well).
4. Click **⚡ START PROCESSING**.

### ReScan (LiDAR) Mode
1. Select **Rescan (LiDAR)** mode.
2. Browse to the dataset containing `rgb.mp4` (Apple ProRes Log) or already exported frames into `/images`, `odometry.csv`, and `camera_matrix.csv`.
3. Choose processing approach:
   - **Full SfM (Approach B)** — Runs complete SfM. Most robust.
   - **ARKit Poses (Approach A)** — Uses known poses for fast triangulation.
4. Set extraction FPS to perfectly sync Log video and ARKit odometry.

### Recommended Settings

| Setting | Video (Standard) | Video (Apple Log) | ReScan (LiDAR) |
|---|---|---|---|
| Features / Matcher | SuperPoint / LightGlue | SuperPoint / LightGlue | SuperPoint / LightGlue |
| SfM Engine | COLMAP / GLOMAP | COLMAP / GLOMAP | COLMAP |
| OCIO | Off | On (Log → ACEScg) | Off |
| Image Output | PNG (8-bit) | EXR (16-bit Half) | PNG (8-bit) |

---

## 📄 License & 🤝 Acknowledgements

**License:** Output data (reconstructions) are 100% yours to commercialize. The source code is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (Non-Commercial). 

**Thanks to:** [HLoc](https://github.com/cvg/Hierarchical-Localization) | [COLMAP](https://colmap.github.io/) | [GLOMAP](https://github.com/colmap/glomap) | [SuperPoint & LightGlue](https://github.com/cvg/LightGlue) | [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) | [ReScan](https://github.com/YvigUnderscore/ReScan)
