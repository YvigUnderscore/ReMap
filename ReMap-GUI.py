import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog as tk_filedialog
import shutil
import threading
import sys
import multiprocessing
import subprocess
import secrets
import tempfile
from pathlib import Path
import os
import logging
import math
import time
from tqdm import tqdm as _original_tqdm
import cv2
import numpy as np
from PIL import Image
from backend.color_conversion_worker import process_image_color_worker
from backend.bundle_postprocess import normalize_final_bundle
from backend.frame_filter import reject_low_quality_frames

try:
    import OpenImageIO as oiio
    HAS_OCIO = True
except ImportError:
    oiio = None
    HAS_OCIO = False

# --- Color Space Conversion System ---
# Source profiles: describes what the input data looks like
COLOR_SOURCES = [
    "Auto-detect",
    "Linear BT.2020",
    "Linear ACEScg",
    "Apple Log (BT.2020)",
    "Linear sRGB",
    "sRGB (Rec.709)",
    "HLG (BT.2020)",
]

# Destination profiles: where we want to go
COLOR_DESTINATIONS = [
    "ACEScg (EXR + sRGB PNG)",
    "Linear sRGB",
    "sRGB (Tone Mapped)",
    "Custom OCIO...",
]

ACESCG_OCIO_SPACE = "ACES - ACEScg"

# ffprobe metadata → source profile mapping
_FFPROBE_TO_SOURCE = {
    ("bt2020", "linear"):           "Linear BT.2020",
    ("bt2020", "alog"):             "Apple Log (BT.2020)",
    ("bt2020", "arib-std-b67"):     "HLG (BT.2020)",
    ("bt709", "bt709"):             "sRGB (Rec.709)",
    ("bt709", "linear"):            "Linear sRGB",
    ("bt709", "iec61966-2-1"):      "sRGB (Rec.709)",
    ("bt709", "srgb"):              "sRGB (Rec.709)",
}

# --- Gamut Matrices (computed from CIE XYZ with Bradford D65↔D60 adaptation) ---
# BT.2020 → ACEScg (AP1)
_MAT_BT2020_TO_ACESCG = np.array([
    [ 0.97990525,  0.02225227, -0.03192382],
    [-0.00058388,  0.99476128,  0.01081350],
    [ 0.00046861,  0.01941638,  1.06066918]
], dtype=np.float32)

# ACEScg (AP1) → sRGB (Rec.709) — for dual-output PNG generation
_MAT_ACESCG_TO_SRGB = np.array([
    [ 1.70298067, -0.62451279, -0.03670953],
    [-0.12985749,  1.14073295, -0.01436027],
    [-0.02069324, -0.12236011,  1.05442752]
], dtype=np.float32)

# BT.2020 → sRGB (Rec.709)
_MAT_BT2020_TO_SRGB = np.array([
    [ 1.66022663, -0.58754766, -0.07283817],
    [-0.12455332,  1.13292610, -0.00834968],
    [-0.01815514, -0.10060303,  1.11899821]
], dtype=np.float32)

# sRGB (Rec.709) → ACEScg (AP1)
_MAT_SRGB_TO_ACESCG = np.array([
    [0.61590865, 0.34031053, 0.01410133],
    [0.06855723, 0.91546150, 0.02095702],
    [0.01902455, 0.11135884, 0.94994319]
], dtype=np.float32)


def _apply_matrix(rgb, matrix):
    """Apply a 3×3 color matrix to an (H, W, 3) or (N, 3) array."""
    orig_shape = rgb.shape
    flat = rgb.reshape(-1, 3)
    result = np.dot(flat, matrix.T)
    return result.reshape(orig_shape)


def _apple_log_to_linear(P):
    """
    Decode Apple Log encoded values to scene-linear.
    Based on Apple Log Profile White Paper.
    """
    R_cut = 0.00104
    a = 5.555556
    b = 0.047996
    c = 0.529136
    d = 0.089004
    e_lin = 10.444689
    f = 0.180395
    E_cut = e_lin * R_cut + f  # ~0.1913

    E = P.astype(np.float32)
    R = np.where(
        E >= E_cut,
        (np.power(2.0, (E - d) / c) - b) / a,
        (E - f) / e_lin
    )
    return np.maximum(R, 0.0)


def _hlg_eotf(E):
    """
    Hybrid Log-Gamma (HLG) OETF inverse — decode HLG signal to scene-linear.
    ITU-R BT.2100.
    """
    a = 0.17883277
    b = 1.0 - 4.0 * a
    c = 0.5 - a * np.log(4.0 * a)
    E = np.asarray(E, dtype=np.float32)
    return np.where(
        E <= 0.5,
        (E ** 2) / 3.0,
        (np.exp((E - c) / a) + b) / 12.0
    )


def _srgb_eotf(E):
    """sRGB gamma decode (sRGB display → linear)."""
    E = np.asarray(E, dtype=np.float32)
    return np.where(
        E <= 0.04045,
        E / 12.92,
        np.power(np.maximum((E + 0.055) / 1.055, 0.0), 2.4)
    )


def _srgb_oetf(x):
    """sRGB gamma encoding (linear → sRGB display)."""
    return np.where(x <= 0.0031308, 12.92 * x,
                    1.055 * np.power(np.maximum(x, 1e-7), 1/2.4) - 0.055)


def _aces_tonemap(x):
    """Narkowicz ACES filmic tone mapping curve."""
    a = 2.51; b = 0.03; c = 2.43; d = 0.59; e = 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


def _linearize(rgb_norm, source):
    """Apply the correct EOTF based on source profile → scene-linear RGB."""
    if source in ("Linear BT.2020", "Linear ACEScg", "Linear sRGB"):
        return rgb_norm  # already linear
    elif source == "Apple Log (BT.2020)":
        return _apple_log_to_linear(rgb_norm)
    elif source == "HLG (BT.2020)":
        return _hlg_eotf(rgb_norm)
    elif source == "sRGB (Rec.709)":
        return _srgb_eotf(rgb_norm)
    return rgb_norm


def _source_primaries(source):
    """Return 'bt2020' or 'srgb' for a given source profile."""
    if source in ("Linear BT.2020", "Apple Log (BT.2020)", "HLG (BT.2020)"):
        return "bt2020"
    if source == "Linear ACEScg":
        return "acescg"
    return "srgb"


def _gamut_convert(rgb_linear, src_primaries, dst_primaries):
    """Convert between gamuts using pre-computed 3×3 matrices."""
    key = (src_primaries, dst_primaries)
    matrices = {
        ("bt2020", "acescg"):  _MAT_BT2020_TO_ACESCG,
        ("bt2020", "srgb"):    _MAT_BT2020_TO_SRGB,
        ("srgb",   "acescg"):  _MAT_SRGB_TO_ACESCG,
        ("acescg", "srgb"):    _MAT_ACESCG_TO_SRGB,
    }
    if key in matrices:
        return _apply_matrix(rgb_linear, matrices[key])
    if src_primaries == dst_primaries:
        return rgb_linear
    return rgb_linear  # fallback: no conversion

# --- Theme & Colors ---
try:
    import torch
    from hloc import extract_features, match_features, reconstruction, pairs_from_exhaustive
    from backend.loma_matcher import LoMaMatcher, is_loma_matcher, loma_feature_path, loma_matches_path
    import pycolmap
    from sfm_runner import run_sfm_with_live_export
    from stray_to_colmap import convert_stray_to_colmap
except ImportError as e:
    print(f"CRITICAL: HLoc, PyCOLMAP or Torch not installed. {e}")
    sys.exit(1)

# --- Theme & Colors ---
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

COLORS = {
    "bg_dark": "#0f0f1a",
    "bg_card": "#1a1a2e",
    "bg_card_hover": "#22223a",
    "accent_blue": "#4f6df5",
    "accent_purple": "#7c3aed",
    "accent_gradient_start": "#4f6df5",
    "accent_gradient_end": "#9333ea",
    "text_primary": "#e2e8f0",
    "text_secondary": "#94a3b8",
    "text_muted": "#64748b",
    "success": "#10b981",
    "error": "#ef4444",
    "warning": "#f59e0b",
    "console_bg": "#0a0a14",
    "console_fg": "#4ade80",
    "border": "#2a2a4a",
}


def generate_sequential_pairs(image_dir, pairs_path, overlap=10):
    """Generate sequential pairs: each image is paired with its `overlap` nearest neighbors."""
    images = sorted([f.name for f in Path(image_dir).iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff')])
    pairs = []
    for i in range(len(images)):
        for j in range(i + 1, min(i + 1 + overlap, len(images))):
            pairs.append((images[i], images[j]))
    with open(pairs_path, 'w') as f:
        f.writelines(' '.join(p) + '\n' for p in pairs)
    return len(pairs)


def count_pairs_file(pairs_path):
    try:
        with open(pairs_path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0


def _process_image_color_worker(img_path_str, source_space, dest_space, cs_in, cs_out, colorconfig_path, exr_out_dir_str):
    """
    Modular color conversion worker for parallel executors.

    Pipeline:  Read → Normalize → Linearize (EOTF) → Gamut Matrix → Encode → Write

    Args:
        source_space:  Source profile (e.g. "Linear BT.2020", "Apple Log (BT.2020)")
        dest_space:    Destination profile (e.g. "ACEScg (EXR + sRGB PNG)", "sRGB (Tone Mapped)")
        cs_in/cs_out:  OCIO colorspace names (only used when dest_space == "Custom OCIO...")
        colorconfig_path: Path to .ocio config (only for Custom OCIO)
        exr_out_dir_str:  Directory for EXR output (for dual-output modes)
    Returns: (success_bool, error_msg_or_none)
    """
    import cv2, numpy as np
    img_path = Path(img_path_str)

    # --- Custom OCIO passthrough ---
    if dest_space == "Custom OCIO..." and HAS_OCIO:
        try:
            buf = oiio.ImageBuf(str(img_path))
            if not buf.has_error:
                res = oiio.ImageBufAlgo.colorconvert(buf, buf, cs_in, cs_out, colorconfig=colorconfig_path or "")
                if res:
                    buf.write(str(img_path))
                    return True, None
        except Exception as e:
            return False, str(e)
        return False, "OCIO Error"

    # --- Native math pipeline ---
    try:
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return False, "Failed to read image"
        max_val = 65535.0 if img.dtype == np.uint16 else 255.0
        if len(img.shape) < 3 or img.shape[2] < 3:
            return False, "Unsupported channels"

        # BGR(A) → RGB, normalize to [0,1] float32
        try:
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img)
            gpu_rgb = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2RGB if img.shape[2] == 3 else cv2.COLOR_BGRA2RGB)
            img_rgb = gpu_rgb.download().astype(np.float32) / max_val
        except Exception:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB if img.shape[2] == 3 else cv2.COLOR_BGRA2RGB).astype(np.float32) / max_val

        h, w, ch = img_rgb.shape

        # Step 1: Linearize (apply EOTF based on source)
        linear_rgb = _linearize(img_rgb, source_space)

        # Step 2: Determine source and destination gamut primaries
        src_prim = _source_primaries(source_space)

        if dest_space == "ACEScg (EXR + sRGB PNG)":
            # --- ACEScg dual output: EXR (linear ACEScg) + sRGB PNG ---
            acescg_rgb = _gamut_convert(linear_rgb, src_prim, "acescg")

            # Write EXR (ACEScg linear, 32-bit float)
            if exr_out_dir_str:
                os.makedirs(exr_out_dir_str, exist_ok=True)
                out_exr = str(Path(exr_out_dir_str) / f"{img_path.stem}.exr")
            else:
                out_exr = str(img_path).rsplit('.', 1)[0] + '.exr'

            if HAS_OCIO:
                spec_exr = oiio.ImageSpec(w, h, ch, oiio.FLOAT)
                spec_exr.attribute("oiio:ColorSpace", ACESCG_OCIO_SPACE)
                buf_exr = oiio.ImageBuf(spec_exr)
                buf_exr.set_pixels(oiio.ROI(), acescg_rgb)
                buf_exr.write(out_exr)
            else:
                # Fallback: write via cv2 (limited EXR support)
                exr_bgr = cv2.cvtColor(acescg_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(out_exr, exr_bgr)

            # Write sRGB PNG (tone mapped, 16-bit) — for SfM
            srgb_linear = _gamut_convert(linear_rgb, src_prim, "srgb")
            srgb_tonemapped = _aces_tonemap(np.maximum(srgb_linear, 0.0))
            srgb_display = _srgb_oetf(srgb_tonemapped)
            srgb_16 = np.clip(srgb_display * 65535, 0, 65535).astype(np.uint16)
            srgb_bgr = cv2.cvtColor(srgb_16, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(img_path), srgb_bgr)
            return True, None

        elif dest_space == "sRGB (Tone Mapped)":
            # --- sRGB tone-mapped output (16-bit PNG) ---
            srgb_linear = _gamut_convert(linear_rgb, src_prim, "srgb")
            srgb_tonemapped = _aces_tonemap(np.maximum(srgb_linear, 0.0))
            srgb_display = _srgb_oetf(srgb_tonemapped)
            srgb_16 = np.clip(srgb_display * 65535, 0, 65535).astype(np.uint16)
            srgb_bgr = cv2.cvtColor(srgb_16, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(img_path), srgb_bgr)
            return True, None

        elif dest_space == "Linear sRGB":
            # --- Linear sRGB (16-bit PNG, no gamma) ---
            srgb_linear = _gamut_convert(linear_rgb, src_prim, "srgb")
            srgb_16 = np.clip(srgb_linear * 65535, 0, 65535).astype(np.uint16)
            srgb_bgr = cv2.cvtColor(srgb_16, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(img_path), srgb_bgr)
            return True, None

        return False, f"Unknown destination: {dest_space}"
    except Exception as e:
        return False, str(e)

def _detect_16bit_from_images(image_dir):
    """Check if existing images in a directory are 16-bit."""
    for ext in ('*.png', '*.tif', '*.tiff'):
        for img_path in Path(image_dir).glob(ext):
            try:
                with Image.open(img_path) as img:
                    if img.mode.startswith("I;16"):
                        return True

                    if img.format == 'PNG':
                        with open(img_path, 'rb') as f:
                            if f.read(8) == b'\x89PNG\r\n\x1a\n':
                                f.seek(24)
                                b = f.read(1)
                                if b and b[0] == 16:
                                    return True

                    if img.format == 'TIFF' and img.mode in ('RGB', 'RGBA'):
                        img_cv = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                        if img_cv is not None and img_cv.dtype == np.uint16:
                            return True

                    return False
            except Exception:
                pass
            break
    return False


def _ocio_needs_16bit(colorspace_name):
    """Return True if the OCIO output colorspace is linear and needs 16-bit precision."""
    if not colorspace_name:
        return False
    lower = colorspace_name.lower()
    return any(kw in lower for kw in ('linear', 'acescg', 'scene-linear', 'scene_linear', 'aces - acescg'))


class CancelledError(Exception):
    """Raised when the user cancels processing."""
    pass


class GUIProgressTqdm(_original_tqdm):
    """Tqdm replacement that reports progress to the GUI step label, throttled."""
    _gui_app = None
    _step_index = 0
    _step_name = ""

    def __init__(self, *args, **kwargs):
        kwargs['file'] = open(os.devnull, 'w')
        kwargs['disable'] = False
        super().__init__(*args, **kwargs)
        self._last_gui_update = 0
        self._devnull_fp = kwargs.get('file')

    def update(self, n=1):
        super().update(n)
        # Check for cancellation
        app = GUIProgressTqdm._gui_app
        if app and app._cancelled:
            raise CancelledError("Processing cancelled by user")
        now = time.monotonic()
        if now - self._last_gui_update < 0.25:  # max 4 updates/sec
            return
        self._last_gui_update = now
        if app and self.total:
            pct = self.n / self.total
            step_i = GUIProgressTqdm._step_index
            total_steps = len(app.STEPS)
            overall = (step_i + pct) / total_steps
            text = (
                f"Step {step_i + 1}/{total_steps} — {GUIProgressTqdm._step_name}  "
                f"({self.n}/{self.total}, {pct * 100:.0f}%)"
            )
            app.after(0, lambda: app.step_label.configure(text=text, text_color=COLORS["accent_blue"]))
            app.after(0, lambda: app.progress_bar.set(overall))

    def close(self):
        try:
            if hasattr(self, '_devnull_fp') and self._devnull_fp and not self._devnull_fp.closed:
                self._devnull_fp.close()
        except Exception:
            pass
        super().close()



class GuiLogHandler(logging.Handler):
    """Custom logging handler to redirect logs to the GUI console."""
    def __init__(self, callback):
        super().__init__()
        self.callback = callback
        self.tag = ""

    def set_tag(self, tag):
        self.tag = tag

    def emit(self, record):
        try:
            msg = self.format(record)
            if self.tag:
                msg = f"{self.tag} {msg}"
            self.callback(msg)
        except Exception:
            self.handleError(record)


class GuiStream:
    """Redirect sys.stdout/sys.stderr to the GUI console during processing."""
    def __init__(self, callback):
        self.callback = callback
        self._buf = ""

    def write(self, msg):
        if not msg:
            return
        self._buf += msg
        while '\n' in self._buf:
            line, self._buf = self._buf.split('\n', 1)
            if line.strip():
                self.callback(line)

    def flush(self):
        if self._buf.strip():
            self.callback(self._buf)
            self._buf = ""

    def isatty(self):
        return False


class SectionCard(ctk.CTkFrame):
    """A card-style section with a title, styled border, and inner padding."""
    def __init__(self, master, title, icon="", **kwargs):
        super().__init__(master, fg_color=COLORS["bg_card"], corner_radius=12, border_width=1,
                         border_color=COLORS["border"], **kwargs)
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=16, pady=(14, 4))
        ctk.CTkLabel(header, text=f"{icon}  {title}", font=ctk.CTkFont(size=15, weight="bold"),
                     text_color=COLORS["text_primary"]).pack(side="left")
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.pack(fill="x", padx=16, pady=(4, 14))


class InfoTooltip(ctk.CTkFrame):
    """A small info button that toggles a collapsible explanation panel.
    
    The info panel is created as a child of the SectionCard (grandparent)
    and packed between the header and content areas to avoid grid/pack conflicts.
    """
    def __init__(self, master, text, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self._expanded = False
        self._text = text
        self.btn = ctk.CTkButton(self, text="ⓘ", width=28, height=28, corner_radius=14,
                                  fg_color=COLORS["bg_card_hover"], hover_color=COLORS["accent_blue"],
                                  text_color=COLORS["text_secondary"], font=ctk.CTkFont(size=14),
                                  command=self._toggle)
        self.btn.pack(side="left")
        # Create info_frame as child of the SectionCard (grandparent of tooltip)
        # SectionCard uses pack for header + content, so we can safely pack between them
        self._card = master.master  # SectionCard
        self._card_content = master  # card.content (pack before this)
        self.info_frame = ctk.CTkFrame(self._card, fg_color=COLORS["bg_dark"], corner_radius=8,
                                       border_width=1, border_color=COLORS["border"])
        self.info_label = ctk.CTkLabel(self.info_frame, text=self._text,
                                        text_color=COLORS["text_secondary"],
                                        font=ctk.CTkFont(size=12), wraplength=600, justify="left")
        self.info_label.pack(padx=12, pady=10, anchor="w")

    def _toggle(self):
        if self._expanded:
            self.info_frame.pack_forget()
            self.btn.configure(fg_color=COLORS["bg_card_hover"])
        else:
            # Pack in the SectionCard, right before card.content
            self.info_frame.pack(fill="x", padx=16, pady=(0, 8), before=self._card_content)
            self.btn.configure(fg_color=COLORS["accent_blue"])
        self._expanded = not self._expanded

    def pack_info_after(self, widget):
        """No-op, kept for backwards compatibility."""
        pass


class SearchableSelectionWindow(ctk.CTkToplevel):
    def __init__(self, master, title, options, current_selection, callback):
        super().__init__(master)
        self.title(title)
        self.geometry("400x500")
        self.minsize(300, 400)
        self.configure(fg_color=COLORS["bg_dark"])
        
        # Make it modal
        self.transient(master)
        
        # Defer grab_set to avoid "window not viewable" TclError
        self.after(100, self.grab_set)
        
        self.options = options
        self.callback = callback
        
        # Search Entry
        self.search_var = ctk.StringVar()
        self.search_var.trace_add("write", self._filter_options)
        
        self.search_entry = ctk.CTkEntry(self, textvariable=self.search_var, placeholder_text="Search...",
                                         fg_color=COLORS["bg_card"], border_color=COLORS["border"],
                                         text_color=COLORS["text_primary"], height=36)
        self.search_entry.pack(fill="x", padx=16, pady=(16, 8))
        self.search_entry.focus_set()
        
        # Scrollable List
        self.scroll_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.scroll_frame.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        
        self.buttons = []
        self._populate_list(self.options, current_selection)

    def _filter_options(self, *args):
        query = self.search_var.get().lower()
        if not query:
            filtered = self.options
        else:
            filtered = [opt for opt in self.options if query in opt.lower()]
        self._populate_list(filtered)

    def _populate_list(self, items, current_selection=None):
        for btn in self.buttons:
            btn.destroy()
        self.buttons.clear()
        
        for item in items:
            is_selected = item == current_selection
            fg_color = COLORS["accent_blue"] if is_selected else COLORS["bg_card"]
            hover_color = COLORS["accent_purple"] if is_selected else COLORS["bg_card_hover"]
            
            btn = ctk.CTkButton(self.scroll_frame, text=item, anchor="w", 
                                fg_color=fg_color, hover_color=hover_color,
                                text_color=COLORS["text_primary"], corner_radius=6,
                                command=lambda val=item: self._select_item(val))
            btn.pack(fill="x", pady=2)
            self.buttons.append(btn)

    def _select_item(self, value):
        self.callback(value)
        self.destroy()


class SfMApp(ctk.CTk):
    STEPS = ["FFmpeg/Prep", "OCIO", "Features", "Pairs", "Matching", "SfM"]
    STEPS_STRAY_A = ["Rescan→COLMAP", "OCIO", "Features", "Pairs", "Matching", "Triangulation"]
    STEPS_STRAY_B = ["Rescan→COLMAP", "OCIO", "Features", "Pairs", "Matching", "SfM"]

    def __init__(self):
        super().__init__()
        self.title("ReMap — Gaussian Splatting Preparation Pipeline")
        self.geometry("1000x860")
        self.minsize(640, 600)
        self.configure(fg_color=COLORS["bg_dark"])

        self._cancelled = False
        self._processing = False
        self._server_process = None  # subprocess.Popen handle for the API server

        # --- Variables ---
        self.video_paths = []  # List of Path objects for multi-video
        self.stray_paths = []  # List of Path objects for multi-Rescan datasets
        self.video_path = ctk.StringVar()  # Display variable for the entry field
        self.output_path = ctk.StringVar()
        self.fps_extract = ctk.DoubleVar(value=4.0)
        self.force_16bit = ctk.BooleanVar(value=False)
        self.camera_model = ctk.StringVar(value="OPENCV")
        self.feature_type = ctk.StringVar(value="superpoint_aachen")
        self.matcher_type = ctk.StringVar(value="superpoint+lightglue")
        self.max_keypoints = ctk.StringVar(value="4096")
        self.pairing_mode = ctk.StringVar(value="Sequential (Video)")
        self.fps_label_var = ctk.StringVar(value="4.0 FPS")
        self.mapper_type = ctk.StringVar(value="COLMAP")
        # Check for GLOMAP
        self.has_glomap = shutil.which("glomap") is not None
        if self.has_glomap:
            self.mapper_type.set("GLOMAP")
        
        self.input_mode = ctk.StringVar(value="Video (.mp4, .mov)")

        # Rescan specific variables
        self.stray_approach = ctk.StringVar(value="full_sfm")
        self.stray_confidence = ctk.IntVar(value=2)
        self.stray_depth_subsample = ctk.IntVar(value=2)
        self.stray_gen_pointcloud = ctk.BooleanVar(value=True)

        # Color conversion variables (Source/Destination system)
        self.color_enabled = ctk.BooleanVar(value=False)
        self.color_source = ctk.StringVar(value="Auto-detect")
        self.color_dest = ctk.StringVar(value="ACEScg (EXR + sRGB PNG)")
        self.detected_color_profile = ctk.StringVar(value="")  # Filled by ffprobe
        # OCIO (kept for "Custom OCIO..." destination)
        self.ocio_path = ctk.StringVar(value=os.environ.get("OCIO", ""))
        self.ocio_in_cs = ctk.StringVar(value="")
        self.ocio_out_cs = ctk.StringVar(value="")
        self.use_acescg_exr = ctk.BooleanVar(value=True)
        self.ocio_spaces = []
        self.has_ocio_lib = HAS_OCIO
        # Input probing data (for frame count estimates)
        self._video_infos = []    # [{path, duration, native_fps, total_frames}]
        self._rescan_infos = []   # [{path, total_frames}]
        self.frame_estimate_var = ctk.StringVar(value="")

        # Dashboard stats (updated during processing)
        self._dash_stats = {"images": 0, "features": 0, "matches": 0, "points3d": 0}

        # Worker configuration
        default_workers = multiprocessing.cpu_count()
        self.num_workers = ctk.IntVar(value=default_workers)
        self.workers_label_var = ctk.StringVar(value=f"{default_workers} Threads")

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _on_closing(self):
        """Terminate the server subprocess (if running) before closing the window."""
        if self._server_process is not None:
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._server_process.kill()
        if hasattr(self, "_server_log_fp") and self._server_log_fp:
            try:
                self._server_log_fp.close()
            except Exception:
                pass
        self.destroy()

    # -------------------------------------------------------------------------
    #  UI BUILDING
    # -------------------------------------------------------------------------
    def _build_ui(self):
        # --- Title Bar ---
        title_frame = ctk.CTkFrame(self, fg_color="transparent")
        title_frame.pack(fill="x", padx=24, pady=(18, 6))

        ctk.CTkLabel(title_frame, text="◆", font=ctk.CTkFont(size=28),
                     text_color=COLORS["accent_blue"]).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(title_frame, text="Re", font=ctk.CTkFont(size=24, weight="bold"),
                     text_color=COLORS["text_primary"]).pack(side="left")
        ctk.CTkLabel(title_frame, text="Map", font=ctk.CTkFont(size=24),
                     text_color=COLORS["accent_purple"]).pack(side="left", padx=(4, 0))
        ctk.CTkLabel(title_frame, text="Gaussian Splatting Preparation Pipeline",
                     font=ctk.CTkFont(size=12), text_color=COLORS["text_muted"]).pack(side="left", padx=(16, 0))

        # --- Scrollable main area ---
        main_scroll = ctk.CTkScrollableFrame(self, fg_color="transparent", corner_radius=0)
        main_scroll.pack(fill="both", expand=True, padx=20, pady=(6, 0))

        # Fix mouse wheel scrolling on Linux (X11 uses Button-4 / Button-5)
        def _on_mousewheel(event):
            canvas = main_scroll._parent_canvas
            if event.num == 4:
                canvas.yview_scroll(-3, "units")
            elif event.num == 5:
                canvas.yview_scroll(3, "units")
        main_scroll.bind_all("<Button-4>", _on_mousewheel)
        main_scroll.bind_all("<Button-5>", _on_mousewheel)

        # --- 1. Input / Output ---
        card_io = SectionCard(main_scroll, "Input / Output", icon="📁")
        card_io.pack(fill="x", pady=(0, 10))
        self._card_io_ref = card_io  # Reference for dynamic card positioning

        # Info tooltip for I/O
        io_tip = InfoTooltip(card_io.content,
            "Select your input source and output directory.\n"
            "The output will contain images/, sparse/0/ and hloc_outputs/ folders\n"
            "compatible with 3DGS training tools (e.g. gsplat, nerfstudio).")
        io_tip.grid(row=0, column=2, sticky="e", pady=(0, 6))
        io_tip.pack_info_after(card_io.content)

        # Input Mode Selection
        mode_row = ctk.CTkFrame(card_io.content, fg_color="transparent")
        mode_row.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        ctk.CTkLabel(mode_row, text="Input Mode", text_color=COLORS["text_secondary"], 
                     font=ctk.CTkFont(size=13)).pack(side="left", padx=(0, 10))
        
        self.seg_input_mode = ctk.CTkSegmentedButton(
            mode_row, values=["Video (.mp4, .mov)", "Image Folder", "Rescan (LiDAR)"],
            variable=self.input_mode, command=self._on_input_mode_change,
            selected_color=COLORS["accent_blue"],
            selected_hover_color=COLORS["accent_blue"],
            unselected_color=COLORS["bg_dark"],
            text_color=COLORS["text_primary"]
        )
        self.seg_input_mode.pack(side="left", fill="x", expand=True)
        self.seg_input_mode.set("Video (.mp4, .mov)")

        self.input_label_var = ctk.StringVar(value="Video File(s)")
        self._file_row(card_io.content, self.input_label_var, self.video_path, self._browse_input, row=1)
        self._file_row(card_io.content, "Output Folder", self.output_path, self._browse_output, row=2)

        # --- FPS Global Row ---
        self.fps_frame = ctk.CTkFrame(card_io.content, fg_color="transparent")
        self.fps_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        
        lbl_row = ctk.CTkFrame(self.fps_frame, fg_color="transparent")
        lbl_row.pack(fill="x")
        ctk.CTkLabel(lbl_row, text="Extraction FPS", text_color=COLORS["text_secondary"], font=ctk.CTkFont(size=13)).pack(side="left")
        self.fps_label = ctk.CTkLabel(lbl_row, textvariable=self.fps_label_var, text_color=COLORS["accent_blue"], font=ctk.CTkFont(size=13, weight="bold"))
        self.fps_label.pack(side="right", padx=(0, 4))
        
        self.slider_fps = ctk.CTkSlider(self.fps_frame, from_=0.5, to=30, number_of_steps=59,
                               variable=self.fps_extract, command=self._on_fps_change,
                               progress_color=COLORS["accent_blue"],
                               button_color=COLORS["accent_purple"],
                               fg_color=COLORS["border"])
        self.slider_fps.pack(fill="x", pady=(4, 2))

        # --- 2. Video Extraction ---
        self.card_vid = SectionCard(main_scroll, "Video Extraction (FFmpeg)", icon="🎬")
        self.card_vid.pack(fill="x", pady=(0, 10))

        # Info tooltip for Video Extraction
        vid_tip = InfoTooltip(self.card_vid.content,
            "Controls how frames are extracted from video files.\n\n"
            "FPS: Higher FPS = more images = better coverage but slower processing.\n"
            "• Walkthrough / indoor: 2–4 FPS\n"
            "• Drone / FPV: 4–8 FPS\n"
            "• Small object / turntable: 8–15 FPS\n\n"
            "16-bit output is auto-detected from OCIO settings, or use Force 16-bit.\n"
            "Extraction is automatically skipped if images already exist in the output.")
        vid_tip.pack(anchor="e", pady=(0, 4))
        vid_tip.pack_info_after(self.card_vid.content)

        # Force 16-bit Row
        bit_row = ctk.CTkFrame(self.card_vid.content, fg_color="transparent")
        bit_row.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(bit_row, text="Force 16-bit PNG output", 
                     text_color=COLORS["text_secondary"], font=ctk.CTkFont(size=13)).pack(side="left")
        ctk.CTkSwitch(bit_row, variable=self.force_16bit, text="", width=46,
                      progress_color=COLORS["accent_purple"],
                      button_color=COLORS["text_primary"]).pack(side="right")

        # Frame estimate label (video)
        self.vid_frame_estimate_label = ctk.CTkLabel(
            self.card_vid.content, textvariable=self.frame_estimate_var,
            text_color=COLORS["accent_purple"], font=ctk.CTkFont(size=12, weight="bold"))
        self.vid_frame_estimate_label.pack(anchor="w", pady=(4, 0))

        # --- 2b. Rescan Settings (hidden by default) ---
        self.card_stray = SectionCard(main_scroll, "Rescan (LiDAR)", icon="📱")
        # Hidden by default — shown when Rescan mode is selected

        # Info tooltip for Rescan
        stray_tip = InfoTooltip(self.card_stray.content,
            "Settings for LiDAR scan datasets (Stray Scanner / Rescan app).\n\n"
            "Approach B (Full SfM): Recalculates camera poses from scratch.\n"
            "  → Best quality, recommended for most cases.\n"
            "Approach A (ARKit Poses): Uses device odometry directly.\n"
            "  → Faster but potentially less accurate.\n\n"
            "Subsampling: Use every Nth frame. Higher = fewer frames = faster.\n"
            "LiDAR Confidence: Filter out low-confidence depth measurements.")
        stray_tip.pack(anchor="e", pady=(0, 4))
        stray_tip.pack_info_after(self.card_stray.content)

        # Approach selection
        approach_row = ctk.CTkFrame(self.card_stray.content, fg_color="transparent")
        approach_row.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(approach_row, text="Approach", text_color=COLORS["text_secondary"],
                     font=ctk.CTkFont(size=13)).pack(side="left", padx=(0, 10))
        self.seg_stray_approach = ctk.CTkSegmentedButton(
            approach_row, values=["B — Full SfM (default)", "A — ARKit Poses"],
            variable=self.stray_approach,
            selected_color=COLORS["accent_purple"],
            selected_hover_color=COLORS["accent_purple"],
            unselected_color=COLORS["bg_dark"],
            text_color=COLORS["text_primary"],
            command=self._on_stray_approach_change
        )
        self.seg_stray_approach.pack(side="left", fill="x", expand=True)
        self.seg_stray_approach.set("B — Full SfM (default)")

        # Frame estimate label (Rescan)
        self.rescan_frame_estimate_label = ctk.CTkLabel(
            self.card_stray.content, text="",
            text_color=COLORS["accent_purple"], font=ctk.CTkFont(size=12, weight="bold"))
        self.rescan_frame_estimate_label.pack(anchor="w", pady=(0, 6))

        # Confidence threshold
        conf_row = ctk.CTkFrame(self.card_stray.content, fg_color="transparent")
        conf_row.pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(conf_row, text="Min. LiDAR Confidence", text_color=COLORS["text_secondary"],
                     font=ctk.CTkFont(size=13)).pack(side="left")
        ctk.CTkComboBox(conf_row, variable=self.stray_confidence,
                        values=["0 (all)", "1 (medium)", "2 (high)"], state="readonly",
                        width=140, dropdown_fg_color=COLORS["bg_card"],
                        button_color=COLORS["accent_blue"],
                        border_color=COLORS["border"],
                        fg_color=COLORS["bg_dark"],
                        command=self._on_stray_confidence_change).pack(side="right")
        # Set default display
        self.stray_confidence.set(2)

        # Point cloud toggle
        pc_row = ctk.CTkFrame(self.card_stray.content, fg_color="transparent")
        pc_row.pack(fill="x", pady=(4, 0))
        ctk.CTkLabel(pc_row, text="Generate LiDAR point cloud",
                     text_color=COLORS["text_secondary"], font=ctk.CTkFont(size=13)).pack(side="left")
        ctk.CTkSwitch(pc_row, variable=self.stray_gen_pointcloud, text="", width=46,
                      progress_color=COLORS["accent_blue"],
                      button_color=COLORS["text_primary"]).pack(side="right")

        exr_row = ctk.CTkFrame(self.card_stray.content, fg_color="transparent")
        exr_row.pack(fill="x", pady=(4, 0))
        ctk.CTkLabel(exr_row, text="Generate / patch ACEScg EXR output",
                     text_color=COLORS["text_secondary"], font=ctk.CTkFont(size=13)).pack(side="left")
        ctk.CTkSwitch(exr_row, variable=self.use_acescg_exr, text="", width=46,
                      progress_color=COLORS["accent_purple"],
                      button_color=COLORS["text_primary"]).pack(side="right")

        ctk.CTkLabel(self.card_stray.content, text="Full SfM: COLMAP recalculates poses. ARKit Poses: uses device odometry directly.",
                     text_color=COLORS["text_muted"], font=ctk.CTkFont(size=11), wraplength=600).pack(anchor="w", pady=(4, 0))


        # --- 3. SfM Pipeline ---
        card_sfm = SectionCard(main_scroll, "SfM Pipeline", icon="🔬")
        card_sfm.pack(fill="x", pady=(0, 10))

        # Info tooltip for SfM Pipeline
        sfm_tip = InfoTooltip(card_sfm.content,
            "Feature detection and matching settings.\n\n"
            "Features: SuperPoint is recommended for most cases.\n"
            "  DISK or ALIKED for challenging lighting/textures.\n"
            "Matcher: LightGlue is fastest. SuperGlue more robust for difficult scenes.\n"
            "Max Keypoints: 4096 is a good default. Increase to 8192 for complex scenes.\n"
            "Pairing: Sequential for video/scan. Exhaustive for small unordered sets (<200).\n"
            "  Exhaustive is more precise for spatialization but much slower.\n\n"
            "SfM Engine: GLOMAP is faster (GPU). COLMAP is the robust default.")
        sfm_tip.grid(row=0, column=3, sticky="e", pady=(0, 4))
        sfm_tip.pack_info_after(card_sfm.content)

        grid = card_sfm.content
        grid.columnconfigure(1, weight=1)
        grid.columnconfigure(3, weight=1)
        grid.columnconfigure(0, minsize=120)
        grid.columnconfigure(2, minsize=120)

        features_list = ["superpoint_aachen", "superpoint_max", "disk", "aliked-n16", "sift"]
        matchers_list = ["superpoint+lightglue", "superglue", "disk+lightglue", "adalam", "loma_b", "loma_g"]
        pairing_list = ["Sequential (Video)", "Exhaustive (Small dataset < 200)"]

        self._combo_row(grid, "Features", self.feature_type, features_list, row=1, col=0)
        self._entry_row(grid, "Max Keypoints", self.max_keypoints, row=1, col=2, width=100)
        self._combo_row(grid, "Matcher", self.matcher_type, matchers_list, row=2, col=0)
        self._combo_row(grid, "Pair Strategy", self.pairing_mode, pairing_list, row=2, col=2)

        # --- Mapper Select (Row 3) ---
        ctk.CTkLabel(grid, text="SfM Engine", text_color=COLORS["text_secondary"],
                     font=ctk.CTkFont(size=12, weight="bold")).grid(row=3, column=0, sticky="w", pady=(10, 0))
        
        map_values = ["GLOMAP", "COLMAP"]
        if not self.has_glomap:
            map_values = ["GLOMAP (Not installed)", "COLMAP"]
            
        self.seg_map = ctk.CTkSegmentedButton(grid, values=map_values, variable=self.mapper_type,
                                         selected_color=COLORS["accent_purple"],
                                         selected_hover_color=COLORS["accent_purple"],
                                         unselected_color=COLORS["bg_card"],
                                         text_color=COLORS["text_primary"],
                                         command=self._on_mapper_change)
        self.seg_map.grid(row=3, column=1, columnspan=3, sticky="ew", pady=(10, 0), padx=(5, 0))
        
        if not self.has_glomap:
            self.seg_map.set("COLMAP")
            self.seg_map.configure(state="disabled")

        # --- Color Management (Source / Destination) ---
        card_color = SectionCard(main_scroll, "Color Management", icon="🎨")
        card_color.pack(fill="x", pady=(0, 10))

        # Info tooltip
        color_tip = InfoTooltip(card_color.content,
            "Convert extracted frames between color spaces.\n\n"
            "Source: The color profile of the input video.\n"
            "  • Auto-detect reads ffprobe metadata (recommended)\n"
            "  • Manual override for known sources\n\n"
            "Destination: Target color space for processing.\n"
            "  • ACEScg: Industry-standard scene-linear (EXR + sRGB PNG)\n"
            "  • Linear sRGB: Scene-linear Rec.709\n"
            "  • sRGB (Tone Mapped): Display-ready with ACES filmic curve\n"
            "  • Custom OCIO: Use your own .ocio config\n\n"
            "16-bit output is automatic for linear/log sources.")
        color_tip.pack(anchor="e", pady=(0, 4))
        color_tip.pack_info_after(card_color.content)

        # Enable toggle row
        color_toggle_row = ctk.CTkFrame(card_color.content, fg_color="transparent")
        color_toggle_row.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(color_toggle_row, text="Color Conversion",
                    text_color=COLORS["text_secondary"], font=ctk.CTkFont(size=13)).pack(side="left")
        ctk.CTkSwitch(color_toggle_row, variable=self.color_enabled, text="", width=46,
                      progress_color=COLORS["accent_purple"],
                      button_color=COLORS["text_primary"],
                      command=self._on_color_enabled_change).pack(side="right")

        # Source / Destination frame (shown when enabled)
        self.color_sd_frame = ctk.CTkFrame(card_color.content, fg_color="transparent")

        sd_row = ctk.CTkFrame(self.color_sd_frame, fg_color="transparent")
        sd_row.pack(fill="x", pady=(0, 4))
        sd_row.columnconfigure(1, weight=1)
        sd_row.columnconfigure(3, weight=1)

        ctk.CTkLabel(sd_row, text="Source", text_color=COLORS["text_secondary"],
                     font=ctk.CTkFont(size=13)).grid(row=0, column=0, padx=(0, 8), sticky="w")
        self.combo_color_source = ctk.CTkComboBox(
            sd_row, variable=self.color_source, values=COLOR_SOURCES, state="readonly",
            dropdown_fg_color=COLORS["bg_card"], button_color=COLORS["accent_blue"],
            border_color=COLORS["border"], fg_color=COLORS["bg_dark"], width=200)
        self.combo_color_source.grid(row=0, column=1, sticky="w", padx=(0, 20))

        ctk.CTkLabel(sd_row, text="Destination", text_color=COLORS["text_secondary"],
                     font=ctk.CTkFont(size=13)).grid(row=0, column=2, padx=(0, 8), sticky="w")
        self.combo_color_dest = ctk.CTkComboBox(
            sd_row, variable=self.color_dest, values=COLOR_DESTINATIONS, state="readonly",
            dropdown_fg_color=COLORS["bg_card"], button_color=COLORS["accent_blue"],
            border_color=COLORS["border"], fg_color=COLORS["bg_dark"], width=220,
            command=self._on_color_dest_change)
        self.combo_color_dest.grid(row=0, column=3, sticky="w")

        # Detected profile label
        self.detected_profile_label = ctk.CTkLabel(
            self.color_sd_frame, text="", text_color=COLORS["accent_purple"],
            font=ctk.CTkFont(size=12, weight="bold"))
        self.detected_profile_label.pack(anchor="w", pady=(4, 0))

        # OCIO options (hidden by default, shown only when dest = "Custom OCIO...")
        self.ocio_options_frame = ctk.CTkFrame(self.color_sd_frame, fg_color="transparent")

        self._file_row(self.ocio_options_frame, "Config OCIO", self.ocio_path, self._browse_ocio, row=0)

        for child in self.ocio_options_frame.winfo_children():
            if isinstance(child, ctk.CTkEntry):
                child.bind("<FocusOut>", lambda e: self._update_ocio_dropdowns())
                child.bind("<Return>", lambda e: self._update_ocio_dropdowns())

        self.ocio_btn_in = ctk.CTkButton(self.ocio_options_frame, textvariable=self.ocio_in_cs,
                                         fg_color=COLORS["bg_card"], hover_color=COLORS["bg_card_hover"],
                                         border_color=COLORS["border"], border_width=1,
                                         text_color=COLORS["text_primary"], command=self._open_ocio_in_selection, width=180)
        self.ocio_btn_out = ctk.CTkButton(self.ocio_options_frame, textvariable=self.ocio_out_cs,
                                          fg_color=COLORS["bg_card"], hover_color=COLORS["bg_card_hover"],
                                          border_color=COLORS["border"], border_width=1,
                                          text_color=COLORS["text_primary"], command=self._open_ocio_out_selection, width=180)

        cs_row = ctk.CTkFrame(self.ocio_options_frame, fg_color="transparent")
        cs_row.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        ctk.CTkLabel(cs_row, text="Input Space", text_color=COLORS["text_secondary"], font=ctk.CTkFont(size=13)).grid(row=0, column=0, padx=(0, 6), sticky="w")
        self.ocio_btn_in.grid(row=0, column=1, sticky="w")

        ctk.CTkLabel(cs_row, text="Output Space", text_color=COLORS["text_secondary"], font=ctk.CTkFont(size=13)).grid(row=0, column=2, padx=(20, 6), sticky="w")
        self.ocio_btn_out.grid(row=0, column=3, sticky="w")

        # Initially hidden (color disabled by default)
        # self.color_sd_frame is packed/forgotten by _on_color_enabled_change

        # --- 4. Camera Model ---
        card_cam = SectionCard(main_scroll, "Camera Model (COLMAP)", icon="📷")
        card_cam.pack(fill="x", pady=(0, 10))

        # Info tooltip for Camera Model
        cam_tip = InfoTooltip(card_cam.content,
            "COLMAP camera distortion model.\n\n"
            "OPENCV: Best for action cameras / GoPro / FPV with lens distortion.\n"
            "  Recommended default for most cases.\n"
            "PINHOLE: For rectilinear lenses with no distortion.\n"
            "SIMPLE_RADIAL: Simpler distortion model, faster convergence.\n"
            "OPENCV_FISHEYE: For ultra-wide fisheye lenses.\n\n"
            "Threads: Number of CPU threads for parallel processing.")
        cam_tip.pack(anchor="e", pady=(0, 4))
        cam_tip.pack_info_after(card_cam.content)

        cam_row = ctk.CTkFrame(card_cam.content, fg_color="transparent")
        cam_row.pack(fill="x")

        cams_list = ["OPENCV", "PINHOLE", "SIMPLE_RADIAL", "OPENCV_FISHEYE"]
        ctk.CTkComboBox(cam_row, variable=self.camera_model, values=cams_list, state="readonly",
                        width=220, dropdown_fg_color=COLORS["bg_card"],
                        button_color=COLORS["accent_blue"],
                        border_color=COLORS["border"],
                        fg_color=COLORS["bg_dark"]).pack(side="left")

        # Workers Slider
        workers_frame = ctk.CTkFrame(card_cam.content, fg_color="transparent")
        workers_frame.pack(fill="x", pady=(8, 0))
        
        lbl_frame = ctk.CTkFrame(workers_frame, fg_color="transparent")
        lbl_frame.pack(fill="x")
        ctk.CTkLabel(lbl_frame, text="Performance (Threads)", text_color=COLORS["text_secondary"],
                     font=ctk.CTkFont(size=12)).pack(side="left")
        ctk.CTkLabel(lbl_frame, textvariable=self.workers_label_var,
                     text_color=COLORS["accent_blue"], font=ctk.CTkFont(size=12, weight="bold")).pack(side="right")
        
        max_cpu = multiprocessing.cpu_count()
        slider_workers = ctk.CTkSlider(workers_frame, from_=1, to=max_cpu, number_of_steps=max_cpu-1,
                                       variable=self.num_workers, command=self._on_workers_change,
                                       progress_color=COLORS["accent_blue"],
                                       button_color=COLORS["accent_purple"],
                                       fg_color=COLORS["border"], height=16)
        slider_workers.pack(fill="x", pady=(2, 0))

        # --- 5. API Server ---
        card_server = SectionCard(main_scroll, "API Server (ReScan Remote)", icon="🌐")
        card_server.pack(fill="x", pady=(0, 10))

        server_tip = InfoTooltip(card_server.content,
            "Start a local REST API server so the ReScan iOS app\n"
            "can send datasets directly to this machine over the network.\n\n"
            "When enabled, ReScan uploads are received, processed\n"
            "automatically, and results are available for download.\n\n"
            "The API key is required for authentication (Bearer token).\n"
            "You can type your own key or click 🔄 Generate for a new one.\n"
            "The key is locked while the server is running.\n"
            "Use 🔌 Check to test connectivity (no key required).\n"
            "See API_DOCUMENTATION.md for the full endpoint reference.")
        server_tip.pack(anchor="e", pady=(0, 4))
        server_tip.pack_info_after(card_server.content)

        srv_row1 = ctk.CTkFrame(card_server.content, fg_color="transparent")
        srv_row1.pack(fill="x", pady=(0, 4))

        self.server_running = False
        self._server_process = None
        self._server_api_key = None

        self.server_port = ctk.IntVar(value=5000)
        ctk.CTkLabel(srv_row1, text="Port", text_color=COLORS["text_secondary"],
                     font=ctk.CTkFont(size=13)).pack(side="left", padx=(0, 6))
        ctk.CTkEntry(srv_row1, textvariable=self.server_port, width=80,
                     fg_color=COLORS["bg_dark"], border_color=COLORS["border"],
                     text_color=COLORS["text_primary"]).pack(side="left", padx=(0, 12))

        self.btn_server_toggle = ctk.CTkButton(
            srv_row1, text="▶  Start Server", width=160, height=32, corner_radius=8,
            fg_color=COLORS["accent_blue"], hover_color=COLORS["accent_purple"],
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._toggle_server,
        )
        self.btn_server_toggle.pack(side="left", padx=(0, 12))

        self.btn_server_check = ctk.CTkButton(
            srv_row1, text="🔌 Check", width=90, height=32, corner_radius=8,
            fg_color=COLORS["bg_card"], hover_color=COLORS["bg_card_hover"],
            text_color=COLORS["text_secondary"], font=ctk.CTkFont(size=12),
            command=self._check_server_connection,
        )
        self.btn_server_check.pack(side="left", padx=(0, 12))

        self.server_status_label = ctk.CTkLabel(
            srv_row1, text="● Stopped", text_color=COLORS["text_muted"],
            font=ctk.CTkFont(size=12))
        self.server_status_label.pack(side="left")

        # API Key row — editable before server starts
        srv_row2 = ctk.CTkFrame(card_server.content, fg_color="transparent")
        srv_row2.pack(fill="x", pady=(2, 0))
        ctk.CTkLabel(srv_row2, text="API Key", text_color=COLORS["text_secondary"],
                     font=ctk.CTkFont(size=12)).pack(side="left", padx=(0, 6))
        self.server_key_var = ctk.StringVar(value=secrets.token_urlsafe(32))
        self.server_key_entry = ctk.CTkEntry(
            srv_row2, textvariable=self.server_key_var, width=380,
            fg_color=COLORS["bg_dark"], border_color=COLORS["border"],
            text_color=COLORS["text_primary"], state="normal",
            font=ctk.CTkFont(family="Consolas, monospace", size=11))
        self.server_key_entry.pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            srv_row2, text="🔄 Generate", width=100, height=28, corner_radius=6,
            fg_color=COLORS["bg_card"], hover_color=COLORS["bg_card_hover"],
            text_color=COLORS["text_secondary"], font=ctk.CTkFont(size=11),
            command=self._generate_api_key,
        ).pack(side="left")

        # --- Progress Bar ---
        progress_frame = ctk.CTkFrame(main_scroll, fg_color="transparent")
        progress_frame.pack(fill="x", pady=(4, 2))

        self.step_label = ctk.CTkLabel(progress_frame, text="Waiting...",
                                        text_color=COLORS["text_muted"], font=ctk.CTkFont(size=12))
        self.step_label.pack(anchor="w")

        self.progress_bar = ctk.CTkProgressBar(progress_frame, height=6,
                                                progress_color=COLORS["accent_blue"],
                                                fg_color=COLORS["border"], corner_radius=3)
        self.progress_bar.pack(fill="x", pady=(4, 0))
        self.progress_bar.set(0)

        # --- Buttons Row (Launch + Cancel) ---
        btn_frame = ctk.CTkFrame(main_scroll, fg_color="transparent")
        btn_frame.pack(fill="x", pady=(10, 8))
        btn_frame.columnconfigure(0, weight=3)
        btn_frame.columnconfigure(1, weight=1)

        self.btn_run = ctk.CTkButton(
            btn_frame, text="⚡  START PROCESSING",
            font=ctk.CTkFont(size=16, weight="bold"), height=48, corner_radius=10,
            fg_color=COLORS["accent_blue"], hover_color=COLORS["accent_purple"],
            command=self._start_thread
        )
        self.btn_run.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.btn_cancel = ctk.CTkButton(
            btn_frame, text="✕  Cancel",
            font=ctk.CTkFont(size=14, weight="bold"), height=48, corner_radius=10,
            fg_color=COLORS["error"], hover_color="#dc2626",
            state="disabled", command=self._cancel_process
        )
        self.btn_cancel.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        # --- Console ---
        console_header = ctk.CTkFrame(main_scroll, fg_color="transparent")
        console_header.pack(fill="x")
        ctk.CTkLabel(console_header, text="Console", text_color=COLORS["text_muted"],
                     font=ctk.CTkFont(size=12)).pack(side="left")
        self._console_fullscreen = False
        self.btn_console_fs = ctk.CTkButton(
            console_header, text="⛶  Fullscreen", width=110, height=24, corner_radius=6,
            fg_color=COLORS["bg_card"], hover_color=COLORS["bg_card_hover"],
            text_color=COLORS["text_secondary"], font=ctk.CTkFont(size=11),
            command=self._toggle_console_fullscreen)
        self.btn_console_fs.pack(side="right")
        ctk.CTkButton(
            console_header, text="🗑  Clear", width=80, height=24, corner_radius=6,
            fg_color=COLORS["bg_card"], hover_color=COLORS["bg_card_hover"],
            text_color=COLORS["text_secondary"], font=ctk.CTkFont(size=11),
            command=self._clear_console).pack(side="right", padx=(0, 6))

        self.console = ctk.CTkTextbox(main_scroll, height=180,
                                       fg_color=COLORS["console_bg"],
                                       text_color=COLORS["console_fg"],
                                       font=ctk.CTkFont(family="Consolas, monospace", size=12),
                                       corner_radius=8, border_width=1,
                                       border_color=COLORS["border"],
                                       state="disabled")
        self.console.pack(fill="both", expand=True, pady=(4, 10))

    # -------------------------------------------------------------------------
    #  HELPER WIDGETS
    # -------------------------------------------------------------------------
    def _file_row(self, parent, label_var_or_str, var, browse_cmd, row):
        parent.columnconfigure(1, weight=1)
        if isinstance(label_var_or_str, ctk.StringVar):
            ctk.CTkLabel(parent, textvariable=label_var_or_str, text_color=COLORS["text_secondary"],
                         font=ctk.CTkFont(size=13)).grid(row=row, column=0, sticky="w", pady=4, padx=(0, 10))
        else:
             ctk.CTkLabel(parent, text=label_var_or_str, text_color=COLORS["text_secondary"],
                         font=ctk.CTkFont(size=13)).grid(row=row, column=0, sticky="w", pady=4, padx=(0, 10))
            
        entry = ctk.CTkEntry(parent, textvariable=var, fg_color=COLORS["bg_dark"],
                             border_color=COLORS["border"], text_color=COLORS["text_primary"])
        entry.grid(row=row, column=1, sticky="ew", pady=4)
        ctk.CTkButton(parent, text="Browse", width=90, height=30, corner_radius=6,
                      fg_color=COLORS["accent_blue"], hover_color=COLORS["accent_purple"],
                      font=ctk.CTkFont(size=12), command=browse_cmd).grid(row=row, column=2, padx=(8, 0), pady=4)

    def _combo_row(self, parent, label, var, values, row, col):
        ctk.CTkLabel(parent, text=label, text_color=COLORS["text_secondary"],
                     font=ctk.CTkFont(size=13)).grid(row=row, column=col, sticky="w", pady=6, padx=(0, 8))
        ctk.CTkComboBox(parent, variable=var, values=values, state="readonly",
                        dropdown_fg_color=COLORS["bg_card"], button_color=COLORS["accent_blue"],
                        border_color=COLORS["border"],
                        fg_color=COLORS["bg_dark"]).grid(row=row, column=col + 1, sticky="ew", pady=6, padx=(0, 20))

    def _entry_row(self, parent, label, var, row, col, width=100):
        ctk.CTkLabel(parent, text=label, text_color=COLORS["text_secondary"],
                     font=ctk.CTkFont(size=13)).grid(row=row, column=col, sticky="w", pady=6, padx=(0, 8))
        ctk.CTkEntry(parent, textvariable=var, width=width, fg_color=COLORS["bg_dark"],
                     border_color=COLORS["border"],
                     text_color=COLORS["text_primary"]).grid(row=row, column=col + 1, sticky="ew", pady=6)

    # -------------------------------------------------------------------------
    #  CONSOLE FULLSCREEN
    # -------------------------------------------------------------------------
    def _toggle_console_fullscreen(self):
        if self._console_fullscreen:
            return  # Already open, ignore

        self._console_fullscreen = True
        win = ctk.CTkToplevel(self)
        win.title("ReMap — Console")
        win.geometry("1200x700")
        win.configure(fg_color=COLORS["console_bg"])
        win.transient(self)
        win.after(100, win.grab_set)

        fs_console = ctk.CTkTextbox(
            win, fg_color=COLORS["console_bg"], text_color=COLORS["console_fg"],
            font=ctk.CTkFont(family="Consolas, monospace", size=13),
            corner_radius=0, border_width=0, state="disabled")
        fs_console.pack(fill="both", expand=True, padx=8, pady=8)

        # Copy existing console content
        existing_text = self.console.get("1.0", "end").strip()
        if existing_text:
            fs_console.configure(state="normal")
            fs_console.insert("end", existing_text + "\n")
            fs_console.see("end")
            fs_console.configure(state="disabled")

        # Mirror new log messages to this window
        original_log_safe = self._log_safe

        def mirrored_log_safe(msg):
            original_log_safe(msg)
            try:
                if win.winfo_exists():
                    fs_console.configure(state="normal")
                    fs_console.insert("end", msg + "\n")
                    fs_console.see("end")
                    fs_console.configure(state="disabled")
            except Exception:
                pass

        self._log_safe = mirrored_log_safe

        def on_close():
            self._console_fullscreen = False
            self._log_safe = original_log_safe
            win.grab_release()
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_close)

    # -------------------------------------------------------------------------
    #  CLEAR CONSOLE
    # -------------------------------------------------------------------------
    def _clear_console(self):
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.configure(state="disabled")

    # -------------------------------------------------------------------------
    #  CALLBACKS
    # -------------------------------------------------------------------------
    def _on_color_enabled_change(self):
        """Show/hide the Source/Destination panel based on the color toggle."""
        if self.color_enabled.get():
            self.color_sd_frame.pack(fill="x")
            # Show OCIO options if needed
            if self.color_dest.get() == "Custom OCIO...":
                self.ocio_options_frame.pack(fill="x")
                self._update_ocio_dropdowns()
            # Update detected profile display
            self._update_detected_color_display()
        else:
            self.color_sd_frame.pack_forget()

    def _on_color_dest_change(self, value):
        """Show/hide OCIO options based on destination selection."""
        if value == "Custom OCIO...":
            self.ocio_options_frame.pack(fill="x")
            self._update_ocio_dropdowns()
        else:
            self.ocio_options_frame.pack_forget()

    def _update_detected_color_display(self):
        """Update the detected color profile label from probed data."""
        profile = self.detected_color_profile.get()
        if profile:
            self.detected_profile_label.configure(text=f"🔍 Detected: {profile}")
        else:
            self.detected_profile_label.configure(text="")

    def _browse_ocio(self):
        path = self._native_file_dialog(mode="file", title="Select an OCIO configuration", file_filter="OCIO Config | *.ocio")
        if path:
            self.ocio_path.set(path)
            self._update_ocio_dropdowns()

    def _update_ocio_dropdowns(self):
        if not self.has_ocio_lib or self.color_dest.get() != "Custom OCIO...":
            return
        
        raw_path = self.ocio_path.get()
        if not raw_path:
            return
            
        path = raw_path.strip().strip("'").strip('"')
        if not Path(path).exists():
            self._log(f"[OCIO] File not found or invalid: {path}")
            self.ocio_spaces = []
            self.ocio_in_cs.set("Invalid file")
            self.ocio_out_cs.set("Invalid file")
            return

        try:
            if HAS_OCIO:
                config = oiio.ColorConfig(path)
                names = config.getColorSpaceNames()
                spaces = sorted(list(names)) if names else []
            else:
                spaces = []
                
            self.ocio_spaces = spaces
            
            if self.ocio_spaces:
                if not self.ocio_in_cs.get() in self.ocio_spaces:
                    self.ocio_in_cs.set(self.ocio_spaces[0])
                
                # Default ACEScg output to the canonical OCIO colorspace name.
                default_out = ACESCG_OCIO_SPACE
                if default_out in self.ocio_spaces:
                    self.ocio_out_cs.set(default_out)
                elif not self.ocio_out_cs.get() in self.ocio_spaces:
                    self.ocio_out_cs.set(self.ocio_spaces[0])
                    
                # Explicit update of button text just in case textvariable fails to redraw
                if hasattr(self, 'ocio_btn_in'):
                    self.ocio_btn_in.configure(text=self.ocio_in_cs.get())
                if hasattr(self, 'ocio_btn_out'):
                    self.ocio_btn_out.configure(text=self.ocio_out_cs.get())
                    
                self._log(f"[OCIO] {len(self.ocio_spaces)} colorspaces loaded.")
        except Exception as e:
            self._log(f"[OCIO] Error loading config: {e}")
            self.ocio_spaces = []
            self.ocio_in_cs.set("Error")
            self.ocio_out_cs.set("Error")

    def _open_ocio_in_selection(self):
        if not self.ocio_spaces:
            self._log("[OCIO] Colorspace list is empty. Please fix the config file path.")
            return
        
        def set_val(val):
            self.ocio_in_cs.set(val)
            self.ocio_btn_in.configure(text=val)
            
        SearchableSelectionWindow(self, "Select Input Space", self.ocio_spaces, 
                                  self.ocio_in_cs.get(), set_val)

    def _open_ocio_out_selection(self):
        if not self.ocio_spaces:
            self._log("[OCIO] Colorspace list is empty. Please fix the config file path.")
            return
            
        def set_val(val):
            self.ocio_out_cs.set(val)
            self.ocio_btn_out.configure(text=val)
            
        SearchableSelectionWindow(self, "Select Output Space", self.ocio_spaces, 
                                  self.ocio_out_cs.get(), set_val)

    def _on_fps_change(self, value):
        self.fps_label_var.set(f"{value:.1f} FPS")
        self._update_frame_estimate()

    def _on_workers_change(self, value):
        self.workers_label_var.set(f"{int(value)} Threads")

    def _on_mapper_change(self, value):
        if "GLOMAP" in value and not self.has_glomap:
            self.mapper_type.set("COLMAP")
            self._log("⚠️ GLOMAP is not installed on this system. Using COLMAP (Standard).")

    def _native_file_dialog(self, mode="file", title="Select", file_filter=None, multiple=False):
        """Use zenity (native GTK) for file dialogs on Linux, fallback to tkinter.
        When multiple=True, returns a list of paths (or empty list)."""
        if shutil.which("zenity"):
            cmd = ["zenity", "--file-selection", f"--title={title}"]
            if mode == "directory":
                cmd.append("--directory")
            if multiple:
                cmd.append("--multiple")
                cmd.extend(["--separator", "|"])
            if file_filter:
                cmd.extend(["--file-filter", file_filter])
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if result.returncode == 0:
                    raw = result.stdout.strip()
                    if multiple:
                        return [p for p in raw.split("|") if p] if raw else []
                    return raw
                return [] if multiple else None
            except Exception:
                pass
        if mode == "directory":
            result = tk_filedialog.askdirectory(title=title)
            if multiple:
                return [result] if result else []
            return result or None
        if multiple:
            result = tk_filedialog.askopenfilenames(title=title)
            return list(result) if result else []
        return tk_filedialog.askopenfilename(title=title) or None

    def _on_stray_approach_change(self, value):
        # Keep the selected value as-is; we'll read the segmented button directly later
        pass



    def _on_stray_confidence_change(self, value):
        try:
            self.stray_confidence.set(int(str(value)[0]))
        except Exception:
            pass

    def _on_input_mode_change(self, value):
        if value == "Video (.mp4, .mov)":
            self.input_label_var.set("Video File(s)")
            self.fps_frame.grid()
            # Show video card, hide stray card
            self.card_vid.pack(fill="x", pady=(0, 10), after=self._card_io_ref)
            self.card_stray.pack_forget()
            # Enable video extraction card
            for child in self.card_vid.content.winfo_children():
                try: child.configure(state="normal")
                except: pass
            self.slider_fps.configure(state="normal")
            # Reset display
            self.video_paths = []
            self.stray_paths = []
            self._video_infos = []
            self._rescan_infos = []
            self.video_path.set("")
            self.frame_estimate_var.set("")
        elif value == "Rescan (LiDAR)":
            self.input_label_var.set("Rescan Folder(s)")
            self.fps_frame.grid()
            # Hide video card, show stray card
            self.card_vid.pack_forget()
            self.card_stray.pack(fill="x", pady=(0, 10), after=self._card_io_ref)
            self.video_paths = []
            self.stray_paths = []
            self._video_infos = []
            self._rescan_infos = []
            self.video_path.set("")
            self.frame_estimate_var.set("")
            self.rescan_frame_estimate_label.configure(text="")
            # Auto-preset color conversion for Rescan
            self.color_enabled.set(True)
            self.color_source.set("Auto-detect")
            self.color_dest.set("ACEScg (EXR + sRGB PNG)")
            self.use_acescg_exr.set(True)
            self._on_color_enabled_change()
        else:
            self.input_label_var.set("Image Folder")
            self.fps_frame.grid_remove()
            # Hide both special cards
            self.card_vid.pack(fill="x", pady=(0, 10), after=self._card_io_ref)
            self.card_stray.pack_forget()
            self.video_paths = []
            self.stray_paths = []
            self._video_infos = []
            self._rescan_infos = []
            self.video_path.set("")
            self.frame_estimate_var.set("")
            
    def _probe_video_duration(self, video_path):
        """Use ffprobe to get video duration, native FPS, and color profile."""
        info = {"path": str(video_path), "duration": 0, "native_fps": 30,
                "total_frames": 0, "color_profile": ""}

        def _parse_rate(value, default=0.0):
            try:
                text = str(value or "").strip()
                if "/" in text:
                    num, den = text.split("/", 1)
                    den_value = float(den)
                    if den_value > 0:
                        return float(num) / den_value
                parsed = float(text)
                return parsed if parsed > 0 else default
            except Exception:
                return default

        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", str(video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                info["duration"] = float(data.get("format", {}).get("duration", 0))
                # Find video stream for FPS + color info
                for s in data.get("streams", []):
                    if s.get("codec_type") == "video":
                        native_fps = _parse_rate(s.get("avg_frame_rate"), 0.0)
                        if native_fps <= 0:
                            native_fps = _parse_rate(s.get("r_frame_rate"), 30.0)
                        info["native_fps"] = native_fps or 30.0
                        if not info["duration"]:
                            info["duration"] = float(s.get("duration", 0) or 0)
                        # Detect color profile
                        c_prim = s.get("color_primaries", "").lower()
                        c_trc = s.get("color_transfer", "").lower()
                        key = (c_prim, c_trc)
                        info["color_profile"] = _FFPROBE_TO_SOURCE.get(key, "")
                        info["color_primaries"] = c_prim
                        info["color_transfer"] = c_trc
                        try:
                            nb_frames = s.get("nb_frames")
                            if nb_frames and nb_frames != "N/A":
                                info["total_frames"] = int(float(nb_frames))
                        except Exception:
                            pass
                        break
                if not info["total_frames"] and info["duration"] > 0:
                    info["total_frames"] = round(info["duration"] * info["native_fps"])
        except Exception:
            pass
        return info

    def _probe_rescan_dataset(self, dataset_path):
        """Count frames in a Rescan dataset via odometry.csv and get native FPS."""
        odom_file = Path(dataset_path) / "odometry.csv"
        
        # Check both mp4 and mov
        vid_file = Path(dataset_path) / "rgb.mp4"
        if not vid_file.exists():
            vid_file = Path(dataset_path) / "rgb.mov"
        
        native_fps = 60.0 # final fallback
        
        # 1. Try to get FPS from MP4 if it exists
        if vid_file.exists():
            vid_info = self._probe_video_duration(vid_file)
            if vid_info.get("native_fps"):
                native_fps = vid_info["native_fps"]
            
        # 2. Get exact FPS from odometry logic
        try:
            with open(odom_file) as f:
                # header
                next(f, None)

                first_line = None
                last_line = None
                count = 0

                # Find first valid line
                for line in f:
                    line = line.strip()
                    if line:
                        first_line = line
                        last_line = line
                        count = 1
                        break

                # Count rest
                if count == 1:
                    for line in f:
                        line = line.strip()
                        if line:
                            count += 1
                            last_line = line

                total = count
                
                # If we have enough frames, calculate FPS from timestamps
                if total > 1 and first_line and last_line:
                    try:
                        t_start = float(first_line.split(',')[0])
                        t_end = float(last_line.split(',')[0])
                        duration = t_end - t_start
                        if duration > 0:
                            odometry_fps = (total - 1) / duration
                            if odometry_fps > 0.1: # simple sanity check
                                native_fps = odometry_fps
                    except Exception:
                        pass
                        
            return {"path": str(dataset_path), "total_frames": max(0, total), "native_fps": native_fps}
        except Exception:
            return {"path": str(dataset_path), "total_frames": 0, "native_fps": native_fps}

    def _update_frame_estimate(self):
        """Recalculate and display frame count estimates based on current settings."""
        mode = self.input_mode.get()
        if mode == "Video (.mp4, .mov)" and self._video_infos:
            fps = self.fps_extract.get()
            parts = []
            total = 0
            for info in self._video_infos:
                total_frames = info.get("total_frames", 0)
                native_fps = info.get("native_fps", fps)
                step = max(1, round(native_fps / fps)) if fps > 0 else 1
                est = math.ceil(total_frames / step) if total_frames > 0 else (round(info["duration"] * fps) if info["duration"] > 0 else 0)
                name = Path(info["path"]).stem
                if len(name) > 20:
                    name = name[:17] + "…"
                parts.append(f"{name}: ~{est}")
                total += est
            if len(self._video_infos) == 1:
                self.frame_estimate_var.set(f"~{total} estimated frames")
            else:
                detail = "\n".join(parts)
                self.frame_estimate_var.set(f"~{total} estimated frames total:\n{detail}")
        elif mode == "Rescan (LiDAR)" and self._rescan_infos:
            fps = self.fps_extract.get()
            parts = []
            total = 0
            for info in self._rescan_infos:
                nat_fps = info.get("native_fps", 30.0)
                step = max(1, round(nat_fps / fps))
                est = math.ceil(info.get("total_frames", 0) / step)
                name = Path(info["path"]).name
                if len(name) > 20:
                    name = name[:17] + "…"
                parts.append(f"{name}: ~{est}")
                total += est
            
            if len(self._rescan_infos) == 1:
                # Need step for display (assume 1 dataset logic)
                nat_fps = self._rescan_infos[0].get("native_fps", 30.0)
                step = max(1, round(nat_fps / fps))
                txt = f"~{total} frames (from {self._rescan_infos[0]['total_frames']} poses, step={step})"
            else:
                txt = f"~{total} frames total:\n" + "\n".join(parts)
            self.rescan_frame_estimate_label.configure(text=txt)
        else:
            self.frame_estimate_var.set("")
            if hasattr(self, 'rescan_frame_estimate_label'):
                self.rescan_frame_estimate_label.configure(text="")

    def _browse_input(self):
        mode = self.input_mode.get()
        if mode == "VIDEO" or mode == "Video (.mp4, .mov)":
            files = self._native_file_dialog(mode="file", title="Select one or more videos",
                                              file_filter="Videos | *.mp4 *.mov *.avi *.webm",
                                              multiple=True)
            if files:
                self.video_paths = [Path(f) for f in files]
                if len(self.video_paths) == 1:
                    self.video_path.set(str(self.video_paths[0]))
                else:
                    self.video_path.set(f"{len(self.video_paths)} videos selected")
                # Probe videos for frame estimates (in background)
                def probe():
                    self._video_infos = [self._probe_video_duration(v) for v in self.video_paths]
                    # Detect color profile from first video
                    detected = ""
                    for info in self._video_infos:
                        if info.get("color_profile"):
                            detected = info["color_profile"]
                            break
                    def update_ui():
                        self._update_frame_estimate()
                        if detected:
                            self.detected_color_profile.set(detected)
                            self._update_detected_color_display()
                    self.after(0, update_ui)
                threading.Thread(target=probe, daemon=True).start()
        elif mode == "Rescan (LiDAR)":
            dirs = self._native_file_dialog(mode="directory", title="Select a Rescan folder (or parent folder containing multiple)", multiple=True)
            if not dirs:
                return
            if isinstance(dirs, str):
                dirs = [dirs]
            valid_dirs = []
            for d in dirs:
                p = Path(d)
                has_rgb_video = (p / "rgb.mp4").exists() or (p / "rgb.mov").exists()
                has_rgb_seq = (p / "rgb").is_dir() and any(
                    f.is_file() and f.suffix.lower() in {".exr", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
                    for f in (p / "rgb").iterdir()
                )
                has_rgb = has_rgb_video or has_rgb_seq
                has_odom = (p / "odometry.csv").exists()
                has_cam = (p / "camera_matrix.csv").exists()
                if has_rgb and has_odom and has_cam:
                    valid_dirs.append(p)
                    self._log(f"📱 Rescan dataset detected: {p.name}")
                    depth_dir = p / "depth"
                    if depth_dir.exists():
                        n_depth = len(list(depth_dir.glob("*.png")))
                        self._log(f"   → {n_depth} depth maps available")
                else:
                    found_any = False
                    try:
                        for sub_p in p.iterdir():
                            if sub_p.is_dir():
                                s_has_rgb_video = (sub_p / "rgb.mp4").exists() or (sub_p / "rgb.mov").exists()
                                s_has_rgb_seq = (sub_p / "rgb").is_dir() and any(
                                    f.is_file() and f.suffix.lower() in {".exr", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
                                    for f in (sub_p / "rgb").iterdir()
                                )
                                s_has_rgb = s_has_rgb_video or s_has_rgb_seq
                                s_has_odom = (sub_p / "odometry.csv").exists()
                                s_has_cam = (sub_p / "camera_matrix.csv").exists()
                                if s_has_rgb and s_has_odom and s_has_cam:
                                    valid_dirs.append(sub_p)
                                    self._log(f"📱 Rescan dataset detected inside parent: {sub_p.name}")
                                    depth_dir = sub_p / "depth"
                                    if depth_dir.exists():
                                        n_depth = len(list(depth_dir.glob("*.png")))
                                        self._log(f"   → {n_depth} depth maps available")
                                    found_any = True
                    except Exception:
                        pass

                    if not found_any:
                        missing = []
                        if not has_rgb: missing.append("rgb.mp4 or rgb.mov or rgb/ image folder")
                        if not has_odom: missing.append("odometry.csv")
                        if not has_cam: missing.append("camera_matrix.csv")
                        self._log(f"⚠ {p.name}: invalid Rescan dataset or parent folder.")
                        self._log(f"   Missing files: {', '.join(missing)}")
            if valid_dirs:
                self.stray_paths = valid_dirs
                self.video_paths = []
                if len(valid_dirs) == 1:
                    self.video_path.set(str(valid_dirs[0]))
                else:
                    self.video_path.set(f"{len(valid_dirs)} LiDAR datasets selected")

                # Probe Rescan datasets for frame estimates (in background)
                def probe_rescan():
                    infos = [self._probe_rescan_dataset(d) for d in valid_dirs]
                    # Detect color profile from first dataset's video
                    detected = ""
                    for d in valid_dirs:
                        for vname in ["rgb.mov", "rgb.mp4"]:
                            vid = d / vname
                            if vid.exists():
                                vid_info = self._probe_video_duration(vid)
                                if vid_info.get("color_profile"):
                                    detected = vid_info["color_profile"]
                                    break
                        if detected:
                            break

                    def update_ui():
                        self._rescan_infos = infos
                        self._update_frame_estimate()
                        if detected:
                            self.detected_color_profile.set(detected)
                            self._update_detected_color_display()

                    self.after(0, update_ui)

                self.rescan_frame_estimate_label.configure(text="Probing datasets...")
                threading.Thread(target=probe_rescan, daemon=True).start()
        else:
             d = self._native_file_dialog(mode="directory", title="Select the image folder")
             if d:
                 self.video_paths = []
                 self.video_path.set(d)

    def _browse_output(self):
        d = self._native_file_dialog(mode="directory", title="Select the output folder")
        if d:
            self.output_path.set(d)

    def _log(self, msg):
        self.console.after(0, self._log_safe, msg)

    def _log_tagged(self, tag, msg):
        """Helper to log with a manual tag independent of the GuiLogHandler state"""
        self._log(f"{tag} {msg}")

    def _log_safe(self, msg):
        self.console.configure(state="normal")
        self.console.insert("end", msg + "\n")
        self.console.see("end")
        self.console.configure(state="disabled")

    def _run_ffmpeg_streamed(self, cmd, log_tag):
        """Run an FFmpeg command and stream its stderr to the GUI console in real-time.

        Returns (proc, stderr_lines, thread) where stderr_lines is a list populated
        asynchronously by the reader thread. Join the thread before reading stderr_lines
        to ensure all output has been captured.
        """
        stderr_lines = []

        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

        def _reader():
            for raw_line in proc.stderr:
                line = raw_line.strip()
                if line:
                    stderr_lines.append(line)
                    self._log_tagged(log_tag, f"         {line}")

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        return proc, stderr_lines, t

    def _set_step(self, index, total=None):
        total = total or len(self.STEPS)
        name = self.STEPS[index] if index < len(self.STEPS) else "Done"
        progress = index / total
        GUIProgressTqdm._step_index = index
        GUIProgressTqdm._step_name = name
        self.after(0, lambda: self.step_label.configure(
            text=f"Step {index + 1}/{total} — {name}",
            text_color=COLORS["accent_blue"]))
        self.after(0, lambda: self.progress_bar.set(progress))

    def _check_cancelled(self):
        """Call between steps to check if the user requested cancellation."""
        if self._cancelled:
            raise CancelledError("Processing cancelled by user")

    def _filter_frame_quality(self, image_dir, tag="[CPU]"):
        """Optionally move blurry/black frames out of the active image set."""
        def _get_var(name, default):
            value = getattr(self, name, None)
            if value is None:
                return default
            try:
                return value.get()
            except Exception:
                return default

        reject_blurry = bool(_get_var("exclude_blurry", False))
        reject_black = bool(_get_var("exclude_black", False))
        if not (reject_blurry or reject_black):
            return
        result = reject_low_quality_frames(
            image_dir,
            reject_blurry=reject_blurry,
            reject_black=reject_black,
            blur_threshold=float(_get_var("blur_threshold", 75.0)),
            black_threshold=float(_get_var("black_threshold", 0.08)),
            logger=lambda msg: self._log_tagged(tag, f"       {msg}"),
        )
        if result.get("rejected"):
            self._log_tagged(tag, f"       -> Quality filter moved {len(result['rejected'])} frame(s) to _rejected_frames")

    def _apply_quality_sweep_sample(self, image_dir, tag="[CPU]"):
        def _get_var(name, default):
            value = getattr(self, name, None)
            if value is None:
                return default
            try:
                return value.get()
            except Exception:
                return default

        if not bool(_get_var("quality_sweep", False)):
            return
        limit = int(_get_var("sweep_sample_frames", 80))
        if limit <= 0:
            return
        directory = Path(image_dir)
        images = sorted([
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".exr")
        ])
        if len(images) <= limit:
            return
        keep_indexes = set()
        if limit == 1:
            keep_indexes.add(len(images) // 2)
        else:
            for idx in range(limit):
                keep_indexes.add(round(idx * (len(images) - 1) / (limit - 1)))
        unused_dir = directory / "_sweep_unused_frames"
        unused_dir.mkdir(parents=True, exist_ok=True)
        moved = 0
        for idx, path in enumerate(images):
            if idx not in keep_indexes:
                shutil.move(str(path), unused_dir / path.name)
                moved += 1
        self._log_tagged(tag, f"       -> Quality sweep sampled {len(images) - moved}/{len(images)} frames")

    def _finish(self, success=True, cancelled=False):
        if cancelled:
            color = COLORS["warning"]
            text = "⏹ Processing cancelled"
        elif success:
            color = COLORS["success"]
            text = "✓ Completed successfully"
        else:
            color = COLORS["error"]
            text = "✗ Error during processing"
        self.after(0, lambda: self.step_label.configure(text=text, text_color=color))
        if success and not cancelled:
            self.after(0, lambda: self.progress_bar.set(1.0))
        self.after(0, lambda: self.btn_run.configure(state="normal", text="⚡  START PROCESSING"))
        self.after(0, lambda: self.btn_cancel.configure(state="disabled"))
        self._processing = False

    def _generate_api_key(self):
        """Generate and display a new random API key (only when server is stopped)."""
        if not self.server_running:
            self.server_key_var.set(secrets.token_urlsafe(32))

    def _check_server_connection(self):
        """Test connectivity to the local server via the public /health endpoint."""
        try:
            port = self.server_port.get()
        except Exception:
            self._log_tagged("[SERVER]", "❌ Invalid port number")
            return

        def _do_check():
            try:
                import urllib.request
                url = f"http://127.0.0.1:{port}/api/v1/health"
                with urllib.request.urlopen(url, timeout=3) as resp:
                    self.after(0, lambda: self._log_tagged(
                        "[SERVER]", f"✅ Server reachable on port {port} (HTTP {resp.status})"))
            except Exception as exc:
                self.after(0, lambda: self._log_tagged(
                    "[SERVER]", f"❌ Server not reachable on port {port}: {exc}"))

        threading.Thread(target=_do_check, daemon=True).start()

    def _toggle_server(self):
        """Start or stop the built-in API server."""
        if self.server_running:
            self.server_running = False
            self.btn_server_toggle.configure(text="▶  Start Server",
                                             fg_color=COLORS["accent_blue"])
            self.server_status_label.configure(text="● Stopped",
                                               text_color=COLORS["text_muted"])
            self._log_tagged("[SERVER]", "API server stopped")
            if self._server_process is not None:
                self._server_process.terminate()
                try:
                    self._server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._server_process.kill()
                self._server_process = None
            # Re-enable key entry so user can change it before next start
            self.server_key_entry.configure(state="normal")
        else:
            try:
                port = self.server_port.get()
            except Exception:
                self._log_tagged("[SERVER]", "❌ Invalid port number")
                return

            api_key = self.server_key_var.get().strip()
            if not api_key:
                api_key = secrets.token_urlsafe(32)
                self.server_key_var.set(api_key)

            server_script = Path(__file__).resolve().parent / "remap_server.py"
            if not server_script.exists():
                self._log_tagged("[SERVER]", f"❌ Server script not found: {server_script}")
                return

            log_path = Path(tempfile.gettempdir()) / "remap_server.log"
            self._server_log_fp = open(log_path, "w")  # noqa: SIM115
            self._server_process = subprocess.Popen(
                [sys.executable, str(server_script),
                 "--host", "0.0.0.0", "--port", str(port),
                 "--api-key", api_key],
                stdout=self._server_log_fp,
                stderr=subprocess.STDOUT,
            )
            self._server_api_key = api_key
            self.server_running = True
            # Lock key entry while server is running to avoid confusion
            self.server_key_entry.configure(state="disabled")
            self.btn_server_toggle.configure(text="■  Stop Server",
                                             fg_color=COLORS["error"])
            self.server_status_label.configure(text=f"● Running on port {port}",
                                               text_color=COLORS["success"])
            self._log_tagged("[SERVER]", f"API server started on http://0.0.0.0:{port}")
            self._log_tagged("[SERVER]", f"API Key: {api_key}")
            self._log_tagged("[SERVER]", f"Server log: {log_path}")

    def _cancel_process(self):
        """Set the cancellation flag. The processing thread checks it between steps."""
        if self._processing:
            self._cancelled = True
            self._log("⏹ Cancellation requested...")

    # -------------------------------------------------------------------------
    #  PROCESSING THREAD
    # -------------------------------------------------------------------------
    def _start_thread(self):
        if not self.output_path.get():
            self._log("❌ ERROR: Output folder is missing.")
            return

        is_video = (self.input_mode.get() == "VIDEO" or self.input_mode.get() == "Video (.mp4, .mov)")
        is_stray = (self.input_mode.get() == "Rescan (LiDAR)")
        if is_video and not self.video_paths:
            self._log("❌ ERROR: No video selected.")
            return
        if is_stray and not self.stray_paths:
            self._log("❌ ERROR: No Rescan dataset selected.")
            return
        if not is_video and not is_stray and not self.video_path.get():
            self._log("❌ ERROR: No input selected.")
            return

        self._cancelled = False
        self._processing = True
        self.btn_run.configure(state="disabled", text="⏳  Processing...")
        self.btn_cancel.configure(state="normal")
        self.progress_bar.set(0)
        self.step_label.configure(text="Starting...", text_color=COLORS["accent_blue"])
        threading.Thread(target=self._run_process, daemon=True).start()

    def _run_process(self):
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        # Initialize variables that might be used in finally block
        hloc_logger = None
        gui_handler = None
        old_tqdm = None
        import tqdm as tqdm_module

        def apply_color_conversion(target_dir, tag_prefix, step_description, exr_out_dir=None):
            if not self.color_enabled.get():
                return
            
            _source = self.color_source.get()
            _dest = self.color_dest.get()
            is_rescan_color = self.input_mode.get() == "Rescan (LiDAR)"

            # Resolve "Auto-detect" to the detected profile
            if _source == "Auto-detect":
                _source = self.detected_color_profile.get()
                if not _source:
                    _source = "Apple Log (BT.2020)" if is_rescan_color else "Linear BT.2020"
                    self._log_tagged(tag_prefix, f"       ⚠ No color profile detected, assuming {_source}")

            if is_rescan_color and not self.use_acescg_exr.get() and _dest == "ACEScg (EXR + sRGB PNG)":
                _dest = "sRGB (Tone Mapped)"
                exr_out_dir = None

            self._check_cancelled()
            self._log_tagged(tag_prefix, step_description)
            self._log_tagged(tag_prefix, f"       ↳ {_source} → {_dest}")

            # Validate OCIO destination
            if _dest == "Custom OCIO...":
                cfg_path = self.ocio_path.get()
                cs_in = self.ocio_in_cs.get()
                cs_out = self.ocio_out_cs.get()
                if not cs_in or not cs_out:
                    self._log(f"❌ OCIO ERROR: Colorspaces not defined.")
                    raise ValueError("OCIO colorspaces missing")
                if not HAS_OCIO:
                    raise ValueError("Custom OCIO destination selected but OpenImageIO not available.")
            else:
                cfg_path = None
                cs_in = None
                cs_out = None

            exts = (".exr", ".jpg", ".jpeg", ".png", ".tif", ".tiff")
            all_images = []
            with os.scandir(target_dir) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name.lower().endswith(exts):
                        all_images.append(Path(entry.path))
            
            if not all_images:
                raise ValueError(f"No images found for color conversion in {target_dir}")

            total_color = len(all_images)

            success_count = 0
            errors = []
            try:
                configured_threads = int(self.num_workers.get())
            except Exception:
                configured_threads = 1
            threads = max(1, configured_threads)
            self._log_tagged(tag_prefix, f"       -> Converting {total_color} images with {threads} isolated worker process(es)...")
            
            exr_dir_str = str(exr_out_dir) if exr_out_dir else None

            with tqdm_module.tqdm(total=total_color, desc="Color Conversion") as pbar:
                from concurrent.futures import ProcessPoolExecutor, as_completed
                from concurrent.futures.process import BrokenProcessPool
                futures = []
                try:
                    with ProcessPoolExecutor(max_workers=threads) as executor:
                        for img_path in all_images:
                            futures.append(executor.submit(
                                process_image_color_worker,
                                str(img_path),
                                _source, _dest,
                                cs_in, cs_out, cfg_path, exr_dir_str
                            ))

                        for future in as_completed(futures):
                            if self._cancelled:
                                executor.shutdown(wait=False, cancel_futures=True)
                                break
                            try:
                                res, err = future.result()
                            except BrokenProcessPool as e:
                                res, err = False, f"Native color worker crashed: {e}"
                            except Exception as e:
                                res, err = False, str(e)
                            if res:
                                success_count += 1
                            else:
                                if err:
                                    errors.append(err)
                            pbar.update(1)
                            processed = success_count + len(errors)
                            if processed % 50 == 0 or processed == total_color:
                                self._log_tagged(
                                    tag_prefix,
                                    f"       -> Color conversion progress: {processed}/{total_color}",
                                )
                except BrokenProcessPool as e:
                    errors.append(f"Native color worker pool crashed: {e}")

            if errors:
                first_few_errors = list(set(errors))[:3]
                self._log_tagged(tag_prefix, f"       ❌ Encountered {len(errors)} errors. Examples: {first_few_errors}")
                raise RuntimeError(f"Color conversion failed for {len(errors)} image(s): {first_few_errors}")

            if success_count != total_color:
                raise RuntimeError(f"Color conversion incomplete: {success_count}/{total_color} images converted")

            self._log_tagged(tag_prefix, f"       → OK ({success_count}/{total_color} converted)")

        try:
            gui_stream = GuiStream(self._log)
            sys.stdout = gui_stream
            sys.stderr = gui_stream

            # Monkeypatch tqdm for GUI progress
            GUIProgressTqdm._gui_app = self
            old_tqdm = tqdm_module.tqdm
            tqdm_module.tqdm = GUIProgressTqdm
            for mod in [extract_features, match_features, reconstruction]:
                if hasattr(mod, 'tqdm'):
                    mod.tqdm = GUIProgressTqdm

            # Custom Logging Handler for Hloc / internal logs
            hloc_logger = logging.getLogger("hloc")
            hloc_logger.setLevel(logging.DEBUG)
            
            gui_handler = GuiLogHandler(self._log)
            formatter = logging.Formatter('%(name)s: %(message)s')
            gui_handler.setFormatter(formatter)
            hloc_logger.addHandler(gui_handler)

            base_out = Path(self.output_path.get())

            try:
                import torch
                has_gpu = torch.cuda.is_available()
            except ImportError:
                has_gpu = False
                
            device_tag = "[GPU]" if has_gpu else "[CPU]"

            self._log_tagged(device_tag, "ReMap SfM Pipeline started")

            # 1. PREP / FFMPEG
            self._set_step(0)
            self._check_cancelled()
            gui_handler.set_tag(device_tag)
            
            # Determine Input Mode
            is_video_mode = (self.input_mode.get() == "VIDEO" or self.input_mode.get() == "Video (.mp4, .mov)")
            is_stray_mode = (self.input_mode.get() == "Rescan (LiDAR)")
            stray_result = None  # Will hold stray conversion result if applicable

            images_dir = base_out / "images"
            outputs_dir = base_out / "hloc_outputs"
            sfm_dir = base_out / "sparse" / "0"

            if not is_stray_mode:
                images_dir.mkdir(parents=True, exist_ok=True)
                outputs_dir.mkdir(parents=True, exist_ok=True)
                sfm_dir.mkdir(parents=True, exist_ok=True)

            if is_stray_mode:
                # ═══ RESCAN MODE — INDEPENDENT PROCESSING PER DATASET ═══
                stray_dirs = self.stray_paths
                if not stray_dirs:
                    raise ValueError("No Rescan folder selected!")

                approach = self.stray_approach.get()
                # Map segmented button label → internal mode string
                if "ARKit" in approach or "known" in approach:
                    stray_mode = "known_poses"
                else:
                    stray_mode = "full_sfm"

                # Use appropriate step list
                if stray_mode == "known_poses":
                    self.STEPS = self.STEPS_STRAY_A
                else:
                    self.STEPS = self.STEPS_STRAY_B

                total_ds = len(stray_dirs)
                self._log_tagged("[LiDAR]", f"═══ Processing {total_ds} Rescan dataset(s) independently (mode: {stray_mode}) ═══")

                for ds_idx, stray_dir in enumerate(stray_dirs, start=1):
                    self._check_cancelled()
                    ds_name = stray_dir.name
                    ds_out = base_out if total_ds == 1 else base_out / ds_name
                    ds_images = ds_out / "images"
                    ds_hloc = ds_out / "hloc_outputs"
                    ds_sfm = ds_out / "sparse" / "0"
                    ds_final = ds_out / f"{ds_out.name}_SfM_Dataset_Output"
                    ds_final_images = ds_final / "images"
                    ds_images.mkdir(parents=True, exist_ok=True)
                    ds_hloc.mkdir(parents=True, exist_ok=True)
                    ds_sfm.mkdir(parents=True, exist_ok=True)
                    if self.use_acescg_exr.get():
                        ds_final_images.mkdir(parents=True, exist_ok=True)

                    ds_tag = f"[{ds_idx}/{total_ds}]"
                    self._log_tagged("[LiDAR]", f"\n{'═' * 60}")
                    self._log_tagged("[LiDAR]", f"{ds_tag} Dataset : {ds_name}")
                    self._log_tagged("[LiDAR]", f"{'═' * 60}")

                    # ── Step 1: Stray → COLMAP conversion ──
                    self._set_step(0)
                    gui_handler.set_tag("[LiDAR]")
                    self._log_tagged("[LiDAR]", f"{ds_tag} [1/5] Stray → COLMAP conversion...")

                    ds_info = next((info for info in self._rescan_infos if info["path"] == str(stray_dir)), None)
                    native_fps = ds_info["native_fps"] if ds_info else 60.0
                    target_fps = max(float(self.fps_extract.get()), 0.01)
                    computed_subsample = max(1, round(native_fps / target_fps))
                    self._log_tagged(
                        "[LiDAR]",
                        f"{ds_tag}   Sampling: native {native_fps:.2f} FPS -> target {target_fps:.1f} FPS (step={computed_subsample})",
                    )

                    stray_result = convert_stray_to_colmap(
                        input_dir=stray_dir,
                        output_dir=ds_out,
                        mode=stray_mode,
                        subsample=computed_subsample,
                        confidence_threshold=self.stray_confidence.get(),
                        depth_subsample=self.stray_depth_subsample.get(),
                        skip_pointcloud=not self.stray_gen_pointcloud.get(),
                        use_cuda=True,
                        image_prefix="",
                        sfm_source_space=(
                            self.detected_color_profile.get() or "Auto-detect"
                            if self.color_source.get() == "Auto-detect"
                            else self.color_source.get()
                        ),
                        logger=lambda msg, _t=ds_tag: self._log_tagged("[LiDAR]", f"{_t} {msg}"),
                        cancel_check=self._check_cancelled,
                    )
                    self._log_tagged("[LiDAR]", f"{ds_tag}   ✓ {stray_result['n_images']} images, "
                                     f"{stray_result['n_points']:,} points LiDAR")

                    if stray_result.get("has_exr_source", False):
                        self._log_tagged(
                            "[CPU]",
                            f"{ds_tag} [*/5] EXR source detected: using tone-mapped sRGB PNG proxies for SfM; "
                            "EXR color conversion is deferred until after reconstruction.",
                        )
                    else:
                        apply_color_conversion(
                            ds_images,
                            "[CPU]",
                            f"{ds_tag} [*/5] Applying Color Conversion ({self.color_source.get()} → {self.color_dest.get()})...",
                            exr_out_dir=ds_final_images if self.use_acescg_exr.get() else None,
                        )

                    # ── Step 2: Features ──
                    self._filter_frame_quality(ds_images, "[CPU]")
                    self._apply_quality_sweep_sample(ds_images, "[CPU]")
                    self._check_cancelled()
                    self._set_step(1)
                    device_str = "[GPU]" if torch.cuda.is_available() else "[CPU]"
                    gui_handler.set_tag(device_str)
                    matcher_name = self.matcher_type.get()
                    loma_pipeline = (
                        LoMaMatcher(
                            matcher_name,
                            max_keypoints=int(self.max_keypoints.get()),
                            logger=lambda msg, _t=ds_tag, _d=device_str: self._log_tagged(_d, f"{_t} {msg}"),
                            cancel_check=self._check_cancelled,
                        )
                        if is_loma_matcher(matcher_name)
                        else None
                    )

                    ds_features_out = loma_feature_path(ds_hloc, matcher_name) if loma_pipeline else ds_hloc / 'features.h5'
                    if loma_pipeline is not None:
                        self._log_tagged(device_str, f"{ds_tag} [2/5] LoMa feature cache - {matcher_name} (max {self.max_keypoints.get()})...")
                        ds_features_path = loma_pipeline.extract_features(ds_images, ds_features_out)
                        self._log_tagged(device_str, f"{ds_tag}   LoMa features ready")
                    elif ds_features_out.exists():
                        self._log_tagged(device_str, f"{ds_tag} [2/5] Features skipped (already present)")
                        ds_features_path = ds_features_out
                    else:
                        self._log_tagged(device_str, f"{ds_tag} [2/5] Features — {self.feature_type.get()} (max {self.max_keypoints.get()})...")
                        conf_feature = extract_features.confs[self.feature_type.get()]
                        conf_feature['model']['max_keypoints'] = int(self.max_keypoints.get())
                        ds_features_path = extract_features.main(
                            conf_feature, ds_images, feature_path=ds_features_out
                        )
                        self._log_tagged(device_str, f"{ds_tag}   ✓ Features extracted")

                    # ── Step 3: Pairs ──
                    self._check_cancelled()
                    self._set_step(2)
                    gui_handler.set_tag("[CPU]")
                    pair_mode = self.pairing_mode.get()
                    ds_pairs_path = ds_hloc / 'pairs.txt'
                    if ds_pairs_path.exists():
                        _pair_count = count_pairs_file(ds_pairs_path)
                        self._log_tagged("[CPU]", f"{ds_tag} [3/5] Pairs skipped ({_pair_count} pairs already present)")
                    else:
                        self._log_tagged("[CPU]", f"{ds_tag} [3/5] Pairs — {pair_mode}...")
                        if "Sequential" in pair_mode:
                            _pair_count = generate_sequential_pairs(ds_images, ds_pairs_path, overlap=20)
                        else:
                            pairs_from_exhaustive.main(ds_pairs_path, image_list=None, features=ds_features_path)
                            _pair_count = count_pairs_file(ds_pairs_path)
                        self._log_tagged("[CPU]", f"{ds_tag}   ✓ {_pair_count} pairs generated")
                    self.after(0, lambda _n=_pair_count, _t=ds_tag: self.step_label.configure(
                        text=f"{_t} Pairs: {_n} pairs (100%)",
                        text_color=COLORS["accent_blue"],
                    ))

                    # ── Step 4: Matching ──
                    self._check_cancelled()
                    self._set_step(3)
                    gui_handler.set_tag(device_str)
                    ds_matches_out = loma_matches_path(ds_hloc, matcher_name) if loma_pipeline else ds_hloc / 'matches.h5'
                    if loma_pipeline is not None:
                        self._log_tagged(device_str, f"{ds_tag} [4/5] Matching - {matcher_name}...")
                        ds_matches_path = loma_pipeline.match_pairs(ds_pairs_path, ds_features_path, ds_matches_out)
                        self._log_tagged(device_str, f"{ds_tag}   LoMa matching complete")
                    elif ds_matches_out.exists():
                        self._log_tagged(device_str, f"{ds_tag} [4/5] Matching skipped (already present)")
                        ds_matches_path = ds_matches_out
                    else:
                        self._log_tagged(device_str, f"{ds_tag} [4/5] Matching — {self.matcher_type.get()}...")
                        conf_match = match_features.confs[self.matcher_type.get()]
                        ds_matches_path = match_features.main(
                            conf_match, ds_pairs_path, features=ds_features_path, matches=ds_matches_out
                        )
                        self._log_tagged(device_str, f"{ds_tag}   ✓ Matching complete")

                    # ── Step 5: SfM / Triangulation ──
                    self._check_cancelled()
                    self._set_step(4)


                    if stray_result and stray_result["mode"] == "known_poses":
                        sfm_tag = "[CPU/GPU]"
                        gui_handler.set_tag(sfm_tag)
                        self._log_tagged(sfm_tag, f"{ds_tag} [5/5] Triangulation COLMAP (poses ARKit fixées)...")

                        def tagged_logger(msg, _t=ds_tag, _st=sfm_tag):
                            self._log(f"{_st} {_t} {msg}")

                        tagged_logger("   → Creating COLMAP database...")
                        db_path = ds_hloc / "database.db"

                        from hloc.reconstruction import (
                            create_empty_db, import_images, get_image_ids,
                            import_features, import_matches,
                            estimation_and_geometric_verification,
                        )
                        create_empty_db(db_path)
                        import_images(ds_images, db_path, pycolmap.CameraMode.SINGLE, "PINHOLE")
                        image_ids = get_image_ids(db_path)
                        import_features(image_ids, db_path, ds_features_path)
                        import_matches(image_ids, db_path, ds_pairs_path, ds_matches_path)
                        estimation_and_geometric_verification(db_path, ds_pairs_path)

                        tagged_logger("   → Triangulating 3D points...")
                        recon = pycolmap.Reconstruction(str(ds_sfm))
                        pycolmap.triangulate_points(
                            recon, str(db_path), str(ds_images), str(ds_sfm),
                        )
                        recon.write(str(ds_sfm))
                        n_pts = recon.num_points3D()
                        n_imgs = recon.num_reg_images()
                        tagged_logger(f"   ✓ Triangulation complete: {n_imgs} images, {n_pts:,} 3D points")
                    else:
                        sfm_tag = "[GPU]" if self.mapper_type.get() == "GLOMAP" else "[CPU/GPU]"
                        gui_handler.set_tag(sfm_tag)
                        self._log_tagged(sfm_tag, f"{ds_tag} [5/5] Reconstruction SfM — {self.mapper_type.get()} (PINHOLE, auto)...")

                        def tagged_logger(msg, _t=ds_tag, _st=sfm_tag):
                            self._log(f"{_st} {_t} {msg}")

                        run_sfm_with_live_export(
                            sfm_dir=ds_sfm,
                            image_dir=ds_images,
                            pairs=ds_pairs_path,
                            features=ds_features_path,
                            matches=ds_matches_path,
                            camera_mode=pycolmap.CameraMode.AUTO,
                            camera_model="PINHOLE",
                            mapper_type=self.mapper_type.get(),
                            shared_dir=ds_out / "live_reconstruction",
                            export_every=5,
                            cancel_check=self._check_cancelled,
                            logger=tagged_logger,
                            num_threads=self.num_workers.get(),
                        )
                        self._log_tagged(sfm_tag, f"{ds_tag}   ✓ Reconstruction complete")

                    # ── EXR remapping (when source was EXR image sequence) ──
                    if stray_result and stray_result.get("has_exr_source", False) and self.use_acescg_exr.get():
                        gui_handler.set_tag("[CPU]")
                        self._log_tagged("[CPU]", f"{ds_tag} EXR remapping: replacing PNG with "
                                         "colorspace-converted source EXR files...")
                        frame_to_filename = stray_result.get("frame_to_filename", {})
                        src_rgb_dir = stray_dir / "rgb"
                        # Copy original EXR files to a temp dir, apply color pipeline,
                        # then move to images/ so apply_color_conversion only sees EXRs.
                        # The temp dir is populated exclusively from frame_to_filename so
                        # only the expected frames are moved to images/.
                        with tempfile.TemporaryDirectory() as _tmpdir:
                            _tmpdir_path = Path(_tmpdir)
                            _copied = 0
                            for _fidx, _png_name in frame_to_filename.items():
                                _exr_src = src_rgb_dir / f"{_fidx:06d}.exr"
                                if _exr_src.exists():
                                    shutil.copy2(_exr_src, _tmpdir_path / f"{_fidx:06d}.exr")
                                    _copied += 1
                                else:
                                    self._log_tagged("[CPU]", f"{ds_tag}   ⚠ Missing source EXR: "
                                                     f"{_exr_src.name}")
                            self._log_tagged("[CPU]", f"{ds_tag}   → {_copied} EXR file(s) ready "
                                             "for color pipeline")
                            # Apply color pipeline to EXR files only (temp dir)
                            if self.color_enabled.get():
                                apply_color_conversion(
                                    _tmpdir_path, "[CPU]",
                                    f"{ds_tag}   EXR remapping: applying color conversion "
                                    f"({self.color_source.get()} → {self.color_dest.get()})...",
                                )
                            # Move converted EXR files to images/
                            for _exr_f in _tmpdir_path.glob("*.exr"):
                                shutil.move(str(_exr_f), ds_images / _exr_f.name)
                        # Rename image entries in sparse model (png → exr)
                        try:
                            from backend.colmap_images import read_images_bin as _read_images_bin, write_images_bin as _write_images_bin
                            _bin_path = ds_sfm / "images.bin"
                            if _bin_path.exists():
                                _images_data = _read_images_bin(_bin_path)
                                _renamed = 0
                                for _img in _images_data:
                                    if _img["name"].endswith(".png"):
                                        _img["name"] = _img["name"][:-4] + ".exr"
                                        _renamed += 1
                                if _renamed > 0:
                                    _write_images_bin(_bin_path, _images_data)
                                    self._log_tagged("[CPU]", f"{ds_tag}   → Sparse model updated: "
                                                     f"{_renamed} image reference(s) renamed "
                                                     "(.png → .exr)")
                            else:
                                self._log_tagged("[CPU]", f"{ds_tag}   ⚠ images.bin not found")
                        except Exception as _exc:
                            self._log_tagged("[CPU]", f"{ds_tag}   ⚠ Could not update sparse "
                                             f"model image names: {_exc}")
                        # Delete intermediate PNG files
                        _deleted = 0
                        for _, _png_name in frame_to_filename.items():
                            _png_path = ds_images / _png_name
                            if _png_path.exists():
                                _png_path.unlink()
                                _deleted += 1
                        self._log_tagged("[CPU]", f"{ds_tag}   → {_deleted} intermediate PNG "
                                         "file(s) removed")
                        self._log_tagged("[CPU]", f"{ds_tag}   ✓ EXR remapping complete")
                    try:
                        final_bundle = normalize_final_bundle(
                            ds_out,
                            keep_srgb_png=False,
                            use_acescg_exr=self.use_acescg_exr.get(),
                        )
                        if final_bundle:
                            self._log_tagged("[CPU]", f"{ds_tag}   -> Final SfM dataset output: {final_bundle.name}")
                    except Exception as _bundle_exc:
                        self._log_tagged("[CPU]", f"{ds_tag}   ⚠ Could not build final SfM dataset output: {_bundle_exc}")

                    self._log_tagged("[OK]", f"{ds_tag} ✓ Dataset {ds_name} complete -> {ds_out}")
                    continue

                    # Move images/ into sparse/0/models/0/0/
                    shutil.move(str(ds_images), str(ds_models))
                    self._log_tagged("[CPU]", f"{ds_tag}   → images/ moved to {ds_models.relative_to(ds_out)}")
                    # Update sparse model so image paths point to new location
                    try:
                        from backend.colmap_images import read_images_bin as _read_images_bin, write_images_bin as _write_images_bin
                        _prefix = "models/0/0/images/"
                        _bin_path = ds_sfm / "images.bin"
                        if _bin_path.exists():
                            _images_data = _read_images_bin(_bin_path)
                            _prefixed = 0
                            for _img2 in _images_data:
                                if not _img2["name"].startswith(_prefix):
                                    _img2["name"] = _prefix + _img2["name"]
                                    _prefixed += 1
                            if _prefixed > 0:
                                _write_images_bin(_bin_path, _images_data)
                                self._log_tagged("[CPU]", f"{ds_tag}   → Sparse model updated: "
                                                 f"{_prefixed} image path(s) prefixed "
                                                 f"with '{_prefix}'")
                        else:
                            self._log_tagged("[CPU]", f"{ds_tag}   ⚠ images.bin not found")
                    except Exception as _exc2:
                        self._log_tagged("[CPU]", f"{ds_tag}   ⚠ Could not update sparse "
                                         f"model image paths: {_exc2}")

                    # Copy the updated .bin files into models/0/0/ so that directory
                    # contains the fully up-to-date reconstruction (EXR names + prefix),
                    # overwriting any stale SfM output that may have been left there.
                    for _bin_fname in ("cameras.bin", "images.bin", "points3D.bin", "frames.bin", "rigs.bin"):
                        _src_bin = ds_sfm / _bin_fname
                        if _src_bin.exists():
                            shutil.copy2(str(_src_bin), str(ds_models / _bin_fname))
                    try:
                        from backend.colmap_images import normalize_images_bin_for_image_dir
                        normalize_images_bin_for_image_dir(
                            ds_models / "images.bin",
                            ds_models / "images",
                            "images/",
                            lambda msg, _t=ds_tag: self._log_tagged("[CPU]", f"{_t}   {msg.strip()}"),
                        )
                    except Exception as _exc3:
                        self._log_tagged("[CPU]", f"{ds_tag}   ⚠ Could not normalize final images.bin: {_exc3}")
                    self._log_tagged("[CPU]", f"{ds_tag}   → Updated sparse .bin files copied to {ds_models.relative_to(ds_out)}")
                    # Clean up non-essential intermediate files and stale
                    # text-format sparse model files (only binary .bin files
                    # are kept up-to-date by pycolmap; the .txt versions
                    # written by stray_to_colmap still reference .png names).
                    for _fname in ["database.db",
                                   "cameras.txt", "images.txt", "points3D.txt"]:
                        _f = ds_sfm / _fname
                        if _f.exists():
                            try:
                                _f.unlink()
                            except PermissionError:
                                self._log_tagged("[CPU]", f"{ds_tag}   ⚠ Could not remove {_f.name} (file in use) — skipping")
                        # Remove stale SfM output in models/0/ (GLOMAP copies)
                        _sfm_model0 = ds_sfm / "models" / "0"
                        if _sfm_model0.is_dir():
                            for _stale in ("cameras.bin", "images.bin", "points3D.bin", "frames.bin", "rigs.bin"):
                                _sf = _sfm_model0 / _stale
                                if _sf.exists():
                                    try:
                                        _sf.unlink()
                                    except PermissionError:
                                        self._log_tagged("[CPU]", f"{ds_tag}   ⚠ Could not remove {_sf.name} (file in use) — skipping")
                        for _log_f in ds_sfm.glob("colmap.LOG*"):
                            try:
                                _log_f.unlink()
                            except PermissionError:
                                self._log_tagged("[CPU]", f"{ds_tag}   ⚠ Could not remove {_log_f.name} (file in use) — skipping")
                        self._log_tagged("[CPU]", f"{ds_tag}   → Intermediate files cleaned up from sparse/0/")

                    self._log_tagged("[OK]", f"{ds_tag} ✓ Dataset {ds_name} complete → {ds_out}")

                if total_ds > 1:
                    for _empty_name in ("images", "hloc_outputs", "sparse"):
                        _empty_dir = base_out / _empty_name
                        try:
                            if _empty_dir.is_dir() and not any(_empty_dir.iterdir()):
                                _empty_dir.rmdir()
                        except Exception:
                            pass

                self._log_tagged("[OK]", f"\n✓ SUCCESS — {total_ds} dataset(s) processed independently in: {base_out}")
                self._finish(success=True)
                return

            elif not is_video_mode:
                # --- IMAGES MODE ---
                video = Path(self.video_path.get())
                if not str(video) or str(video) == ".":
                    raise ValueError("No input selected!")
                self._set_step(0)
                self._log_tagged("[CPU]", f"[1/5] Image Mode: Copying from {video.name}...")
                if not video.is_dir():
                     raise ValueError(f"Input path is not a directory: {video}")
                
                input_images = sorted([
                    p for p in video.iterdir() 
                    if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
                ])
                
                if not input_images:
                    raise ValueError(f"No images (jpg/png) found in {video}")

                self._log_tagged("[CPU]", f"       → {len(input_images)} images found")
                
                # Copy images
                for img_path in input_images:
                    shutil.copy2(img_path, images_dir / img_path.name)
                
                self._log_tagged("[CPU]", "       ✓ Images copied")
            else:
                # --- VIDEO MODE (multi-video) ---
                # Auto-detect: skip extraction if images already exist
                exts_check = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]
                existing_imgs = 0
                for ext in exts_check:
                    existing_imgs += len(list(images_dir.glob(ext)))
                
                if existing_imgs > 0:
                    self._set_step(0)
                    self._log_tagged("[CPU]", f"[1/5] FFmpeg skipped — {existing_imgs} images already present")
                else:
                    videos = self.video_paths
                    if not videos:
                        raise ValueError("No video selected!")
                    
                    for v in videos:
                        if not v.is_file():
                            raise ValueError(f"Video file does not exist: {v}")

                    total_videos = len(videos)
                    
                    # Detect CUDA hardware acceleration for FFmpeg
                    use_cuda = False
                    try:
                        probe = subprocess.run(
                            ["ffmpeg", "-hwaccels"], capture_output=True, text=True, timeout=5
                        )
                        if "cuda" in probe.stdout.lower():
                            use_cuda = True
                    except Exception:
                        pass
                    
                    accel_tag = "[GPU]" if use_cuda else "[CPU]"
                    # Determine 16-bit mode: auto-detect from color settings or force toggle
                    use_16bit = self.force_16bit.get()
                    if not use_16bit and self.color_enabled.get():
                        dest = self.color_dest.get()
                        # Auto-enable 16-bit for destinations that need linear precision
                        if dest in ("ACEScg (EXR + sRGB PNG)", "Linear sRGB"):
                            use_16bit = True
                        elif dest == "Custom OCIO..." and self.has_ocio_lib:
                            use_16bit = _ocio_needs_16bit(self.ocio_out_cs.get())
                    
                    self._log_tagged(accel_tag, f"[1/5] Extracting {total_videos} video(s) at {self.fps_extract.get():.1f} FPS...")
                    if use_cuda:
                        self._log_tagged("[GPU]", "       → CUDA acceleration enabled (GPU decoding)")
                    if use_16bit:
                        self._log_tagged(accel_tag, "       → 16-bit PNG output enabled")
                    
                    for vid_idx, video in enumerate(videos, start=1):
                        self._check_cancelled()
                        prefix = f"vid{vid_idx:02d}"
                        self._log_tagged(accel_tag, f"       → [{vid_idx}/{total_videos}] {video.name} (prefix: {prefix}_)")
                        
                        # Build FFmpeg command with optional CUDA hwaccel
                        hwaccel_args = ["-hwaccel", "cuda"] if use_cuda else []
                        
                        # Find native_fps to compute step (to avoid VFR sync issues)
                        vid_info = next((info for info in self._video_infos if info["path"] == str(video)), None)
                        native_fps = vid_info["native_fps"] if vid_info else 30.0
                        target_fps = max(float(self.fps_extract.get()), 0.01)
                        step = max(1, round(native_fps / target_fps))
                        self._log_tagged(
                            accel_tag,
                            f"         Sampling: native {native_fps:.2f} FPS -> target {target_fps:.1f} FPS (step={step})",
                        )
                        
                        vf_args = ["-vf", f"select='not(mod(n\\,{step}))',setpts=N/FRAME_RATE/TB", "-vsync", "vfr"] if step > 1 else []
                        
                        if use_16bit:
                            ext = "png"
                            cmd = [
                                "ffmpeg", "-y",
                                *hwaccel_args,
                                "-i", str(video),
                                *vf_args,
                                "-pix_fmt", "rgb48be",
                                str(images_dir / f"{prefix}_%04d.{ext}")
                            ]
                        else:
                            ext = "png"
                            cmd = [
                                "ffmpeg", "-y",
                                *hwaccel_args,
                                "-i", str(video),
                                *vf_args,
                                "-qscale:v", "2",
                                str(images_dir / f"{prefix}_%04d.{ext}")
                            ]

                        proc, stderr_lines, reader_thread = self._run_ffmpeg_streamed(cmd, accel_tag)
                        # Poll so we can kill FFmpeg immediately on cancel
                        while proc.poll() is None:
                            if self._cancelled:
                                proc.kill()
                                proc.wait()
                                reader_thread.join(timeout=2)
                                raise CancelledError("Processing cancelled by user")
                            time.sleep(0.25)
                        reader_thread.join(timeout=5)
                        
                        # Fallback: if CUDA failed on first video, retry without it
                        if proc.returncode != 0 and use_cuda and vid_idx == 1:
                            self._log_tagged("[CPU]", "       ⚠ CUDA failed, falling back to CPU...")
                            use_cuda = False
                            accel_tag = "[CPU]"
                            # Remove CUDA frames if any partial output
                            for f in images_dir.glob(f"{prefix}_*.{ext}"):
                                f.unlink()
                            
                            # Rebuild command without hwaccel
                            if use_16bit:
                                cmd = [
                                    "ffmpeg", "-y",
                                    "-i", str(video),
                                    *vf_args,
                                    "-pix_fmt", "rgb48be",
                                    str(images_dir / f"{prefix}_%04d.{ext}")
                                ]
                            else:
                                cmd = [
                                    "ffmpeg", "-y",
                                    "-i", str(video),
                                    *vf_args,
                                    "-qscale:v", "2",
                                    str(images_dir / f"{prefix}_%04d.{ext}")
                                ]
                            proc, stderr_lines, reader_thread = self._run_ffmpeg_streamed(cmd, accel_tag)
                            while proc.poll() is None:
                                if self._cancelled:
                                    proc.kill()
                                    proc.wait()
                                    reader_thread.join(timeout=2)
                                    raise CancelledError("Processing cancelled by user")
                                time.sleep(0.25)
                            reader_thread.join(timeout=5)
                        
                        if proc.returncode != 0:
                            stderr_out = '\n'.join(stderr_lines)
                            self._log_tagged("[ERR]", f"         FFmpeg error: {stderr_out[-500:] if stderr_out else 'unknown'}")
                            raise RuntimeError(f"FFmpeg failed on {video.name}")
                        
                        # Count extracted frames
                        num_frames = len(list(images_dir.glob(f"{prefix}_*.{ext}")))
                        self._log_tagged(accel_tag, f"         ✓ {num_frames} frames extracted")
                    
                    # Count total
                    total_imgs = len([f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
                    self._log_tagged(accel_tag, f"       ✓ Extraction complete — {total_imgs} images total")

            # --- 1.5 COLOR CONVERSION ---
            if self.color_enabled.get():
                color_step_idx = 1
                GUIProgressTqdm._step_index = color_step_idx
                GUIProgressTqdm._step_name = "Color Conversion"
                global_models_dir = sfm_dir / "models" / "0" / "0"
                global_models_dir.mkdir(parents=True, exist_ok=True)
                apply_color_conversion(images_dir, "[CPU]", f"[{color_step_idx+1}/{len(self.STEPS)}] Applying Color Conversion ({self.color_source.get()} → {self.color_dest.get()})...", exr_out_dir=global_models_dir)

            if not is_stray_mode:
                # ═══ SHARED PIPELINE (Video / Images modes only) ═══
                self._filter_frame_quality(images_dir, "[CPU]")
                self._apply_quality_sweep_sample(images_dir, "[CPU]")

                # 2. FEATURES
                self._check_cancelled()
                self._set_step(1)
                
                device_str = "[GPU]" if torch.cuda.is_available() else "[CPU]"
                gui_handler.set_tag(device_str)
                matcher_name = self.matcher_type.get()
                loma_pipeline = (
                    LoMaMatcher(
                        matcher_name,
                        max_keypoints=int(self.max_keypoints.get()),
                        logger=lambda msg, _d=device_str: self._log_tagged(_d, msg),
                        cancel_check=self._check_cancelled,
                    )
                    if is_loma_matcher(matcher_name)
                    else None
                )
                 
                features_out = loma_feature_path(outputs_dir, matcher_name) if loma_pipeline else outputs_dir / 'features.h5'
                if loma_pipeline is not None:
                    self._log_tagged(device_str, f"[2/5] LoMa feature cache - {matcher_name} (max {self.max_keypoints.get()})...")
                    features_path = loma_pipeline.extract_features(images_dir, features_out)
                    self._log_tagged(device_str, "       LoMa features ready")
                elif features_out.exists():
                    self._log_tagged(device_str, f"[2/5] Features skipped (already present: {features_out.name})")
                    features_path = features_out
                else:
                    self._log_tagged(device_str, f"[2/5] Features — {self.feature_type.get()} (max {self.max_keypoints.get()})...")
                    conf_feature = extract_features.confs[self.feature_type.get()]
                    conf_feature['model']['max_keypoints'] = int(self.max_keypoints.get())
                    features_path = extract_features.main(
                        conf_feature, images_dir, feature_path=features_out
                    )
                    self._log_tagged(device_str, "       ✓ Features extracted")

                # 3. PAIRS
                self._check_cancelled()
                self._set_step(2)
                gui_handler.set_tag("[CPU]")
                mode = self.pairing_mode.get()
                pairs_path = outputs_dir / 'pairs.txt'
                if pairs_path.exists():
                    _pair_count = count_pairs_file(pairs_path)
                    self._log_tagged("[CPU]", f"[3/5] Pairs skipped ({_pair_count} pairs already present)")
                else:
                    self._log_tagged("[CPU]", f"[3/5] Pairs — {mode}...")
                    if "Sequential" in mode:
                        _pair_count = generate_sequential_pairs(images_dir, pairs_path, overlap=20)
                    else:
                        pairs_from_exhaustive.main(pairs_path, image_list=None, features=features_path)
                        _pair_count = count_pairs_file(pairs_path)
                    self._log_tagged("[CPU]", f"       ✓ {_pair_count} pairs generated")
                self.after(0, lambda _n=_pair_count: self.step_label.configure(
                    text=f"Pairs: {_n} pairs (100%)",
                    text_color=COLORS["accent_blue"],
                ))

                # 4. MATCHING
                self._check_cancelled()
                self._set_step(3)
                gui_handler.set_tag(device_str) # Re-use GPU/CPU tag
                matches_out = loma_matches_path(outputs_dir, matcher_name) if loma_pipeline else outputs_dir / 'matches.h5'
                if loma_pipeline is not None:
                    self._log_tagged(device_str, f"[4/5] Matching - {matcher_name}...")
                    matches_path = loma_pipeline.match_pairs(pairs_path, features_path, matches_out)
                    self._log_tagged(device_str, "       LoMa matching complete")
                elif matches_out.exists():
                    self._log_tagged(device_str, f"[4/5] Matching skipped (already present: {matches_out.name})")
                    matches_path = matches_out
                else:
                    self._log_tagged(device_str, f"[4/5] Matching — {self.matcher_type.get()}...")
                    conf_match = match_features.confs[self.matcher_type.get()]
                    matches_path = match_features.main(
                        conf_match, pairs_path, features=features_path, matches=matches_out
                    )
                    self._log_tagged(device_str, "       ✓ Matching complete")

                # 5. SFM / TRIANGULATION
                self._check_cancelled()
                self._set_step(4)



                cam_map = {
                    "PINHOLE": "PINHOLE", "OPENCV": "OPENCV",
                    "SIMPLE_RADIAL": "SIMPLE_RADIAL", "OPENCV_FISHEYE": "OPENCV_FISHEYE"
                }

                # Standard SfM pipeline
                sfm_tag = "[GPU]" if self.mapper_type.get() == "GLOMAP" else "[CPU/GPU]"
                gui_handler.set_tag(sfm_tag)

                cam_model = cam_map[self.camera_model.get()]
                self._log_tagged(sfm_tag, f"[5/5] Reconstruction SfM — {self.mapper_type.get()} ({cam_model})...")

                def tagged_logger(msg):
                    self._log(f"{sfm_tag} {msg}")

                run_sfm_with_live_export(
                    sfm_dir=sfm_dir,
                    image_dir=images_dir,
                    pairs=pairs_path,
                    features=features_path,
                    matches=matches_path,
                    camera_mode=pycolmap.CameraMode.AUTO,
                    camera_model=cam_model,
                    mapper_type=self.mapper_type.get(),
                    shared_dir=base_out / "live_reconstruction",
                    export_every=5,
                    cancel_check=self._check_cancelled,
                    logger=tagged_logger,
                    num_threads=self.num_workers.get(),
                )
                self._log_tagged(sfm_tag, "       ✓ Reconstruction complete")

                try:
                    normalize_final_bundle(
                        base_out,
                        keep_srgb_png=True,
                        use_acescg_exr=self.use_acescg_exr.get(),
                    )
                    self._log_tagged("[CPU]", "       -> Final bundle normalized; images.bin patched")
                except Exception as _bundle_exc:
                    self._log_tagged("[CPU]", f"       ⚠ Could not normalize final bundle: {_bundle_exc}")

                self._log_tagged("[OK]", f"\n✓ SUCCESS — Dataset ready: {base_out}")
                self._finish(success=True)

        except CancelledError:
            self._log_tagged("[INFO]", "⏹ Processing cancelled.")
            self._finish(cancelled=True)
        except Exception as e:
            # First restore stdout/stderr to ensure we can print if needed, 
            # though we primarily use self._log_tagged which writes to GUI
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            import traceback
            traceback.print_exc() # Print to terminal for debugging
            self._log_tagged("[ERR]", f"\n❌ ERROR: {e}")
            self._finish(success=False)
        finally:
            # Restore everything
            if hloc_logger and gui_handler:
                hloc_logger.removeHandler(gui_handler)
            
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            if old_tqdm:
                tqdm_module.tqdm = old_tqdm
                for mod in [extract_features, match_features, reconstruction]:
                    if hasattr(mod, 'tqdm'):
                        mod.tqdm = old_tqdm
            
            GUIProgressTqdm._gui_app = None
            pass  # Cleanup complete


if __name__ == "__main__":
    app = SfMApp()
    app.mainloop()
