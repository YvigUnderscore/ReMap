from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

import cv2
import numpy as np

try:
    import OpenImageIO as oiio

    HAS_OCIO = True
except ImportError:
    oiio = None
    HAS_OCIO = False


ACESCG_OCIO_SPACE = "ACES - ACEScg"
LINEAR_ACESCG_SOURCE = "Linear ACEScg"
_AUTO_SOURCE = "Auto-detect"
ROOT_DIR = Path(__file__).resolve().parent.parent
APPLE_LOG_TO_LIN_LUT = ROOT_DIR / "LUTS" / "AppleLogToLin-v1.0.cube"

_MAT_BT2020_TO_ACESCG = np.array(
    [
        [0.97990525, 0.02225227, -0.03192382],
        [-0.00058388, 0.99476128, 0.01081350],
        [0.00046861, 0.01941638, 1.06066918],
    ],
    dtype=np.float32,
)

_MAT_ACESCG_TO_SRGB = np.array(
    [
        [1.70298067, -0.62451279, -0.03670953],
        [-0.12985749, 1.14073295, -0.01436027],
        [-0.02069324, -0.12236011, 1.05442752],
    ],
    dtype=np.float32,
)

_MAT_BT2020_TO_SRGB = np.array(
    [
        [1.66022663, -0.58754766, -0.07283817],
        [-0.12455332, 1.13292610, -0.00834968],
        [-0.01815514, -0.10060303, 1.11899821],
    ],
    dtype=np.float32,
)

_MAT_SRGB_TO_ACESCG = np.array(
    [
        [0.61590865, 0.34031053, 0.01410133],
        [0.06855723, 0.91546150, 0.02095702],
        [0.01902455, 0.11135884, 0.94994319],
    ],
    dtype=np.float32,
)


def _apply_matrix(rgb: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    orig_shape = rgb.shape
    flat = rgb.reshape(-1, 3)
    result = np.dot(flat, matrix.T)
    return result.reshape(orig_shape)


def _apple_log_to_linear(values: np.ndarray) -> np.ndarray:
    r_cut = 0.00104
    a = 5.555556
    b = 0.047996
    c = 0.529136
    d = 0.089004
    e_lin = 10.444689
    f = 0.180395
    e_cut = e_lin * r_cut + f

    encoded = values.astype(np.float32)
    linear = np.where(
        encoded >= e_cut,
        (np.power(2.0, (encoded - d) / c) - b) / a,
        (encoded - f) / e_lin,
    )
    return np.maximum(linear, 0.0)


@lru_cache(maxsize=2)
def _load_cube_1d(path_str: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(path_str)
    domain_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    domain_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    rows: list[list[float]] = []
    expected_size: int | None = None

    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            upper = line.upper()
            if upper.startswith("TITLE"):
                continue
            if upper.startswith("LUT_1D_SIZE"):
                expected_size = int(line.split()[1])
                continue
            if upper.startswith("DOMAIN_MIN"):
                domain_min = np.array([float(v) for v in line.split()[1:4]], dtype=np.float32)
                continue
            if upper.startswith("DOMAIN_MAX"):
                domain_max = np.array([float(v) for v in line.split()[1:4]], dtype=np.float32)
                continue
            parts = line.split()
            if len(parts) >= 3:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])

    table = np.asarray(rows, dtype=np.float32)
    if table.ndim != 2 or table.shape[1] != 3:
        raise ValueError(f"Invalid 1D LUT: {path}")
    if expected_size is not None and table.shape[0] != expected_size:
        raise ValueError(f"Invalid 1D LUT size: expected {expected_size}, got {table.shape[0]}")
    return table, domain_min, domain_max


def _apply_cube_1d(values: np.ndarray, path: Path) -> np.ndarray:
    table, domain_min, domain_max = _load_cube_1d(str(path))
    rgb = np.asarray(values, dtype=np.float32)
    out = np.empty_like(rgb, dtype=np.float32)
    x = np.linspace(0.0, 1.0, table.shape[0], dtype=np.float32)
    denom = np.maximum(domain_max - domain_min, 1e-7)
    normalized = np.clip((rgb - domain_min) / denom, 0.0, 1.0)
    for channel in range(3):
        out[..., channel] = np.interp(normalized[..., channel], x, table[:, channel])
    return np.maximum(out, 0.0)


def _hlg_eotf(values: np.ndarray) -> np.ndarray:
    a = 0.17883277
    b = 1.0 - 4.0 * a
    c = 0.5 - a * np.log(4.0 * a)
    encoded = np.asarray(values, dtype=np.float32)
    return np.where(
        encoded <= 0.5,
        (encoded**2) / 3.0,
        (np.exp((encoded - c) / a) + b) / 12.0,
    )


def _srgb_eotf(values: np.ndarray) -> np.ndarray:
    encoded = np.asarray(values, dtype=np.float32)
    return np.where(
        encoded <= 0.04045,
        encoded / 12.92,
        np.power(np.maximum((encoded + 0.055) / 1.055, 0.0), 2.4),
    )


def _srgb_oetf(values: np.ndarray) -> np.ndarray:
    values = np.maximum(values, 0.0)
    return np.where(
        values <= 0.0031308,
        12.92 * values,
        1.055 * np.power(np.maximum(values, 1e-7), 1 / 2.4) - 0.055,
    )


def _aces_tonemap(values: np.ndarray) -> np.ndarray:
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return np.clip((values * (a * values + b)) / (values * (c * values + d) + e), 0.0, 1.0)


def _linearize(rgb_norm: np.ndarray, source: str) -> np.ndarray:
    source = _canonical_source(source)
    if source in ("Linear BT.2020", "Linear sRGB"):
        return rgb_norm
    if source == LINEAR_ACESCG_SOURCE:
        return rgb_norm
    if source == "Apple Log (BT.2020)":
        if APPLE_LOG_TO_LIN_LUT.exists():
            return _apply_cube_1d(rgb_norm, APPLE_LOG_TO_LIN_LUT)
        return _apple_log_to_linear(rgb_norm)
    if source == "HLG (BT.2020)":
        return _hlg_eotf(rgb_norm)
    if source == "sRGB (Rec.709)":
        return _srgb_eotf(rgb_norm)
    return rgb_norm


def _source_primaries(source: str) -> str:
    source = _canonical_source(source)
    if source in ("Linear BT.2020", "Apple Log (BT.2020)", "HLG (BT.2020)"):
        return "bt2020"
    if source == LINEAR_ACESCG_SOURCE:
        return "acescg"
    return "srgb"


def _gamut_convert(rgb_linear: np.ndarray, src_primaries: str, dst_primaries: str) -> np.ndarray:
    matrices = {
        ("bt2020", "acescg"): _MAT_BT2020_TO_ACESCG,
        ("bt2020", "srgb"): _MAT_BT2020_TO_SRGB,
        ("srgb", "acescg"): _MAT_SRGB_TO_ACESCG,
        ("acescg", "srgb"): _MAT_ACESCG_TO_SRGB,
    }
    matrix = matrices.get((src_primaries, dst_primaries))
    if matrix is not None:
        return _apply_matrix(rgb_linear, matrix)
    if src_primaries == dst_primaries:
        return rgb_linear
    return rgb_linear


def _canonical_source(source: str | None) -> str:
    normalized = (source or "").strip()
    lower = normalized.lower()
    if lower in {"acescg", "aces - acescg", "linear acescg", "lin_ap1", "ap1"}:
        return LINEAR_ACESCG_SOURCE
    if lower in {"linear", "linear srgb", "lin_srgb"}:
        return "Linear sRGB"
    if lower in {"srgb", "rec709", "rec.709", "rec. 709", "bt709"}:
        return "sRGB (Rec.709)"
    if lower in {"linear bt.2020", "linear bt2020", "bt2020"}:
        return "Linear BT.2020"
    return normalized or "Linear sRGB"


def _source_from_oiio_metadata(path: Path, fallback: str | None = None) -> str:
    if fallback and fallback != _AUTO_SOURCE:
        return _canonical_source(fallback)
    if HAS_OCIO:
        try:
            buf = oiio.ImageBuf(str(path))
            spec = buf.spec()
            metadata = " ".join(
                str(spec.getattribute(name) or "")
                for name in ("oiio:ColorSpace", "ColorSpace", "colorspace", "chromaticities")
            ).lower()
            if "acescg" in metadata or "aces - acescg" in metadata or "ap1" in metadata:
                return LINEAR_ACESCG_SOURCE
            if "2020" in metadata:
                return "Linear BT.2020"
            if "srgb" in metadata or "rec.709" in metadata or "rec709" in metadata or "bt709" in metadata:
                return "Linear sRGB"
        except Exception:
            pass
    if path.suffix.lower() == ".exr":
        return LINEAR_ACESCG_SOURCE
    return _canonical_source(fallback)


def _read_rgb_float(path: Path) -> tuple[np.ndarray, str]:
    """Read an image as RGB float values without crushing EXR HDR range."""
    if HAS_OCIO and path.suffix.lower() == ".exr":
        buf = oiio.ImageBuf(str(path))
        if buf.has_error:
            raise ValueError(f"Could not read image: {buf.geterror()}")
        pixels = buf.get_pixels(oiio.FLOAT)
        if pixels.ndim == 2:
            pixels = pixels[..., None]
        if pixels.shape[2] == 1:
            pixels = np.repeat(pixels, 3, axis=2)
        return pixels[..., :3].astype(np.float32, copy=False), "RGB"

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Failed to read image")
    if len(img.shape) < 3 or img.shape[2] < 3:
        raise ValueError("Unsupported channels")

    code = cv2.COLOR_BGR2RGB if img.shape[2] == 3 else cv2.COLOR_BGRA2RGB
    rgb = cv2.cvtColor(img, code).astype(np.float32)
    if np.issubdtype(img.dtype, np.floating):
        return rgb, "RGB"
    if img.dtype == np.uint16:
        return rgb / 65535.0, "RGB"
    if img.dtype == np.uint8:
        return rgb / 255.0, "RGB"
    max_val = float(np.iinfo(img.dtype).max) if np.issubdtype(img.dtype, np.integer) else 1.0
    return rgb / max_val, "RGB"


def _write_exr(path: Path, rgb_linear: np.ndarray, colorspace: str) -> None:
    rgb = np.asarray(rgb_linear, dtype=np.float32)
    if HAS_OCIO:
        h, w, ch = rgb.shape
        spec = oiio.ImageSpec(w, h, ch, oiio.FLOAT)
        spec.attribute("oiio:ColorSpace", colorspace)
        out = oiio.ImageBuf(spec)
        out.set_pixels(oiio.ROI(), rgb)
        out.write(str(path))
        if out.has_error:
            raise ValueError(out.geterror())
        return
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise ValueError(f"Could not write {path.name}")


def _sfm_proxy_display_rgb(srgb_linear: np.ndarray) -> np.ndarray:
    """Build a display-referred sRGB proxy that keeps HDR detail useful for SfM."""
    linear = np.maximum(np.asarray(srgb_linear, dtype=np.float32), 0.0)
    finite = np.isfinite(linear)
    if not finite.all():
        linear = np.where(finite, linear, 0.0)

    luminance = 0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]
    positive = luminance[np.isfinite(luminance) & (luminance > 1e-6)]
    if positive.size:
        p95 = float(np.percentile(positive, 95.0))
        max_luma = float(np.max(positive))
        if max_luma > 1.0 and p95 > 1e-6:
            linear = linear * min(1.0, 1.0 / p95)

    mapped = linear / (1.0 + linear)
    return np.clip(_srgb_oetf(mapped), 0.0, 1.0)


def _write_srgb_png(path: Path, rgb_display: np.ndarray) -> None:
    rgb_16 = np.clip(rgb_display * 65535.0, 0, 65535).astype(np.uint16)
    bgr_16 = cv2.cvtColor(rgb_16, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr_16):
        raise ValueError(f"Could not write {path.name}")


def write_sfm_proxy_png(
    src_path: str | Path,
    dst_path: str | Path,
    source_space: str | None = _AUTO_SOURCE,
) -> str:
    """Convert an HDR/scene-linear source to a tone-mapped sRGB PNG proxy for SfM."""
    src = Path(src_path)
    dst = Path(dst_path)
    resolved_source = _source_from_oiio_metadata(src, source_space)
    rgb, _ = _read_rgb_float(src)
    linear_rgb = _linearize(rgb, resolved_source)
    srgb_linear = _gamut_convert(linear_rgb, _source_primaries(resolved_source), "srgb")
    _write_srgb_png(dst, _sfm_proxy_display_rgb(srgb_linear))
    return resolved_source


def process_image_color_worker(
    img_path_str: str,
    source_space: str,
    dest_space: str,
    cs_in: str | None,
    cs_out: str | None,
    colorconfig_path: str | None,
    exr_out_dir_str: str | None,
) -> tuple[bool, str | None]:
    img_path = Path(img_path_str)

    if dest_space == "Custom OCIO..." and HAS_OCIO:
        try:
            buf = oiio.ImageBuf(str(img_path))
            if not buf.has_error:
                result = oiio.ImageBufAlgo.colorconvert(
                    buf,
                    buf,
                    cs_in,
                    cs_out,
                    colorconfig=colorconfig_path or "",
                )
                if result:
                    buf.write(str(img_path))
                    return True, None
        except Exception as exc:
            return False, str(exc)
        return False, "OCIO Error"

    try:
        source_space = _source_from_oiio_metadata(img_path, source_space)
        img_rgb, _ = _read_rgb_float(img_path)
        h, w, ch = img_rgb.shape
        linear_rgb = _linearize(img_rgb, source_space)
        src_prim = _source_primaries(source_space)
        input_is_exr = img_path.suffix.lower() == ".exr"

        if dest_space == "ACEScg (EXR + sRGB PNG)":
            acescg_rgb = _gamut_convert(linear_rgb, src_prim, "acescg")
            if exr_out_dir_str:
                os.makedirs(exr_out_dir_str, exist_ok=True)
                out_exr = str(Path(exr_out_dir_str) / f"{img_path.stem}.exr")
            else:
                out_exr = str(img_path).rsplit(".", 1)[0] + ".exr"

            _write_exr(Path(out_exr), acescg_rgb, ACESCG_OCIO_SPACE)

            if input_is_exr:
                return True, None

            srgb_linear = _gamut_convert(linear_rgb, src_prim, "srgb")
            _write_srgb_png(img_path, _sfm_proxy_display_rgb(srgb_linear))
            return True, None

        if dest_space == "sRGB (Tone Mapped)":
            srgb_linear = _gamut_convert(linear_rgb, src_prim, "srgb")
            if input_is_exr:
                _write_exr(img_path, srgb_linear, "Linear")
            else:
                _write_srgb_png(img_path, _sfm_proxy_display_rgb(srgb_linear))
            return True, None

        if dest_space == "Linear sRGB":
            srgb_linear = _gamut_convert(linear_rgb, src_prim, "srgb")
            if input_is_exr:
                _write_exr(img_path, srgb_linear, "Linear")
            else:
                srgb_16 = np.clip(srgb_linear * 65535, 0, 65535).astype(np.uint16)
                srgb_bgr = cv2.cvtColor(srgb_16, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(img_path), srgb_bgr)
            return True, None

        return False, f"Unknown destination: {dest_space}"
    except Exception as exc:
        return False, str(exc)
