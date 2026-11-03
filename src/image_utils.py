"""Image helpers: metadata, JPEG quality, hashing, preprocessing, pixel/FFT stats."""
# Refs: Grommelt et al. 2024 "Fake or JPEG?", arXiv 2403.17608 (why we re-encode everything)
from __future__ import annotations

import hashlib
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None  # big Midjourney images

RESAMPLE = {"bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR,
            "lanczos": Image.LANCZOS, "nearest": Image.NEAREST}

# standard libjpeg luminance table (ITU-T T.81, Annex K)
_STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99], dtype=np.float64)
# zigzag -> natural order
_ZIGZAG = np.array([
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63])


def _ijg_table(q: int) -> np.ndarray:
    """libjpeg luminance table for quality q."""
    scale = 5000 / q if q < 50 else 200 - 2 * q
    return np.clip(np.floor((_STD_LUMA * scale + 50) / 100), 1, 255)


_IJG_TABLES = {q: _ijg_table(q) for q in range(1, 101)}


# Opening and metadata
def safe_open(path: str | Path) -> tuple[Image.Image | None, str | None]:
    """Open + fully decode, returns (img, error)."""
    try:
        im = Image.open(path)
        # load() catches truncated files; no copy() since it drops .format/.quantization
        im.load()
        return im, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def estimate_jpeg_quality(img: Image.Image) -> tuple[float, bool]:
    """JPEG quality (1-100) from the quant table, returns (q, exact_match)."""
    qt = getattr(img, "quantization", None)
    if not qt:
        return float("nan"), False
    table = np.asarray(qt[min(qt.keys())], dtype=np.float64)
    if table.size != 64:
        return float("nan"), False
    best_q, best_err = None, np.inf
    for q, ref in _IJG_TABLES.items():
        # table order differs between Pillow versions
        err = min(np.abs(table - ref).mean(), np.abs(table - ref[_ZIGZAG]).mean())
        if err < best_err:
            best_q, best_err = q, err
    return float(best_q), bool(best_err == 0)


def md5_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def phash(img: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """64-bit DCT perceptual hash."""
    # Refs: Zauner 2010, perceptual image hashing
    from scipy.fft import dctn
    n = hash_size * highfreq_factor
    g = np.asarray(img.convert("L").resize((n, n), Image.LANCZOS), dtype=np.float64)
    low = dctn(g, norm="ortho")[:hash_size, :hash_size]
    bits = (low > np.median(low)).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def is_grayscale(img: Image.Image, tol: int = 2) -> bool:
    """1-channel, or RGB with equal channels."""
    if img.mode in ("L", "LA", "1", "I", "I;16", "F"):
        return True
    small = np.asarray(img.convert("RGB").resize((64, 64)), dtype=np.int16)
    return bool(np.abs(small[..., 0] - small[..., 1]).max() <= tol and
                np.abs(small[..., 1] - small[..., 2]).max() <= tol)


def image_metadata(path: str | Path, with_hashes: bool = True, hash_size: int = 8) -> dict:
    """Format, size, mode, bytes, JPEG quality, hashes for one file."""
    path = Path(path)
    info = {"readable": False, "error": None, "real_format": None, "width": np.nan,
            "height": np.nan, "mode": None, "file_bytes": path.stat().st_size if path.exists() else np.nan,
            "jpeg_quality": np.nan, "jpeg_q_exact": False, "is_gray": None,
            "md5": None, "phash": None}
    img, err = safe_open(path)
    if img is None:
        info["error"] = err
        return info
    q, exact = estimate_jpeg_quality(img)
    info.update(readable=True, real_format=img.format, width=img.width, height=img.height,
                mode=img.mode, jpeg_quality=q, jpeg_q_exact=exact, is_gray=is_grayscale(img))
    if with_hashes:
        info["md5"] = md5_file(path)
        info["phash"] = format(phash(img, hash_size), "016x")
    return info


def metadata_many(paths: list, n_threads: int = 8, progress: bool = True, **kw) -> list[dict]:
    """image_metadata over many files (threaded)."""
    from tqdm.auto import tqdm
    with ThreadPoolExecutor(n_threads) as ex:
        return list(tqdm(ex.map(lambda p: image_metadata(p, **kw), paths), total=len(paths),
                         desc="metadata", disable=not progress))


# Duplicates
def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def near_duplicate_pairs(hashes_hex: list[str], threshold: int, chunk: int = 2000) -> list[tuple]:
    """(i, j, dist) pairs with Hamming distance <= threshold."""
    n = len(hashes_hex)
    if n < 2:
        return []
    ints = np.array([int(h, 16) for h in hashes_hex], dtype=np.uint64)
    bits = np.unpackbits(ints.view(np.uint8).reshape(n, 8), axis=1).astype(np.float32)
    inv = 1.0 - bits
    pairs = []
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = bits[s:e] @ inv.T + inv[s:e] @ bits.T  # hamming, (chunk, n)
        ii, jj = np.nonzero(d <= threshold)
        for i, j in zip(ii + s, jj):
            if i < j:
                pairs.append((int(i), int(j), int(d[i - s, j])))
    return pairs


# Preprocessing (same for real and fake)
def center_crop_square(img: Image.Image) -> Image.Image:
    """Largest centered square."""
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def preprocess_image(img: Image.Image, size: int = 224, resample: str = "bicubic") -> Image.Image:
    """RGB -> center square crop -> resize to size x size."""
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        # transparency -> white
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img)
        img = img.convert("RGB")
    img = center_crop_square(img)
    if img.size != (size, size):
        img = img.resize((size, size), RESAMPLE[resample])
    return img


def save_jpeg(img: Image.Image, path: str | Path, quality: int = 96, subsampling: str = "4:2:0") -> None:
    """Save JPEG with fixed settings, no EXIF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, subsampling=subsampling, optimize=False)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(buf.getvalue())
    tmp.replace(path)


def process_file(src: str | Path, dst: str | Path, size: int = 224, quality: int = 96,
                 subsampling: str = "4:2:0", resample: str = "bicubic") -> str | None:
    """Preprocess one file, returns an error string or None."""
    if Path(dst).exists():
        return None  # already done
    img, err = safe_open(src)
    if img is None:
        return err
    try:
        save_jpeg(preprocess_image(img, size, resample), dst, quality, subsampling)
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def process_many(pairs: list[tuple], n_threads: int = 8, progress: bool = True, **kw) -> list:
    """process_file over (src, dst) pairs (threaded)."""
    from tqdm.auto import tqdm
    with ThreadPoolExecutor(n_threads) as ex:
        return list(tqdm(ex.map(lambda p: process_file(p[0], p[1], **kw), pairs),
                         total=len(pairs), desc="preprocess", disable=not progress))


# EDA stats
def pixel_stats(img: Image.Image, max_side: int = 256) -> dict:
    """Brightness, contrast, saturation, channel means."""
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    a = np.asarray(img, dtype=np.float32) / 255.0
    gray = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
    return {"brightness": float(gray.mean()), "contrast": float(gray.std()),
            "saturation": float(hsv[..., 1].mean()),
            "mean_r": float(a[..., 0].mean()), "mean_g": float(a[..., 1].mean()),
            "mean_b": float(a[..., 2].mean())}


def channel_histograms(img: Image.Image, bins: int = 64, max_side: int = 256) -> np.ndarray:
    """RGB histograms, shape (3, bins)."""
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    a = np.asarray(img)
    return np.stack([np.histogram(a[..., c], bins=bins, range=(0, 256), density=True)[0]
                     for c in range(3)])


def log_fft_spectrum(img: Image.Image, crop: int = 128) -> np.ndarray | None:
    """Log-magnitude FFT of a gray center crop (None if image is too small)."""
    # Refs: Frank et al. 2020, arXiv 2003.08685
    g = img.convert("L")
    if min(g.size) < crop:
        return None
    w, h = g.size
    left, top = (w - crop) // 2, (h - crop) // 2
    a = np.asarray(g.crop((left, top, left + crop, top + crop)), dtype=np.float64)
    a = a - a.mean()
    f = np.fft.fftshift(np.fft.fft2(a))
    return np.log(np.abs(f) + 1e-8)
