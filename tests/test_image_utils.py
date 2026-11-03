"""Tests for the preprocessing pipeline, JPEG quality estimate and hashing on synthetic images."""
import numpy as np
from PIL import Image

from src.image_utils import (center_crop_square, estimate_jpeg_quality, image_metadata,
                             log_fft_spectrum, near_duplicate_pairs, phash, preprocess_image,
                             process_file, safe_open)


def _synthetic(w=300, h=200, seed=0, mode="RGB"):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([(xx * 255 / w), (yy * 255 / h), ((xx + yy) % 255)], -1)
    noise = rng.normal(0, 10, base.shape)
    return Image.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8)).convert(mode)


def test_center_crop_square():
    assert center_crop_square(_synthetic(300, 200)).size == (200, 200)
    assert center_crop_square(_synthetic(100, 250)).size == (100, 100)


def test_preprocess_and_jpeg_pipeline(tmp_path):
    src = tmp_path / "fake.png"
    _synthetic(300, 200).save(src)
    dst = tmp_path / "out" / "fake.jpg"
    assert process_file(src, dst, size=224, quality=96) is None
    img, err = safe_open(dst)
    assert err is None and img.format == "JPEG" and img.size == (224, 224) and img.mode == "RGB"
    q, exact = estimate_jpeg_quality(img)
    assert q == 96 and exact
    assert process_file(src, dst) is None            # second run is a no-op (resumable)


def test_preprocess_handles_modes():
    for mode in ["L", "RGBA", "P", "CMYK"]:
        out = preprocess_image(_synthetic(150, 128, mode=mode), size=224)
        assert out.mode == "RGB" and out.size == (224, 224)


def test_jpeg_quality_estimate_various(tmp_path):
    for q in [50, 75, 90, 100]:
        p = tmp_path / f"q{q}.jpg"
        _synthetic().save(p, quality=q)
        est, exact = estimate_jpeg_quality(safe_open(p)[0])
        assert est == q and exact
    p = tmp_path / "x.png"
    _synthetic().save(p)
    est, _ = estimate_jpeg_quality(safe_open(p)[0])
    assert np.isnan(est)


def test_metadata_and_corrupt_file(tmp_path):
    p = tmp_path / "a.png"
    _synthetic(320, 240).save(p)
    md = image_metadata(p)
    assert md["readable"] and md["real_format"] == "PNG"
    assert (md["width"], md["height"]) == (320, 240)
    assert len(md["md5"]) == 32 and len(md["phash"]) == 16
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    md = image_metadata(bad)
    assert not md["readable"] and md["error"]


def test_phash_near_duplicates():
    a = _synthetic(256, 256, seed=1)
    a_small = a.resize((128, 128))                                 # same picture, resized
    b = _synthetic(256, 256, seed=2).transpose(Image.ROTATE_90)    # different picture
    hashes = [format(phash(x), "016x") for x in (a, a_small, b)]
    pairs = [(i, j) for i, j, _ in near_duplicate_pairs(hashes, threshold=4)]
    assert (0, 1) in pairs
    assert (0, 2) not in pairs


def test_fft_spectrum_shape_and_small_image():
    s = log_fft_spectrum(_synthetic(300, 200), crop=128)
    assert s.shape == (128, 128) and np.isfinite(s).all()
    assert log_fft_spectrum(_synthetic(100, 100), crop=128) is None
