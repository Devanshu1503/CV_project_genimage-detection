"""Image degradations for Problem 2: fixed test conditions + random training augmentation."""
# Refs: Wang et al. 2020 (blur + JPEG augmentation), arXiv 1912.11035
from __future__ import annotations

import hashlib
import io
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from .datasets import GenImageDataset


def stable_seed(*parts) -> int:
    """Same int for the same inputs (python hash() is salted per process)."""
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality), subsampling="4:2:0")
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def resize_roundtrip(img: Image.Image, scale: float) -> Image.Image:
    """Resize by scale then back to the original size, so every model still gets 224 px."""
    w, h = img.size
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    return img.resize((nw, nh), Image.BICUBIC).resize((w, h), Image.BICUBIC)


def blur(img: Image.Image, sigma: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def noise(img: Image.Image, std: float, seed: int = 0) -> Image.Image:
    """Gaussian pixel noise, std in 0-255 units."""
    arr = np.asarray(img, dtype=np.float32)
    arr = arr + np.random.default_rng(seed).normal(0, std, arr.shape)
    return Image.fromarray(arr.clip(0, 255).round().astype(np.uint8))


def apply_steps(img: Image.Image, steps, seed: int = 0) -> Image.Image:
    """steps = [[kind, level], ...] applied in order, e.g. [["resize", 0.5], ["jpeg", 75]]."""
    for kind, level in steps or []:
        if kind == "jpeg":
            img = jpeg(img, level)
        elif kind == "resize":
            img = resize_roundtrip(img, level)
        elif kind == "blur":
            img = blur(img, level)
        elif kind == "noise":
            img = noise(img, level, seed)
        else:
            raise ValueError(f"unknown degradation: {kind}")
    return img


class FixedDegrade:
    """Same degradation for every image (test conditions). Noise is seeded per image."""

    def __init__(self, steps):
        self.steps = steps

    def __call__(self, img, sample_id=""):
        return apply_steps(img, self.steps, stable_seed(sample_id))


class RandomDegrade:
    """Training augmentation: random blur / resize / JPEG, each with prob p.
    deterministic=True gives the same result per (seed, sample_id), used for cached features."""

    def __init__(self, p_blur=0.5, blur_sigma=(0.1, 3.0), p_resize=0.5, resize_scale=(0.5, 1.5),
                 p_jpeg=0.5, jpeg_quality=(60, 100), deterministic=False, seed=42):
        self.p_blur, self.blur_sigma = p_blur, tuple(blur_sigma)
        self.p_resize, self.resize_scale = p_resize, tuple(resize_scale)
        self.p_jpeg, self.jpeg_quality = p_jpeg, tuple(jpeg_quality)
        self.deterministic, self.seed = deterministic, seed

    def sample_steps(self, rng) -> list:
        steps = []
        if rng.random() < self.p_blur:
            steps.append(["blur", rng.uniform(*self.blur_sigma)])
        if rng.random() < self.p_resize:
            steps.append(["resize", rng.uniform(*self.resize_scale)])
        if rng.random() < self.p_jpeg:  # JPEG last, like re-uploading
            steps.append(["jpeg", rng.randint(*self.jpeg_quality)])
        return steps

    def __call__(self, img, sample_id=""):
        # python random is reseeded per DataLoader worker by torch
        rng = random.Random(stable_seed(self.seed, sample_id)) if self.deterministic else random
        return apply_steps(img, self.sample_steps(rng))


class DegradedDataset(GenImageDataset):
    """GenImageDataset + degrade(img, sample_id) before the model transform."""

    def __init__(self, df, root, transform=None, degrade=None, path_col: str = "processed_relpath"):
        super().__init__(df, root, transform, path_col)
        self.degrade = degrade

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        img = Image.open(Path(self.root) / r[self.path_col]).convert("RGB")
        if self.degrade is not None:
            img = self.degrade(img, r["sample_id"])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(r["y"]), i


def condition_family(name: str, steps) -> tuple[str, float]:
    """For plots: 'jpeg75' -> ('jpeg', 75). clean -> ('clean', 0). combos -> ('combo', 0)."""
    if not steps:
        return "clean", 0.0
    if len(steps) == 1:
        return steps[0][0], float(steps[0][1])
    return "combo", 0.0
