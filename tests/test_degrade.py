import numpy as np
import pandas as pd
from PIL import Image

from src.degrade import DegradedDataset, FixedDegrade, RandomDegrade, apply_steps, condition_family


def _img(seed=0):
    return Image.fromarray((np.random.default_rng(seed).random((64, 64, 3)) * 255).astype(np.uint8))


def test_steps_keep_size_and_change_pixels():
    img = _img()
    for steps in [[["jpeg", 50]], [["resize", 0.5]], [["resize", 1.5]], [["blur", 2]], [["noise", 10]],
                  [["resize", 0.75], ["jpeg", 75]]]:
        out = apply_steps(img, steps, seed=1)
        assert out.size == img.size and out.mode == "RGB"
        assert np.abs(np.asarray(out, float) - np.asarray(img, float)).mean() > 0.5
    assert apply_steps(img, []) is img


def test_fixed_noise_is_per_image_deterministic():
    f = FixedDegrade([["noise", 10]])
    a, b, c = f(_img(), "x"), f(_img(), "x"), f(_img(), "y")
    assert np.array_equal(np.asarray(a), np.asarray(b)) and not np.array_equal(np.asarray(a), np.asarray(c))


def test_random_degrade_deterministic_mode():
    rd = RandomDegrade(p_blur=1, p_resize=1, p_jpeg=1, deterministic=True)
    assert np.array_equal(np.asarray(rd(_img(), "s1")), np.asarray(rd(_img(), "s1")))
    import random
    steps = RandomDegrade().sample_steps(random.Random(0))
    assert all(k in {"blur", "resize", "jpeg"} for k, _ in steps)


def test_degraded_dataset(tmp_path):
    _img().save(tmp_path / "a.jpg")
    df = pd.DataFrame({"sample_id": ["a"], "y": [1], "processed_relpath": ["a.jpg"]})
    ds = DegradedDataset(df, tmp_path, transform=None, degrade=FixedDegrade([["jpeg", 50]]))
    img, y, i = ds[0]
    assert img.size == (64, 64) and y == 1 and i == 0


def test_condition_family():
    assert condition_family("clean", []) == ("clean", 0.0)
    assert condition_family("jpeg75", [["jpeg", 75]]) == ("jpeg", 75.0)
    assert condition_family("social", [["resize", 0.75], ["jpeg", 75]])[0] == "combo"
