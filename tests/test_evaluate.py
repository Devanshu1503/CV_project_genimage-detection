"""Tests for metrics on toy arrays and the metadata shortcut test."""
import numpy as np
import pandas as pd

from src.eda_utils import iqr_outlier_mask, metadata_shortcut_test
from src.evaluate import compute_metrics, per_generator_table


def test_compute_metrics_toy():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.6, 0.3, 0.9, 0.8, 0.4, 0.7])
    m = compute_metrics(y, s, threshold=0.5)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (3, 1, 1, 3)
    assert m["accuracy"] == 0.75
    assert m["real_acc"] == 0.75 and m["fake_acc"] == 0.75
    assert abs(m["f1"] - 0.75) < 1e-9
    # 16 (real, fake) pairs; the fake is ranked higher in 15 of them -> AUROC = 15/16.
    assert abs(m["auroc"] - 15 / 16) < 1e-9


def test_compute_metrics_perfect_and_single_class():
    m = compute_metrics([0, 1], [0.0, 1.0])
    assert m["auroc"] == 1.0 and m["accuracy"] == 1.0
    m = compute_metrics([1, 1], [0.9, 0.2])
    assert np.isnan(m["auroc"]) and m["fake_acc"] == 0.5 and np.isnan(m["real_acc"])


def test_per_generator_table_mean_unseen():
    rng = np.random.default_rng(0)
    rows = []
    for model in ["A", "B"]:
        for gen in ["sd", "adm", "biggan"]:
            y = np.array([0] * 20 + [1] * 20)
            rows.append(pd.DataFrame({"model": model, "generator": gen, "y": y,
                                      "score": np.clip(y * 0.5 + rng.random(40) * 0.6, 0, 1)}))
    preds = pd.concat(rows)
    t = per_generator_table(preds, generator_order=["sd", "adm", "biggan"], train_generator="sd")
    assert len(t) == 2 * 3 + 2
    mu = t[(t.model == "A") & (t.generator == "mean_unseen")]["auroc"].iloc[0]
    exp = t[(t.model == "A") & t.generator.isin(["adm", "biggan"])]["auroc"].mean()
    assert abs(mu - exp) < 1e-12


def test_metadata_shortcut_detects_bias():
    # Fakes are PNG (is_jpeg=0) at 512 px, reals are JPEG of random size: trivially separable.
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"y": [1] * n + [0] * n, "is_jpeg": [0] * n + [1] * n,
                       "width": [512] * n + list(rng.integers(200, 800, n))})
    table, cms = metadata_shortcut_test(df, ["is_jpeg", "width"], seed=0, n_splits=5)
    assert (table["accuracy_mean"] > 0.99).all()
    assert cms["logistic_regression"].sum() == 2 * n


def test_iqr_outliers():
    s = pd.Series([10, 11, 12, 11, 10, 12, 100])
    assert iqr_outlier_mask(s).tolist() == [False] * 6 + [True]
