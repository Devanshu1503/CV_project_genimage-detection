"""Plotting and analysis helpers for the EDA notebook.

All figures go through `savefig()` so they land in <project_root>/figures with
consistent style and resolution.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .image_utils import channel_histograms, log_fft_spectrum, pixel_stats, safe_open

REAL_COLOR = "#4C72B0"
FAKE_COLOR = "#DD8452"
LABEL_PALETTE = {"nature": REAL_COLOR, "ai": FAKE_COLOR, "real": REAL_COLOR, "fake": FAKE_COLOR}


def set_style() -> None:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 100, "savefig.bbox": "tight", "axes.titleweight": "bold"})


def savefig(fig, name: str, figures_dir: str | Path, dpi: int = 150) -> Path:
    """Save a figure as PNG in figures_dir and print where it went."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / f"{name}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"[figure] saved {out}")
    return out


# -----------------------------------------------------------------------------
# Image grids
# -----------------------------------------------------------------------------
def plot_image_grid(paths: list, captions: list[str], ncols: int = 8, title: str = "",
                    thumb: int = 256, ax_size: float = 2.0):
    n = len(paths)
    nrows = max(1, int(np.ceil(n / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * ax_size, nrows * ax_size * 1.15))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, p, c in zip(axes, paths, captions):
        img, err = safe_open(p)
        if img is None:
            ax.text(0.5, 0.5, "unreadable", ha="center", va="center")
        else:
            img = img.convert("RGB")
            img.thumbnail((thumb, thumb))
            ax.imshow(img)
        ax.set_title(c, fontsize=7)
    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Pixel statistics
# -----------------------------------------------------------------------------
def sample_groups(df: pd.DataFrame, n_per_group: int, seed: int,
                  group_cols=("generator", "label")) -> pd.DataFrame:
    """Up to n rows per group (reproducible)."""
    return df.sample(frac=1.0, random_state=seed).groupby(list(group_cols)).head(n_per_group)


def compute_pixel_stats(df: pd.DataFrame, root: str | Path, path_col: str, max_side: int = 256,
                        bins: int = 64, n_threads: int = 8) -> tuple[pd.DataFrame, np.ndarray]:
    """Pixel stats for every row of df. Returns (stats DataFrame, histograms (n,3,bins))."""
    from concurrent.futures import ThreadPoolExecutor
    from tqdm.auto import tqdm
    root = Path(root)

    def one(rel):
        img, err = safe_open(root / rel)
        if img is None:
            return None, None
        return pixel_stats(img, max_side), channel_histograms(img, bins, max_side)

    with ThreadPoolExecutor(n_threads) as ex:
        res = list(tqdm(ex.map(one, df[path_col]), total=len(df), desc="pixel stats"))
    stats = [r[0] if r[0] else {} for r in res]
    hists = np.stack([r[1] if r[1] is not None else np.full((3, bins), np.nan) for r in res])
    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(stats)], axis=1)
    return out, hists


def average_spectra(df: pd.DataFrame, root: str | Path, path_col: str, group_col: str,
                    crop: int = 128, n_threads: int = 8) -> tuple[dict, dict]:
    """Mean log-magnitude FFT spectrum per group. Returns (spectra, n_used / n_skipped)."""
    from concurrent.futures import ThreadPoolExecutor
    from tqdm.auto import tqdm
    root = Path(root)

    def one(rel):
        img, _ = safe_open(root / rel)
        return None if img is None else log_fft_spectrum(img, crop)

    spectra, counts = {}, {}
    for g, sub in df.groupby(group_col, sort=False):
        with ThreadPoolExecutor(n_threads) as ex:
            specs = list(tqdm(ex.map(one, sub[path_col]), total=len(sub), desc=f"FFT {g}"))
        ok = [s for s in specs if s is not None]
        counts[g] = {"used": len(ok), "skipped_small_or_unreadable": len(specs) - len(ok)}
        if ok:
            spectra[g] = np.mean(ok, axis=0)
    return spectra, counts


def plot_spectra(spectra: dict, reference: str | None = None, ncols: int = 5, title: str = ""):
    """Show each group's mean spectrum; if `reference` is given, add a row of differences."""
    keys = list(spectra)
    rows = 2 if reference else 1
    ncols = min(ncols, len(keys))
    nr = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nr * rows, ncols, figsize=(ncols * 2.6, nr * rows * 2.7))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    vmin = min(np.percentile(s, 1) for s in spectra.values())
    vmax = max(np.percentile(s, 99.9) for s in spectra.values())
    for i, k in enumerate(keys):
        r, c = divmod(i, ncols)
        ax = axes[r * rows, c]
        ax.imshow(spectra[k], cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(k, fontsize=9)
        if reference and k != reference:
            ax2 = axes[r * rows + 1, c]
            diff = spectra[k] - spectra[reference]
            lim = np.percentile(np.abs(diff), 99.5) + 1e-9
            ax2.imshow(diff, cmap="coolwarm", vmin=-lim, vmax=lim)
            ax2.set_title(f"{k} - {reference}", fontsize=8)
    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Outliers
# -----------------------------------------------------------------------------
def iqr_outlier_mask(s: pd.Series, k: float = 1.5) -> pd.Series:
    """True where a value lies outside [Q1 - k*IQR, Q3 + k*IQR] (Tukey's fences)."""
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return (s < q1 - k * iqr) | (s > q3 + k * iqr)


# -----------------------------------------------------------------------------
# Metadata-only "shortcut" test
# -----------------------------------------------------------------------------
def metadata_shortcut_test(df: pd.DataFrame, feature_cols: list[str], label_col: str = "y",
                           seed: int = 42, n_splits: int = 5) -> tuple[pd.DataFrame, dict]:
    """Can real vs fake be predicted from file METADATA alone (no pixels)?

    If yes, the dataset contains a shortcut that an image model can exploit too.
    Models: logistic regression (standardised features) and a depth-3 decision tree.
    Returns (metrics table: mean +/- std over folds, {model: confusion matrix from
    out-of-fold predictions}).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    X = df[feature_cols].astype(float).fillna(-1).to_numpy()   # NaN (e.g. no JPEG quality) -> -1
    y = df[label_col].astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    models = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "decision_tree_d3": DecisionTreeClassifier(max_depth=3, random_state=seed),
    }
    rows, cms = [], {}
    for name, model in models.items():
        sc = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "f1", "roc_auc"])
        rows.append({"model": name, "features": "+".join(feature_cols),
                     **{f"{m}_mean": float(sc[f"test_{m}"].mean()) for m in ["accuracy", "f1", "roc_auc"]},
                     **{f"{m}_std": float(sc[f"test_{m}"].std()) for m in ["accuracy", "f1", "roc_auc"]}})
        pred = cross_val_predict(model, X, y, cv=cv)
        cms[name] = confusion_matrix(y, pred, labels=[0, 1])
    return pd.DataFrame(rows), cms


def plot_confusion(cm: np.ndarray, title: str = "", ax=None, labels=("real", "fake")):
    import seaborn as sns
    if ax is None:
        _, ax = plt.subplots(figsize=(3.2, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=9)
    return ax
