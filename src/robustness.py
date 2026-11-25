"""Problem 2 scoring loops: every model x test set x degradation, cached so reruns skip finished work."""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import datasets as ds
from .degrade import DegradedDataset, FixedDegrade, condition_family


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def pred_file(out_dir, model_name: str, set_name: str, cond: str) -> Path:
    return Path(out_dir) / f"{slug(model_name)}__{set_name}__{cond}.csv"


def _check_cache(f: Path, df: pd.DataFrame) -> pd.DataFrame | None:
    if not f.exists():
        return None
    old = pd.read_csv(f)
    if old["sample_id"].astype(str).tolist() != df["sample_id"].astype(str).tolist():
        raise RuntimeError(f"{f} was made for a different image list, delete it to recompute")
    return old


def _rows(df, model_name, set_name, cond, scores):
    return (df[["sample_id", "generator", "y"]].reset_index(drop=True)
            .assign(model=model_name, test_set=set_name, condition=cond, score=scores))


def score_resnet(model, df, root, conditions: dict, model_name: str, set_name: str, out_dir,
                 crop_size: int = 192, batch_size: int = 64, num_workers: int = 2, device: str = "cuda",
                 amp: bool = True, logger=None) -> pd.DataFrame:
    """P(fake) for every condition, one cached csv per condition."""
    from torch.utils.data import DataLoader
    from .train import predict_scores
    log = logger.info if logger else print
    out = []
    for cond, steps in conditions.items():
        f = pred_file(out_dir, model_name, set_name, cond)
        old = _check_cache(f, df)
        if old is not None:
            out.append(old)
            continue
        dset = DegradedDataset(df, root, ds.resnet_transforms(crop_size, train=False), FixedDegrade(steps))
        dl = DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=device.startswith("cuda"))
        r = _rows(df, model_name, set_name, cond, predict_scores(model, dl, device, amp))
        f.parent.mkdir(parents=True, exist_ok=True)
        r.to_csv(f, index=False)
        log(f"{model_name} | {set_name} | {cond}: done")
        out.append(r)
    return pd.concat(out, ignore_index=True)


def _features_done(fdir: Path, df: pd.DataFrame, shard_size: int) -> bool:
    ids = fdir / "sample_ids.txt"
    n_shards = int(np.ceil(len(df) / shard_size))
    return (ids.exists() and ids.read_text(encoding="utf-8").splitlines() == df["sample_id"].astype(str).tolist()
            and all((fdir / f"shard_{k:05d}.npy").exists() for k in range(n_shards)))


class LazyExtractor:
    """Loads CLIP / DINOv2 only if some features are not cached yet."""

    def __init__(self, kind: str, cfg: dict):
        self.kind, self.cfg, self.model = kind, cfg, None

    def get(self):
        if self.model is None:
            from .models import build_feature_extractor
            self.model = build_feature_extractor(self.kind, self.cfg)
        return self.model

    def free(self):
        import gc
        import torch
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def features_for(lazy: LazyExtractor, df, root, steps, fdir, image_size=224, batch_size=64,
                 shard_size=2000, num_workers=2, device="cuda", degrade=None) -> np.ndarray:
    """Features of df under one degradation (or a custom degrade fn), cached in fdir."""
    from .models import extract_features_cached
    fdir = Path(fdir)
    if _features_done(fdir, df, shard_size):
        n_shards = int(np.ceil(len(df) / shard_size))
        return np.concatenate([np.load(fdir / f"shard_{k:05d}.npy") for k in range(n_shards)])
    dset = DegradedDataset(df, root, ds.feature_transforms(lazy.kind, image_size),
                           degrade if degrade is not None else FixedDegrade(steps))
    return extract_features_cached(lazy.get(), dset, fdir, batch_size=batch_size, shard_size=shard_size,
                                   num_workers=num_workers, device=device)


def score_probe(lazy: LazyExtractor, clf, df, root, conditions: dict, model_name: str, set_name: str,
                out_dir, feat_root, logger=None, **feat_kw) -> pd.DataFrame:
    """Linear probe P(fake) for every condition, features cached per condition."""
    log = logger.info if logger else print
    out = []
    for cond, steps in conditions.items():
        f = pred_file(out_dir, model_name, set_name, cond)
        old = _check_cache(f, df)
        if old is not None:
            out.append(old)
            continue
        X = features_for(lazy, df, root, steps, Path(feat_root) / lazy.kind / f"{set_name}__{cond}", **feat_kw)
        r = _rows(df, model_name, set_name, cond, clf.predict_proba(X)[:, 1])
        f.parent.mkdir(parents=True, exist_ok=True)
        r.to_csv(f, index=False)
        log(f"{model_name} | {set_name} | {cond}: done")
        out.append(r)
    return pd.concat(out, ignore_index=True)


def results_by_condition(preds: pd.DataFrame, conditions: dict, generator_order: list[str],
                         train_generator: str, threshold: float = 0.5) -> pd.DataFrame:
    """Metrics per (model, condition, generator) + mean_unseen + mean_all."""
    from .evaluate import per_generator_table
    out = []
    for cond, g in preds.groupby("condition", sort=False):
        t = per_generator_table(g, threshold, generator_order, train_generator=train_generator)
        cols = ["accuracy", "f1", "auroc", "avg_precision", "real_acc", "fake_acc"]
        allm = (t.loc[t["generator"] != "mean_unseen"].groupby("model")[cols].mean().reset_index()
                 .assign(generator="mean_all"))
        t = pd.concat([t, allm], ignore_index=True)
        fam, lvl = condition_family(cond, conditions.get(cond, []))
        out.append(t.assign(condition=cond, family=fam, level=lvl))
    return pd.concat(out, ignore_index=True)


# x position of "clean" on each family's axis
CLEAN_X = {"jpeg": 100, "resize": 1.0, "blur": 0.0, "noise": 0.0}
X_LABEL = {"jpeg": "JPEG quality (100 = no extra compression)", "resize": "resize factor (down/up, then back to 224)",
           "blur": "Gaussian blur sigma", "noise": "noise std (0-255)"}


def plot_degradation_curves(res: pd.DataFrame, generator: str = "mean_all", metric: str = "auroc",
                            model_order: list[str] | None = None, styles: dict | None = None, title: str = ""):
    """One subplot per degradation family, metric vs level, one line per model."""
    fams = [f for f in ["jpeg", "resize", "blur", "noise"] if f in set(res["family"])]
    models = model_order or list(dict.fromkeys(res["model"]))
    styles = styles or {}
    fig, axes = plt.subplots(1, len(fams), figsize=(4.2 * len(fams), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    sub = res.loc[res["generator"] == generator]
    for ax, fam in zip(axes, fams):
        for m in models:
            g = sub.loc[(sub["model"] == m) & (sub["family"] == fam)][["level", metric]]
            clean = sub.loc[(sub["model"] == m) & (sub["family"] == "clean"), metric]
            if len(clean):
                g = pd.concat([g, pd.DataFrame({"level": [CLEAN_X[fam]], metric: [clean.iloc[0]]})])
            g = g.sort_values("level")
            ax.plot(g["level"], g[metric], marker="o", label=m, **styles.get(m, {}))
        if fam == "jpeg":
            ax.invert_xaxis()  # left = mild, right = harsh
        ax.axhline(0.5, color="k", ls="--", lw=0.8)
        ax.set(xlabel=X_LABEL[fam], title=fam)
    axes[0].set_ylabel(f"{metric.upper()} ({generator})")
    axes[-1].legend(fontsize=7)
    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig
