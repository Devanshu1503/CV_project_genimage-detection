"""Bootstrap confidence intervals for AUROC (per generator, mean unseen, paired diffs)."""
# Refs: Efron & Tibshirani 1993 (bootstrap), percentile CIs
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def fast_auroc(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Mann-Whitney AUROC, works on a (B, n) batch of resamples. Ties count 0.5."""
    y = np.asarray(y).astype(bool)
    s = np.atleast_2d(np.asarray(scores, dtype=float))
    yy = np.broadcast_to(y, s.shape)
    r = rankdata(s, axis=1)
    n1 = yy.sum(1)
    n0 = yy.shape[1] - n1
    out = ((r * yy).sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)
    return out if np.ndim(scores) > 1 else out[0]


def _boot_idx(y: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Stratified resample: reals and fakes resampled separately (keeps 500/500)."""
    y = np.asarray(y)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return np.concatenate([rng.choice(pos, (n_boot, len(pos))), rng.choice(neg, (n_boot, len(neg)))], axis=1)


def bootstrap_auroc(y, score, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05) -> dict:
    """Point AUROC + percentile CI + all bootstrap values."""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    idx = _boot_idx(y, n_boot, np.random.default_rng(seed))
    boots = fast_auroc(y[idx], score[idx])
    return {"auroc": float(fast_auroc(y, score)), "lo": float(np.quantile(boots, alpha / 2)),
            "hi": float(np.quantile(boots, 1 - alpha / 2)), "se": float(boots.std(ddof=1)), "boots": boots}


def bootstrap_table(preds: pd.DataFrame, generator_order: list[str], train_generator: str | None = None,
                    n_boot: int = 1000, seed: int = 42, alpha: float = 0.05) -> pd.DataFrame:
    """AUROC CI per (model, generator) + mean_unseen (generators resampled independently)."""
    rows = []
    for model, gm in preds.groupby("model", sort=False):
        boots = {}
        for k, gen in enumerate(generator_order):
            g = gm.loc[gm["generator"] == gen]
            if g.empty:
                continue
            # same seed per generator across models -> paired resamples
            r = bootstrap_auroc(g["y"].to_numpy(), g["score"].to_numpy(), n_boot, seed + k, alpha)
            boots[gen] = r.pop("boots")
            rows.append({"model": model, "generator": gen, **r})
        unseen = [g for g in boots if g != train_generator]
        if train_generator and unseen:
            b = np.mean([boots[g] for g in unseen], axis=0)
            point = np.mean([r["auroc"] for r in rows if r["model"] == model and r["generator"] in unseen])
            rows.append({"model": model, "generator": "mean_unseen", "auroc": float(point),
                         "lo": float(np.quantile(b, alpha / 2)), "hi": float(np.quantile(b, 1 - alpha / 2)),
                         "se": float(b.std(ddof=1))})
    return pd.DataFrame(rows)


def paired_diff_table(preds: pd.DataFrame, model_a: str, model_b: str, generator_order: list[str],
                      train_generator: str | None = None, n_boot: int = 1000, seed: int = 42,
                      alpha: float = 0.05) -> pd.DataFrame:
    """AUROC(a) - AUROC(b) with paired bootstrap (same images in both)."""
    rows, diffs = [], {}
    for k, gen in enumerate(generator_order):
        g = preds.loc[(preds["generator"] == gen) & preds["model"].isin([model_a, model_b])]
        if g.empty:
            continue
        w = g.pivot_table(index="sample_id", columns="model", values="score")
        y = g.drop_duplicates("sample_id").set_index("sample_id").loc[w.index, "y"].to_numpy()
        if w[[model_a, model_b]].isna().any().any():
            raise ValueError(f"{gen}: models were not scored on the same images")
        idx = _boot_idx(y, n_boot, np.random.default_rng(seed + k))
        sa, sb = w[model_a].to_numpy(), w[model_b].to_numpy()
        d = fast_auroc(y[idx], sa[idx]) - fast_auroc(y[idx], sb[idx])
        diffs[gen] = d
        rows.append(_diff_row(gen, float(fast_auroc(y, sa) - fast_auroc(y, sb)), d, alpha))
    unseen = [g for g in diffs if g != train_generator]
    if train_generator and unseen:
        point = np.mean([r["diff"] for r in rows if r["generator"] in unseen])
        rows.append(_diff_row("mean_unseen", float(point), np.mean([diffs[g] for g in unseen], 0), alpha))
    return pd.DataFrame(rows).assign(model_a=model_a, model_b=model_b)


def _diff_row(gen, point, d, alpha):
    lo, hi = np.quantile(d, alpha / 2), np.quantile(d, 1 - alpha / 2)
    # two-sided bootstrap p-value for diff = 0
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return {"generator": gen, "diff": point, "lo": float(lo), "hi": float(hi),
            "p_boot": float(p), "significant": bool(lo > 0 or hi < 0)}


def plot_ci(table: pd.DataFrame, generator_order: list[str], names: dict | None = None,
            model_order: list[str] | None = None, title: str = "AUROC with 95% bootstrap CI"):
    """Points + error bars per generator, one colour per model."""
    import matplotlib.pyplot as plt
    names = names or {}
    gens = [g for g in generator_order + ["mean_unseen"] if g in set(table["generator"])]
    models = model_order or list(dict.fromkeys(table["model"]))
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(gens)), 4))
    width = 0.8 / len(models)
    for i, m in enumerate(models):
        t = table.loc[table["model"] == m].set_index("generator").reindex(gens)
        x = np.arange(len(gens)) - 0.4 + width * (i + 0.5)
        ax.errorbar(x, t["auroc"], yerr=[t["auroc"] - t["lo"], t["hi"] - t["auroc"]], fmt="o", capsize=3,
                    label=m)
    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xticks(np.arange(len(gens)), [names.get(g, g) for g in gens])
    ax.set(ylabel="AUROC", ylim=(0.2, 1.02), title=title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
