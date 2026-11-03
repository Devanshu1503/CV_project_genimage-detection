"""Metrics, per-generator results table and plots (y=1 is fake)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             roc_auc_score, roc_curve)


def compute_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    """Accuracy, F1, AUROC, AP, real/fake accuracy."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    both = len(np.unique(y_true)) == 2
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "n": int(len(y_true)), "n_real": int((y_true == 0).sum()), "n_fake": int((y_true == 1).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_score)) if both else float("nan"),
        "avg_precision": float(average_precision_score(y_true, y_score)) if both else float("nan"),
        "real_acc": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "fake_acc": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def per_generator_table(preds: pd.DataFrame, threshold: float = 0.5,
                        generator_order: list[str] | None = None,
                        train_generator: str | None = None) -> pd.DataFrame:
    """Metrics per (model, generator), plus mean_unseen."""
    rows = []
    for (model, gen), g in preds.groupby(["model", "generator"], sort=False):
        rows.append({"model": model, "generator": gen, **compute_metrics(g["y"], g["score"], threshold)})
    res = pd.DataFrame(rows)
    if generator_order:
        res["generator"] = pd.Categorical(res["generator"], generator_order, ordered=True)
        res = res.sort_values(["model", "generator"]).reset_index(drop=True)
        res["generator"] = res["generator"].astype(str)
    train_gen = train_generator
    if train_gen:
        metric_cols = ["accuracy", "f1", "auroc", "avg_precision", "real_acc", "fake_acc"]
        extra = (res.loc[res["generator"] != train_gen].groupby("model")[metric_cols].mean()
                    .reset_index().assign(generator="mean_unseen"))
        res = pd.concat([res, extra], ignore_index=True)
    return res


def plot_roc_per_generator(preds: pd.DataFrame, generator_order: list[str], ncols: int = 4,
                           names: dict | None = None):
    """ROC curves, one subplot per generator."""
    names = names or {}
    n = len(generator_order)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.4, nrows * 3.3), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, gen in zip(axes, generator_order):
        sub = preds.loc[preds["generator"] == gen]
        for model, g in sub.groupby("model", sort=False):
            if g["y"].nunique() < 2:
                continue
            fpr, tpr, _ = roc_curve(g["y"], g["score"])
            ax.plot(fpr, tpr, label=f"{model} ({roc_auc_score(g['y'], g['score']):.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_title(names.get(gen, gen), fontsize=10)
        ax.legend(fontsize=7, loc="lower right")
    for ax in axes[n:]:
        ax.axis("off")
    fig.supxlabel("False positive rate (real called fake)")
    fig.supylabel("True positive rate (fake caught)")
    fig.tight_layout()
    return fig


def plot_confusion_grid(preds: pd.DataFrame, model: str, generator_order: list[str],
                        threshold: float = 0.5, ncols: int = 4, names: dict | None = None):
    import seaborn as sns
    names = names or {}
    n = len(generator_order)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.8, nrows * 2.7))
    axes = np.atleast_1d(axes).ravel()
    for ax, gen in zip(axes, generator_order):
        g = preds.loc[(preds["model"] == model) & (preds["generator"] == gen)]
        cm = confusion_matrix(g["y"], (g["score"] >= threshold).astype(int), labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["real", "fake"], yticklabels=["real", "fake"])
        ax.set_title(names.get(gen, gen), fontsize=9)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"Confusion matrices: {model} (threshold {threshold})", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_metric_bars(results: pd.DataFrame, metric: str = "auroc", generator_order: list[str] | None = None,
                     names: dict | None = None):
    """Grouped bars: metric per generator per model."""
    names = names or {}
    piv = results.pivot(index="generator", columns="model", values=metric)
    if generator_order:
        piv = piv.reindex([g for g in generator_order if g in piv.index])
    piv.index = [names.get(g, g) for g in piv.index]
    fig, ax = plt.subplots(figsize=(max(8, len(piv) * 1.2), 4))
    piv.plot.bar(ax=ax, width=0.8)
    ax.axhline(0.5, color="k", ls="--", lw=0.8, label="chance")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel(metric.upper())
    ax.set_xlabel("")
    ax.set_title(f"{metric.upper()} per test generator", fontweight="bold")
    ax.legend(fontsize=8, ncols=2)
    plt.xticks(rotation=0)
    fig.tight_layout()
    return fig


def save_table(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[results] saved {path}")
