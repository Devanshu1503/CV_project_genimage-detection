"""Training code for Problem 1.

- train_resnet():       fine-tune ResNet-50 with AMP, early stopping on val AUROC,
                        and checkpoint/resume (Colab sessions disconnect).
- predict_scores():     P(fake) for every image of a DataLoader.
- train_linear_probe(): logistic regression on frozen features, C tuned on val AUROC.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


@torch.no_grad()
def predict_scores(model: nn.Module, loader, device: str = "cuda", amp: bool = True) -> np.ndarray:
    """Return sigmoid(logit) = predicted probability of FAKE, in loader order."""
    from tqdm.auto import tqdm
    model.eval()
    use_amp = amp and device.startswith("cuda")
    out = []
    for x, _, _ in tqdm(loader, desc="predict", leave=False):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x.to(device, non_blocking=True)).squeeze(1)
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out)


def _save_ckpt(path: Path, **state) -> None:
    tmp = path.with_name(path.name + ".part")
    torch.save(state, tmp)
    tmp.replace(path)


def train_resnet(model: nn.Module, train_loader, val_loader, val_labels: np.ndarray,
                 ckpt_dir: str | Path, epochs: int = 5, lr: float = 1e-4, weight_decay: float = 1e-4,
                 patience: int = 2, amp: bool = True, device: str = "cuda", logger=None) -> pd.DataFrame:
    """Fine-tune with BCE loss. Saves last.pt every epoch and best.pt on val-AUROC improvement.

    If ckpt_dir/last.pt exists, training RESUMES from it (model, optimizer, scaler,
    epoch counter, history and early-stopping state are all restored).
    Returns the training history (one row per epoch); also saved as history.csv.
    """
    from tqdm.auto import tqdm
    log = logger.info if logger else print
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    use_amp = amp and device.startswith("cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = nn.BCEWithLogitsLoss()

    start_epoch, best_auc, bad_epochs, history = 0, -1.0, 0, []
    last = ckpt_dir / "last.pt"
    if last.exists():
        st = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        start_epoch, best_auc = st["epoch"] + 1, st["best_auc"]
        bad_epochs, history = st["bad_epochs"], st["history"]
        log(f"Resumed from {last} (finished epoch {st['epoch'] + 1}, best val AUROC {best_auc:.4f})")
        if st.get("stopped_early") or start_epoch >= epochs:
            log("Training already finished according to the checkpoint.")
            return pd.DataFrame(history)

    for epoch in range(start_epoch, epochs):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y, _ in tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False):
            x, y = x.to(device, non_blocking=True), y.float().to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model(x).squeeze(1), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(y)
            n += len(y)
        sched.step()

        scores = predict_scores(model, val_loader, device, amp)
        val_auc = float(roc_auc_score(val_labels, scores))
        val_loss = float(nn.functional.binary_cross_entropy(
            torch.tensor(scores).clamp(1e-7, 1 - 1e-7), torch.tensor(val_labels).float()))
        val_acc = float(((scores >= 0.5) == val_labels).mean())
        history.append({"epoch": epoch + 1, "train_loss": total / max(n, 1), "val_loss": val_loss,
                        "val_auroc": val_auc, "val_acc": val_acc, "lr": opt.param_groups[0]["lr"],
                        "seconds": time.time() - t0})
        log(f"epoch {epoch + 1}: train_loss={total / max(n, 1):.4f} val_loss={val_loss:.4f} "
            f"val_auroc={val_auc:.4f} val_acc={val_acc:.4f} ({time.time() - t0:.0f}s)")

        if val_auc > best_auc:
            best_auc, bad_epochs = val_auc, 0
            _save_ckpt(ckpt_dir / "best.pt", model=model.state_dict(), epoch=epoch, val_auroc=val_auc)
            log(f"  new best -> saved {ckpt_dir / 'best.pt'}")
        else:
            bad_epochs += 1
        stop = bad_epochs >= patience
        _save_ckpt(last, model=model.state_dict(), optimizer=opt.state_dict(),
                   scheduler=sched.state_dict(), scaler=scaler.state_dict(), epoch=epoch,
                   best_auc=best_auc, bad_epochs=bad_epochs, history=history, stopped_early=stop)
        pd.DataFrame(history).to_csv(ckpt_dir / "history.csv", index=False)
        if stop:
            log(f"Early stopping: no val AUROC improvement for {patience} epochs.")
            break
    return pd.DataFrame(history)


def train_linear_probe(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                       C_grid: list[float], max_iter: int = 5000, seed: int = 42, logger=None):
    """Standardise features, fit logistic regression for each C, keep the best val AUROC.

    Small C = strong L2 regularisation. Returns (best fitted pipeline, tuning table).
    The final model is the one trained on the TRAIN split only, so the val split
    stays an honest model-selection set.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    log = logger.info if logger else print
    rows, best, best_auc = [], None, -1.0
    for C in C_grid:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=max_iter, random_state=seed))
        t0 = time.time()
        clf.fit(X_train, y_train)
        p = clf.predict_proba(X_val)[:, 1]
        auc = float(roc_auc_score(y_val, p))
        acc = float(((p >= 0.5) == y_val).mean())
        rows.append({"C": C, "val_auroc": auc, "val_acc": acc, "fit_seconds": time.time() - t0})
        log(f"  C={C:g}: val AUROC={auc:.4f} acc={acc:.4f}")
        if auc > best_auc:
            best, best_auc = clf, auc
    return best, pd.DataFrame(rows)


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)
