"""Tiny CPU smoke test of the training code path (dataset -> ResNet-50 -> checkpoint/resume,
and the linear probe). Uses random weights and 16 synthetic images, so it runs in seconds
and needs no data or internet. Skipped if torchvision is not installed."""
import numpy as np
import pandas as pd
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.datasets import GenImageDataset, resnet_transforms  # noqa: E402
from src.models import build_resnet50  # noqa: E402
from src.train import predict_scores, train_linear_probe, train_resnet  # noqa: E402


def _make_images(root, n=16):
    rows = []
    rng = np.random.default_rng(0)
    for i in range(n):
        y = i % 2
        arr = (rng.random((64, 64, 3)) * (255 if y else 120)).astype(np.uint8)
        rel = f"processed/g/{'ai' if y else 'nature'}/{i}.jpg"
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(root / rel, quality=96)
        rows.append({"sample_id": f"g/{i}", "y": y, "processed_relpath": rel})
    return pd.DataFrame(rows)


def test_resnet_train_resume_and_predict(tmp_path):
    from torch.utils.data import DataLoader
    df = _make_images(tmp_path)
    tr = GenImageDataset(df, tmp_path, resnet_transforms(48, train=True))
    va = GenImageDataset(df, tmp_path, resnet_transforms(48, train=False))
    tl = DataLoader(tr, batch_size=8, shuffle=True)
    vl = DataLoader(va, batch_size=8)
    model = build_resnet50(weights=None)
    hist = train_resnet(model, tl, vl, df["y"].to_numpy(), tmp_path / "ckpt", epochs=1,
                        device="cpu", amp=False)
    assert len(hist) == 1 and (tmp_path / "ckpt" / "last.pt").exists()
    # Resuming with more epochs continues from epoch 2.
    hist2 = train_resnet(build_resnet50(weights=None), tl, vl, df["y"].to_numpy(), tmp_path / "ckpt",
                         epochs=2, patience=5, device="cpu", amp=False)
    assert hist2["epoch"].tolist() == [1, 2]
    scores = predict_scores(model, vl, device="cpu", amp=False)
    assert scores.shape == (16,) and ((scores >= 0) & (scores <= 1)).all()


def test_linear_probe():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 10))
    y = (X[:, 0] > 0).astype(int)
    best, table = train_linear_probe(X[:150], y[:150], X[150:], y[150:], C_grid=[0.01, 1.0])
    assert len(table) == 2 and table["val_auroc"].max() > 0.9
    assert best.predict_proba(X[:5]).shape == (5, 2)
