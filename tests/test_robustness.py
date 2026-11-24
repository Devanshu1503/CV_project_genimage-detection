import numpy as np
import pandas as pd
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src import robustness as rb  # noqa: E402
from src.models import build_resnet50  # noqa: E402


def _df(root, n=8):
    rows = []
    rng = np.random.default_rng(0)
    for i in range(n):
        rel = f"p/{i}.jpg"
        (root / "p").mkdir(exist_ok=True)
        Image.fromarray((rng.random((64, 64, 3)) * 255).astype(np.uint8)).save(root / rel)
        rows.append({"sample_id": f"s{i}", "generator": "g1" if i < 4 else "g2", "y": i % 2, "processed_relpath": rel})
    return pd.DataFrame(rows)


def test_score_resnet_cache_and_results(tmp_path):
    df = _df(tmp_path)
    conds = {"clean": [], "jpeg50": [["jpeg", 50]]}
    m = build_resnet50(weights=None).eval()
    kw = dict(crop_size=48, batch_size=4, num_workers=0, device="cpu", amp=False)
    p = rb.score_resnet(m, df, tmp_path, conds, "R", "t", tmp_path / "preds", **kw)
    assert len(p) == 16 and set(p["condition"]) == set(conds)
    p2 = rb.score_resnet(None, df, tmp_path, conds, "R", "t", tmp_path / "preds", **kw)  # from cache
    assert np.allclose(p["score"], p2["score"])
    with pytest.raises(RuntimeError):
        rb.score_resnet(None, df.iloc[::-1], tmp_path, conds, "R", "t", tmp_path / "preds", **kw)
    res = rb.results_by_condition(p, conds, ["g1", "g2"], "g1")
    assert {"mean_unseen", "mean_all"} <= set(res["generator"])
    assert res.loc[res["condition"] == "jpeg50", "family"].iloc[0] == "jpeg"
