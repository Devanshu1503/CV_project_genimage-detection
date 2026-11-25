import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.stats import bootstrap_auroc, bootstrap_table, fast_auroc, paired_diff_table


def _data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(n // 2), np.zeros(n // 2)].astype(int)
    s = rng.normal(size=n) + y
    return y, s


def test_fast_auroc_matches_sklearn():
    y, s = _data()
    assert abs(fast_auroc(y, s) - roc_auc_score(y, s)) < 1e-12
    s2 = np.round(s, 1)  # ties
    assert abs(fast_auroc(y, s2) - roc_auc_score(y, s2)) < 1e-12
    batch = np.stack([s, -s])
    assert np.allclose(fast_auroc(y, batch), [roc_auc_score(y, s), roc_auc_score(y, -s)])


def test_bootstrap_ci_contains_point():
    y, s = _data()
    r = bootstrap_auroc(y, s, n_boot=300)
    assert r["lo"] <= r["auroc"] <= r["hi"] and len(r["boots"]) == 300


def test_tables():
    y, s = _data()
    rows = []
    for gen in ["a", "b"]:
        for m, sc in [("m1", s), ("m2", s * 0.5 + 0.1)]:  # same ranking -> diff 0
            rows.append(pd.DataFrame({"sample_id": [f"{gen}/{i}" for i in range(len(y))], "y": y,
                                      "score": sc, "model": m, "generator": gen}))
    preds = pd.concat(rows)
    t = bootstrap_table(preds, ["a", "b"], train_generator="a", n_boot=200)
    assert set(t["generator"]) == {"a", "b", "mean_unseen"} and len(t) == 6
    d = paired_diff_table(preds, "m1", "m2", ["a", "b"], train_generator="a", n_boot=200)
    assert np.allclose(d["diff"], 0) and not d["significant"].any()
