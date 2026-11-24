import pandas as pd
import pytest

pytest.importorskip("torchvision")

from src.generate import build_sdxl_index, make_prompt_plan  # noqa: E402


def test_prompt_plan_and_index():
    plan = make_prompt_plan(50, seed=42)
    assert len(plan) == 50 and plan["imagenet_idx"].is_unique and plan["seed"].is_unique
    assert plan["prompt"].str.startswith("a photo of a ").all()
    assert plan.equals(make_prompt_plan(50, seed=42))  # reproducible
    reals = pd.DataFrame({"sample_id": [f"r{i}" for i in range(50)], "generator": "sd", "generator_name": "SD",
                          "label": "nature", "y": 0, "split_set": "test", "role": "test",
                          "imagenet_class": "1", "processed_relpath": [f"p/{i}.jpg" for i in range(50)]})
    idx = build_sdxl_index(plan, "problem2/sdxl/processed", reals)
    assert len(idx) == 100 and idx["y"].sum() == 50 and (idx["generator"] == "sdxl_turbo").all()
    assert idx["sample_id"].is_unique and idx.loc[idx["y"] == 1, "processed_relpath"].str.endswith(".jpg").all()
