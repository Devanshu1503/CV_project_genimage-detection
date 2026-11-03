"""Tests for path -> split/label parsing, class parsing and sampling."""
import pandas as pd

from src.archive_utils import parse_7z_slt
from src.manifest import (build_manifest, build_sampling_plan, filename_leakage_report,
                          parse_imagenet_class, parse_inner_path, stratified_sample)
from tests.test_archive_utils import SAMPLE_SLT


def test_parse_inner_path_with_top_folder():
    r = parse_inner_path("imagenet_ai_0419_sdv4/train/ai/001_sdv4_00001.png")
    assert r == {"split": "train", "label": "ai", "filename": "001_sdv4_00001.png",
                 "ext": ".png", "top_folder": "imagenet_ai_0419_sdv4"}


def test_parse_inner_path_without_top_folder_and_case():
    r = parse_inner_path("VAL/Nature/ILSVRC2012_val_00000001.JPEG")
    assert (r["split"], r["label"], r["ext"], r["top_folder"]) == ("val", "nature", ".jpeg", "")


def test_parse_inner_path_unknown():
    r = parse_inner_path("imagenet_x/test/fake/a.png")
    assert r["split"] == "unknown" and r["label"] == "unknown"
    assert r["top_folder"] == "imagenet_x/test/fake"


def test_parse_imagenet_class():
    assert parse_imagenet_class("n01440764_10026.JPEG") == ("n01440764", "synset")
    assert parse_imagenet_class("001_sdv4_00001.png") == ("1", "leading_int")
    assert parse_imagenet_class("ILSVRC2012_val_00000001.JPEG") == (None, None)


def test_build_manifest_from_listing():
    m = build_manifest(parse_7z_slt(SAMPLE_SLT), "stable_diffusion_v_1_4", "/x/a.zip")
    assert len(m) == 3                                   # the folder entry is dropped
    assert m["label"].tolist() == ["ai", "nature", "nature"]
    assert m["split"].tolist() == ["train", "train", "val"]
    assert m["is_image"].all()


def _toy_manifest():
    rows = []
    for gen in ["stable_diffusion_v_1_4", "ADM"]:
        for split in ["train", "val"]:
            for i in range(40):
                cls = f"n{(i % 4):08d}"
                rows.append(dict(generator=gen, split=split, label="nature", filename=f"{cls}_{i}.JPEG",
                                 ext=".jpeg", is_image=True, uncompressed_bytes=100,
                                 imagenet_class=cls, inner_path=f"{split}/nature/{cls}_{i}.JPEG"))
                rows.append(dict(generator=gen, split=split, label="ai", filename=f"{i % 4}_{gen}_{i}.png",
                                 ext=".png", is_image=True, uncompressed_bytes=300,
                                 imagenet_class=str(i % 4), inner_path=f"{split}/ai/{i}.png"))
    return pd.DataFrame(rows)


def test_stratified_sample_balanced_and_deterministic():
    pool = pd.DataFrame({"x": range(100), "cls": ["a"] * 70 + ["b"] * 20 + ["c"] * 10})
    s1 = stratified_sample(pool, 30, "cls", seed=42)
    s2 = stratified_sample(pool, 30, "cls", seed=42)
    assert s1["x"].tolist() == s2["x"].tolist()
    assert s1["cls"].value_counts().to_dict() == {"a": 10, "b": 10, "c": 10}
    assert len(stratified_sample(pool, 500, "cls", 42)) == 100   # smaller pool -> all rows


def test_sampling_plan_excludes_train_filenames_from_test():
    m = _toy_manifest()
    # Make ADM's val nature images reuse the SD1.4 *train* filenames -> leakage candidates.
    sd_train = m[(m.generator == "stable_diffusion_v_1_4") & (m.split == "train") & (m.label == "nature")]
    adm_val = (m.generator == "ADM") & (m.split == "val") & (m.label == "nature")
    m.loc[adm_val, "filename"] = sd_train["filename"].values
    cfg = {"project": {"seed": 42}, "train_generator": "stable_diffusion_v_1_4",
           "generators": [{"folder": "stable_diffusion_v_1_4"}, {"folder": "ADM"}],
           "sampling": {"train_split_name": "train", "test_split_name": "val", "train_per_label": 10,
                        "test_per_label": 5, "stratify_by_class": True}}
    plan = build_sampling_plan(m, cfg, logger=None)
    train_names = set(plan.loc[plan.role == "train", "filename"])
    test_names = set(plan.loc[plan.role == "test", "filename"])
    assert not (train_names & test_names)
    assert (plan.role == "train").sum() == 20
    assert plan["sample_id"].is_unique
    assert set(plan["y"]) == {0, 1}


def test_filename_leakage_report():
    m = _toy_manifest()
    rep = filename_leakage_report(m)
    # Same 40 nature filenames in both generators, and in both train and val.
    assert rep["summary"]["in_more_than_one_generator"] == 40
    assert rep["summary"]["in_both_train_and_val"] == 40


def test_duplicate_pairs_and_leakage():
    from src.manifest import duplicate_pairs, leaking_test_ids
    df = pd.DataFrame({
        "sample_id": ["tr1", "tr2", "te1", "te2", "te3"],
        "generator": ["sd", "sd", "adm", "adm", "glide"],
        "role": ["train", "train", "test", "test", "test"],
        "label": ["nature"] * 5,
        "md5": ["m1", "m2", "m1", "m3", "m4"],                # te1 is an exact copy of tr1
        "phash": ["ffffffffffffffff", "0000000000000000", "ffffffffffffffff",
                  "00000000000000ff", "000000000000000f"],    # te2/te3 near tr2 (8 and 4 bits)
    })
    p = duplicate_pairs(df, near_threshold=4)
    kinds = dict(zip(zip(p.a_sample_id, p.b_sample_id), p.kind))
    assert kinds[("tr1", "te1")] == "exact"
    assert kinds[("tr2", "te3")] == "near"
    assert ("tr2", "te2") not in kinds                       # 8 bits apart > threshold
    assert leaking_test_ids(p) == {"te1", "te3"}
