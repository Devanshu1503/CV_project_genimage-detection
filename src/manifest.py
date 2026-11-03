"""Manifests (one row per file in an archive), sampling and leakage checks."""
# Refs: GenImage, Zhu et al. 2023, arXiv 2306.08571
from __future__ import annotations

import re
from pathlib import PurePosixPath

import numpy as np
import pandas as pd

SPLITS = {"train", "val"}
LABELS = {"ai": 1, "nature": 0}  # 1 = fake
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".gif"}
# keep these as strings (leading zeros)
STR_DTYPES = {"md5": str, "phash": str, "imagenet_class": str, "filename": str}


def read_csv(path, **kw) -> pd.DataFrame:
    """pd.read_csv that keeps hash/class columns as strings."""
    return pd.read_csv(path, dtype=STR_DTYPES, low_memory=False, **kw)

# ImageNet synset, e.g. n01440764
_SYNSET_RE = re.compile(r"(?<![a-z0-9])(n\d{8})(?![0-9])", re.IGNORECASE)
# e.g. 123_sdv4_00001.png, we assume 123 is the class (checked in nb 01)
_LEADING_INT_RE = re.compile(r"^(\d{1,4})_")


def parse_inner_path(inner_path: str) -> dict:
    """Path inside the archive -> split, label, filename, ext, top folder."""
    p = PurePosixPath(inner_path.replace("\\", "/"))
    parts = list(p.parts)
    filename = parts[-1] if parts else ""
    ext = PurePosixPath(filename).suffix.lower()
    split = label = "unknown"
    top_folder = ""
    dirs = [d.lower() for d in parts[:-1]]
    # last train|val dir followed by ai|nature
    for i in range(len(dirs) - 1, -1, -1):
        if dirs[i] in SPLITS and i + 1 < len(dirs) and dirs[i + 1] in LABELS:
            split, label = dirs[i], dirs[i + 1]
            top_folder = "/".join(parts[:i])
            break
    else:
        top_folder = "/".join(parts[:-1])
    return {"split": split, "label": label, "filename": filename, "ext": ext,
            "top_folder": top_folder}


def parse_imagenet_class(filename: str) -> tuple[str | None, str | None]:
    """(class_id, "synset" | "leading_int" | None)."""
    m = _SYNSET_RE.search(filename)
    if m:
        return m.group(1).lower(), "synset"
    m = _LEADING_INT_RE.match(filename)
    if m:
        return str(int(m.group(1))), "leading_int"
    return None, None


def build_manifest(listing: pd.DataFrame, generator: str, archive_path: str) -> pd.DataFrame:
    """Parsed listing -> manifest (files only)."""
    files = listing.loc[~listing["is_dir"]].copy()
    parsed = pd.DataFrame([parse_inner_path(p) for p in files["inner_path"]], index=files.index)
    cls = [parse_imagenet_class(f) for f in parsed["filename"]]
    out = pd.DataFrame({
        "generator": generator,
        "archive_path": archive_path,
        "inner_path": files["inner_path"].values,
        "top_folder": parsed["top_folder"].values,
        "split": parsed["split"].values,
        "label": parsed["label"].values,
        "filename": parsed["filename"].values,
        "ext": parsed["ext"].values,
        "is_image": parsed["ext"].isin(IMAGE_EXTS).values,
        "uncompressed_bytes": files["size"].values,
        "imagenet_class": [c for c, _ in cls],
        "class_source": [s for _, s in cls],
    })
    return out


def sample_id_of(df: pd.DataFrame) -> pd.Series:
    """generator/split/label/filename."""
    return df["generator"] + "/" + df["split"] + "/" + df["label"] + "/" + df["filename"]


# Summaries
def manifest_summary(m: pd.DataFrame) -> pd.DataFrame:
        return (m.groupby(["generator", "split", "label"]).size()
              .unstack(["split", "label"], fill_value=0))


def filename_leakage_report(m: pd.DataFrame) -> dict:
    """Do real filenames repeat across generators or train/val?"""
    nat = m.loc[(m["label"] == "nature") & m["is_image"]]
    per_name = nat.groupby("filename").agg(n_generators=("generator", "nunique"),
                                           n_splits=("split", "nunique"),
                                           n_rows=("generator", "size"))
    report = {
        "n_unique_nature_filenames": len(per_name),
        "in_more_than_one_generator": int((per_name["n_generators"] > 1).sum()),
        "in_both_train_and_val": int((per_name["n_splits"] > 1).sum()),
    }
    dup_within = nat.groupby(["generator", "filename"]).size()
    report["repeated_within_a_generator"] = int((dup_within > 1).sum())
    # overlap between generators
    gens = sorted(nat["generator"].unique())
    sets = {g: set(nat.loc[nat["generator"] == g, "filename"]) for g in gens}
    overlap = pd.DataFrame([[len(sets[a] & sets[b]) for b in gens] for a in gens],
                           index=gens, columns=gens)
    return {"summary": report, "overlap_matrix": overlap, "per_filename": per_name}


# Sampling
def stratified_sample(pool: pd.DataFrame, n: int, strata_col: str | None, seed: int) -> pd.DataFrame:
    """Sample n rows, round-robin over classes so coverage is even."""
    if len(pool) <= n:
        return pool.copy()
    shuffled = pool.sample(frac=1.0, random_state=seed)
    if not strata_col or strata_col not in shuffled or shuffled[strata_col].isna().all():
        return shuffled.head(n).copy()
    key = shuffled[strata_col].fillna("__none__")
    rank = shuffled.groupby(key).cumcount()
    order = np.argsort(rank.to_numpy(), kind="stable")
    return shuffled.iloc[order[:n]].copy()


def build_sampling_plan(m: pd.DataFrame, cfg: dict, logger=None) -> pd.DataFrame:
    """SD1.4 train pool + one test set per generator, no filename overlap."""
    log = logger.info if logger else print
    s = cfg["sampling"]
    seed = cfg["project"]["seed"]
    strata = "imagenet_class" if s["stratify_by_class"] else None
    ok = m.loc[m["is_image"] & m["label"].isin(LABELS.keys())]
    picks = []

    train_gen = cfg["train_generator"]
    for label in ["ai", "nature"]:
        pool = ok.loc[(ok["generator"] == train_gen) & (ok["split"] == s["train_split_name"])
                      & (ok["label"] == label)]
        smp = stratified_sample(pool, s["train_per_label"], strata, seed)
        if len(smp) < s["train_per_label"]:
            log(f"WARNING: train pool {train_gen}/{label} has only {len(pool)} files")
        smp["role"] = "train"
        picks.append(smp)
    train_names = set(pd.concat(picks)["filename"])

    for g in [x["folder"] for x in cfg["generators"]]:
        for label in ["ai", "nature"]:
            pool = ok.loc[(ok["generator"] == g) & (ok["split"] == s["test_split_name"])
                          & (ok["label"] == label)]
            n_before = len(pool)
            pool = pool.loc[~pool["filename"].isin(train_names)]
            if n_before != len(pool):
                log(f"  {g}/{label}: removed {n_before - len(pool)} test candidates whose "
                    "filename is in the training sample")
            smp = stratified_sample(pool, s["test_per_label"], strata, seed)
            if len(smp) < s["test_per_label"]:
                log(f"WARNING: test pool {g}/{label} has only {len(pool)} files")
            smp["role"] = "test"
            picks.append(smp)

    plan = pd.concat(picks, ignore_index=True)
    plan["sample_id"] = sample_id_of(plan)
    plan["y"] = plan["label"].map(LABELS)
    plan["raw_relpath"] = ("raw_subset/" + plan["generator"] + "/" + plan["split"] + "/"
                           + plan["label"] + "/" + plan["filename"])
    assert plan["sample_id"].is_unique, "sample_id must be unique"
    return plan


def class_coverage(plan: pd.DataFrame) -> pd.DataFrame:
    """Classes covered per generator/role/label."""
    g = plan.groupby(["generator", "role", "label"])
    return pd.DataFrame({
        "n_images": g.size(),
        "n_with_class": g["imagenet_class"].apply(lambda s: s.notna().sum()),
        "n_classes": g["imagenet_class"].nunique(),
    }).reset_index()


def estimate_size_gb(plan: pd.DataFrame) -> float:
    return float(plan["uncompressed_bytes"].sum()) / 1e9


def suggest_sample_sizes(plan: pd.DataFrame, budget_gb: float, train_n: int, test_n: int) -> dict:
    """Smaller sample sizes that fit the Drive budget."""
    total = estimate_size_gb(plan)
    factor = min(1.0, 0.9 * budget_gb / max(total, 1e-9))
    return {"estimated_gb": round(total, 2), "budget_gb": budget_gb, "scale": round(factor, 3),
            "suggested_train_per_label": int(train_n * factor),
            "suggested_test_per_label": int(test_n * factor)}


def list_generator_to_manifest(gen: str, inventory: pd.DataFrame, manifests_dir, work_dir,
                               cfg: dict, relist: bool = False, logger=None):
    """List a generator's archive(s) and save manifest_<gen>.csv.gz (loads it if it exists)."""
    from pathlib import Path
    from . import archive_utils as au
    log = logger.info if logger else print
    out = Path(manifests_dir) / f"manifest_{gen}.csv.gz"
    if out.exists() and not relist:
        log(f"{gen}: manifest already exists -> loading {out.name}")
        return read_csv(out), None
    arcs = inventory.loc[(inventory["generator"] == gen) & (inventory["status"] == "ok")]
    if arcs.empty:
        raise RuntimeError(f"{gen}: no complete archive found (see the inventory table).")
    if len(arcs) > 1:
        log(f"{gen}: NOTE {len(arcs)} archives found; all of them are listed.")
    parts, total = [], 0.0
    for zp in arcs["zip_path"]:
        log(f"{gen}: 7z l -slt {zp}")
        sz = cfg["seven_zip"]
        listing, secs = au.list_archive(zp, sz["binary"], sz["list_timeout_sec"],
                                        cache_txt=Path(work_dir) / "listings" / f"{Path(zp).stem}.txt")
        total += secs
        parts.append(build_manifest(listing, gen, zp))
        log(f"{gen}: {len(listing)} entries listed in {secs:.1f}s")
    m = pd.concat(parts, ignore_index=True)
    m.to_csv(out, index=False)
    log(f"{gen}: saved {out} ({len(m)} files)")
    return m, total


# Duplicates / leakage
def duplicate_pairs(df: pd.DataFrame, near_threshold: int) -> pd.DataFrame:
    """Exact (md5) and near (pHash) duplicate pairs."""
    from itertools import combinations
    from .image_utils import near_duplicate_pairs
    df = df.reset_index(drop=True)
    pairs = []
    for _, g in df.groupby("md5"):
        if len(g) > 1:
            pairs += [(a, b, 0, "exact") for a, b in combinations(sorted(g.index), 2)]
    pairs += [(i, j, d, "near") for i, j, d in near_duplicate_pairs(df["phash"].tolist(), near_threshold)]
    cols = ["i", "j", "hamming", "kind"]
    p = pd.DataFrame(pairs, columns=cols).drop_duplicates(["i", "j"], keep="first")
    for side, col in (("a", "i"), ("b", "j")):
        for c in ["sample_id", "generator", "role", "label"]:
            p[f"{side}_{c}"] = df.loc[p[col], c].to_numpy()
    p["roles"] = ["-".join(sorted(x)) for x in zip(p["a_role"], p["b_role"])]
    p["same_generator"] = p["a_generator"] == p["b_generator"]
    return p.drop(columns=["i", "j"]).reset_index(drop=True)


def leaking_test_ids(pairs: pd.DataFrame) -> set:
    """Test images that duplicate a training image."""
    tt = pairs.loc[pairs["roles"] == "test-train"]
    ids = set(tt.loc[tt["a_role"] == "test", "a_sample_id"]) | set(tt.loc[tt["b_role"] == "test", "b_sample_id"])
    return ids
