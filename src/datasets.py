"""Dataset + transforms for the processed images."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
# from github.com/openai/CLIP
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


def load_index(manifests_dir: str | Path) -> pd.DataFrame:
    """processed_index.csv + splits.csv."""
    manifests_dir = Path(manifests_dir)
    # imagenet_class mixes ints and synsets, read as str (no DtypeWarning)
    idx = pd.read_csv(manifests_dir / "processed_index.csv", dtype={"imagenet_class": str}, low_memory=False)
    splits = pd.read_csv(manifests_dir / "splits.csv")
    df = idx.merge(splits[["sample_id", "split_set"]], on="sample_id", how="inner", validate="1:1")
    return df.reset_index(drop=True)


class GenImageDataset:
    """Returns (image, y, idx), y=1 is fake."""

    def __init__(self, df: pd.DataFrame, root: str | Path, transform=None,
                 path_col: str = "processed_relpath"):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform
        self.path_col = path_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        img = Image.open(self.root / r[self.path_col]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(r["y"]), i


def resnet_transforms(crop_size: int, train: bool):
    """Random crop + flip (train) or center crop (eval), no resizing."""
    # Refs: Wang et al. 2020, arXiv 1912.11035
    from torchvision import transforms as T
    if train:
        aug = [T.RandomCrop(crop_size), T.RandomHorizontalFlip()]
    else:
        aug = [T.CenterCrop(crop_size)]
    return T.Compose(aug + [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def feature_transforms(kind: str, size: int = 224):
    """Eval transform for CLIP / DINOv2."""
    from torchvision import transforms as T
    mean, std = (CLIP_MEAN, CLIP_STD) if kind == "clip" else (IMAGENET_MEAN, IMAGENET_STD)
    return T.Compose([T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
                      T.CenterCrop(size), T.ToTensor(), T.Normalize(mean, std)])


def stage_locally(relpaths, src_root: str | Path, dst_root: str | Path, n_threads: int = 16) -> int:
    """Copy files from Drive to local disk (skips existing)."""
    import shutil
    from concurrent.futures import ThreadPoolExecutor
    from tqdm.auto import tqdm
    src_root, dst_root = Path(src_root), Path(dst_root)
    todo = [r for r in relpaths if not (dst_root / r).exists()]

    def copy(rel):
        (dst_root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_root / rel, dst_root / rel)

    with ThreadPoolExecutor(n_threads) as ex:
        list(tqdm(ex.map(copy, todo), total=len(todo), desc="copy to local disk"))
    return len(todo)
