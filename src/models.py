"""Problem 1 models: ResNet-50 and frozen CLIP / DINOv2 feature extractors."""
# Refs: ResNet arXiv 1512.03385, CLIP arXiv 2103.00020, DINOv2 arXiv 2304.07193,
# UnivFD (frozen CLIP + linear probe) arXiv 2302.10174
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def build_resnet50(weights: str | None = "IMAGENET1K_V2") -> nn.Module:
    """ResNet-50 with a 1-logit head."""
    from torchvision.models import ResNet50_Weights, resnet50
    w = getattr(ResNet50_Weights, weights) if weights else None
    model = resnet50(weights=w)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


class CLIPFeatureExtractor(nn.Module):
    """Frozen CLIP image encoder (768-d embedding)."""

    def __init__(self, hf_id: str = "openai/clip-vit-large-patch14"):
        super().__init__()
        from transformers import CLIPVisionModelWithProjection
        self.model = CLIPVisionModelWithProjection.from_pretrained(hf_id).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=x).image_embeds


class DINOv2FeatureExtractor(nn.Module):
    """Frozen DINOv2, CLS token (768-d)."""

    def __init__(self, hf_id: str = "facebook/dinov2-base"):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(hf_id).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=x).pooler_output


def build_feature_extractor(kind: str, cfg: dict) -> nn.Module:
    if kind == "clip":
        return CLIPFeatureExtractor(cfg["models"]["clip"]["hf_id"])
    if kind == "dinov2":
        return DINOv2FeatureExtractor(cfg["models"]["dinov2"]["hf_id"])
    raise ValueError(f"Unknown feature extractor: {kind}")


@torch.no_grad()
def extract_features_cached(extractor: nn.Module, dataset, out_dir: str | Path, batch_size: int = 64,
                            shard_size: int = 2000, num_workers: int = 2, device: str = "cuda",
                            amp: bool = True) -> np.ndarray:
    """Embed all images once, cached as .npy shards (resumable)."""
    from torch.utils.data import DataLoader, Subset
    from tqdm.auto import tqdm

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(dataset)
    # cache only valid for the same images in the same order
    ids = dataset.df["sample_id"].astype(str).tolist() if hasattr(dataset, "df") else None
    ids_file = out_dir / "sample_ids.txt"
    if ids is not None:
        if ids_file.exists():
            if ids_file.read_text(encoding="utf-8").splitlines() != ids:
                raise RuntimeError(
                    f"Cached features in {out_dir} were computed for a DIFFERENT image list. "
                    "Delete that folder to recompute.")
        else:
            ids_file.write_text("\n".join(ids), encoding="utf-8")
    shards = [(s, min(s + shard_size, n)) for s in range(0, n, shard_size)]
    extractor = extractor.to(device).eval()
    use_amp = amp and device.startswith("cuda")
    for k, (s, e) in enumerate(shards):
        f = out_dir / f"shard_{k:05d}.npy"
        if f.exists() and np.load(f, mmap_mode="r").shape[0] == e - s:
            continue
        dl = DataLoader(Subset(dataset, range(s, e)), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=device.startswith("cuda"))
        feats = []
        for x, _, _ in tqdm(dl, desc=f"shard {k + 1}/{len(shards)}"):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                z = extractor(x.to(device, non_blocking=True))
            feats.append(z.float().cpu().numpy())
        arr = np.concatenate(feats).astype(np.float32)
        tmp = f.with_name(f.stem + ".part.npy")
        np.save(tmp, arr)
        tmp.replace(f)
    return np.concatenate([np.load(out_dir / f"shard_{k:05d}.npy") for k in range(len(shards))])
