"""SDXL-Turbo fakes for Problem 2 (new unseen generator)."""
# Refs: Sauer et al. 2023, Adversarial Diffusion Distillation (SDXL-Turbo), arXiv 2311.17042
# diffusers: github.com/huggingface/diffusers
from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd


def imagenet_class_names() -> list[str]:
    """1000 ImageNet class names in index order (shipped with torchvision, no download)."""
    from torchvision.models import ResNet50_Weights
    return list(ResNet50_Weights.IMAGENET1K_V2.meta["categories"])


def make_prompt_plan(n: int, seed: int, template: str = "a photo of a {}") -> pd.DataFrame:
    """n distinct ImageNet classes, one prompt + one seed each."""
    names = imagenet_class_names()
    if n > len(names):
        raise ValueError(f"n={n} > {len(names)} classes")
    rng = np.random.default_rng(seed)
    cls = np.sort(rng.choice(len(names), n, replace=False))
    return pd.DataFrame({
        "gen_id": np.arange(n),
        "imagenet_idx": cls,
        "class_name": [names[c] for c in cls],
        "prompt": [template.format(names[c]) for c in cls],
        "seed": seed * 1000 + np.arange(n),
        "filename": [f"sdxlturbo_{i:04d}_{c:03d}.png" for i, c in enumerate(cls)],
    })


def load_pipeline(model_id: str, device: str = "cuda", fp16: bool = True):
    import torch
    from diffusers import AutoPipelineForText2Image
    kw = {"torch_dtype": torch.float16, "variant": "fp16"} if fp16 else {}
    pipe = AutoPipelineForText2Image.from_pretrained(model_id, **kw).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_from_plan(pipe, plan: pd.DataFrame, out_dir: str | Path, steps: int = 1,
                       guidance: float = 0.0, size: int = 512, batch_size: int = 4,
                       device: str = "cuda") -> int:
    """Generate missing images (resumable). One torch.Generator per image so seeds are exact."""
    import torch
    from tqdm.auto import tqdm
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = plan.loc[[not (out_dir / f).exists() for f in plan["filename"]]]
    for s in tqdm(range(0, len(todo), batch_size), desc="SDXL-Turbo"):
        b = todo.iloc[s:s + batch_size]
        gens = [torch.Generator(device).manual_seed(int(x)) for x in b["seed"]]
        imgs = pipe(prompt=b["prompt"].tolist(), num_inference_steps=steps, guidance_scale=guidance,
                    height=size, width=size, generator=gens).images
        for img, f in zip(imgs, b["filename"]):
            tmp = out_dir / (f + ".part")
            img.save(tmp, format="PNG")
            tmp.replace(out_dir / f)
    return len(todo)


def generation_metadata(model_id: str, steps: int, guidance: float, size: int, template: str,
                        seed: int) -> dict:
    """Everything needed to redo the generation."""
    meta = {"model_id": model_id, "num_inference_steps": steps, "guidance_scale": guidance,
            "size": size, "prompt_template": template, "base_seed": seed, "python": platform.python_version()}
    for pkg in ["torch", "diffusers", "transformers"]:
        try:
            meta[f"{pkg}_version"] = __import__(pkg).__version__
        except ImportError:
            meta[f"{pkg}_version"] = None
    try:  # exact model commit
        from huggingface_hub import HfApi
        meta["model_revision"] = HfApi().model_info(model_id).sha
    except Exception as e:
        meta["model_revision"] = f"unknown ({type(e).__name__})"
    return meta


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def build_sdxl_index(plan: pd.DataFrame, processed_rel_dir: str, reals: pd.DataFrame,
                     gen_folder: str = "sdxl_turbo", gen_name: str = "SDXL-Turbo") -> pd.DataFrame:
    """Test index for the new generator: our fakes + the given real test images."""
    fakes = pd.DataFrame({
        "sample_id": gen_folder + "/ai/" + plan["filename"],
        "generator": gen_folder, "generator_name": gen_name, "label": "ai", "y": 1,
        "split_set": "test", "role": "test",
        "imagenet_class": plan["imagenet_idx"].astype(str),
        "processed_relpath": processed_rel_dir.rstrip("/") + "/" + plan["filename"].str.replace(".png", ".jpg"),
        "prompt": plan["prompt"], "seed": plan["seed"],
    })
    r = reals.copy()
    r["generator"], r["generator_name"] = gen_folder, gen_name
    cols = [c for c in fakes.columns if c in r.columns]
    return pd.concat([r[cols], fakes], ignore_index=True)
