"""Grad-CAM for the 1-logit ResNet-50, own implementation (no pytorch-grad-cam)."""
# Refs: Selvaraju et al. 2017, Grad-CAM, arXiv 1610.02391
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .datasets import IMAGENET_MEAN, IMAGENET_STD


class GradCAM:
    """Hooks one conv layer, cam = relu(sum_k mean(dlogit/dA_k) * A_k)."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model.eval()
        self.acts = None
        self.grads = None
        self._hook = target_layer.register_forward_hook(self._save)

    def _save(self, module, inp, out):
        self.acts = out
        if out.requires_grad:  # skip plain no_grad forwards
            out.register_hook(self._save_grad)

    def _save_grad(self, g):
        self.grads = g

    def remove(self) -> None:
        self._hook.remove()

    def __call__(self, x: torch.Tensor, sign="fake"):
        """x (B,3,H,W) -> cams (B,H,W) in [0,1], P(fake) (B,).
        sign: "fake" explains the fake logit, "real" the real side, "pred" the predicted class."""
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(x).squeeze(1)
            if sign == "pred":
                s = torch.where(logits.detach() >= 0, 1.0, -1.0)  # logit 0 = P 0.5
            else:
                s = {"fake": 1.0, "real": -1.0}[sign]
            # images are independent in eval mode, so sum is fine
            (s * logits).sum().backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)          # channel weights
        cam = F.relu((w * self.acts).sum(dim=1, keepdim=True))  # (B,1,h,w)
        raw = cam.detach()
        cam = F.interpolate(raw, size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        flat = cam.flatten(1)
        lo, hi = flat.min(1, keepdim=True)[0], flat.max(1, keepdim=True)[0]
        cam = ((flat - lo) / (hi - lo + 1e-8)).view_as(cam)
        probs = torch.sigmoid(logits.detach().float())
        return cam.cpu().numpy(), probs.cpu().numpy()


def get_layer(model: nn.Module, name: str) -> nn.Module:
    """e.g. 'layer4' -> model.layer4[-1] (last bottleneck)."""
    layer = getattr(model, name)
    return layer[-1] if isinstance(layer, nn.Sequential) else layer


def to_uint8(x: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalisation, (3,H,W) -> (H,W,3) uint8."""
    m = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    s = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (x.detach().cpu() * s + m).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def overlay(img: np.ndarray, cam: np.ndarray, alpha: float = 0.45, cmap: str = "jet") -> np.ndarray:
    """Heatmap on top of the image, uint8."""
    import matplotlib
    heat = matplotlib.colormaps[cmap](cam)[..., :3]
    out = (1 - alpha) * img / 255.0 + alpha * heat
    return (out.clip(0, 1) * 255).round().astype(np.uint8)


def cam_stats(cams: np.ndarray, top_frac: float = 0.1, center_frac: float = 0.5) -> dict:
    """Simple numbers per cam: how peaked, how central."""
    b, h, w = cams.shape
    flat = cams.reshape(b, -1)
    tot = flat.sum(1) + 1e-8
    k = max(1, int(round(top_frac * flat.shape[1])))
    top = np.sort(flat, axis=1)[:, -k:].sum(1) / tot          # share of mass in top pixels
    ch, cw = int(h * center_frac), int(w * center_frac)
    t, l = (h - ch) // 2, (w - cw) // 2
    center = cams[:, t:t + ch, l:l + cw].reshape(b, -1).sum(1) / tot
    return {"top_mass": top, "center_mass": center}


@torch.no_grad()
def deletion_test(model: nn.Module, x: torch.Tensor, cams: np.ndarray, frac: float = 0.2,
                  seed: int = 42) -> dict:
    """Blank the top-frac cam pixels and compare with two baselines of the same area:
    a random square (fair baseline, one blob like the cam region) and random scattered pixels.
    If the cam is faithful, blanking it should lower P(fake) more than the random square."""
    model.eval()
    b, _, h, w = x.shape
    n = int(round(frac * h * w))
    cam_t = torch.as_tensor(cams, device=x.device).reshape(b, -1)
    g = torch.Generator(device="cpu").manual_seed(seed)

    def blank(keep):  # keep: (b, h*w), 0 = blank. 0 = dataset mean colour after normalisation
        return x * keep.view(b, 1, h, w)

    keep_cam = torch.ones(b, h * w, device=x.device).scatter_(1, cam_t.topk(n, dim=1).indices, 0.0)
    pix = torch.stack([torch.randperm(h * w, generator=g)[:n] for _ in range(b)]).to(x.device)
    keep_pix = torch.ones(b, h * w, device=x.device).scatter_(1, pix, 0.0)
    side = int(round(n ** 0.5))
    keep_box = torch.ones(b, h, w, device=x.device)
    for i in range(b):
        t = int(torch.randint(0, h - side + 1, (1,), generator=g))
        l = int(torch.randint(0, w - side + 1, (1,), generator=g))
        keep_box[i, t:t + side, l:l + side] = 0.0

    def p(inp):
        return torch.sigmoid(model(inp).squeeze(1).float()).cpu().numpy()

    return {"p_orig": p(x), "p_cam_masked": p(blank(keep_cam)), "p_box_masked": p(blank(keep_box.view(b, -1))),
            "p_pixels_masked": p(blank(keep_pix))}
