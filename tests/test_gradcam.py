import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.gradcam import GradCAM, cam_stats, deletion_test, get_layer, overlay, to_uint8  # noqa: E402
from src.models import build_resnet50  # noqa: E402


def test_gradcam_matches_manual():
    torch.manual_seed(0)
    model = build_resnet50(weights=None).eval()
    x = torch.randn(2, 3, 64, 64)
    cam = GradCAM(model, get_layer(model, "layer4"))
    cams, probs = cam(x)
    assert cams.shape == (2, 64, 64) and probs.shape == (2,)
    assert cams.min() >= 0 and cams.max() <= 1 + 1e-6
    # manual grad-cam for image 0 (low-res, before upsampling)
    feats = {}
    h = model.layer4[-1].register_forward_hook(lambda m, i, o: feats.setdefault("a", o))
    logit = model(x[:1]).squeeze()
    a = feats["a"]
    g = torch.autograd.grad(logit, a)[0]
    man = torch.relu((g.mean((2, 3), keepdim=True) * a).sum(1)).detach()[0]
    h.remove()
    up = torch.nn.functional.interpolate(man[None, None], size=(64, 64), mode="bilinear", align_corners=False)[0, 0]
    up = (up - up.min()) / (up.max() - up.min() + 1e-8)
    assert np.allclose(up.numpy(), cams[0], atol=1e-4)
    cam.remove()


def test_helpers():
    torch.manual_seed(0)
    model = build_resnet50(weights=None).eval()
    x = torch.randn(3, 3, 64, 64)
    cams = np.random.default_rng(0).random((3, 64, 64))
    st = cam_stats(cams)
    assert st["top_mass"].shape == (3,) and (st["center_mass"] <= 1).all()
    d = deletion_test(model, x, cams, frac=0.2)
    assert set(d) == {"p_orig", "p_cam_masked", "p_box_masked", "p_pixels_masked"} and d["p_orig"].shape == (3,)
    img = to_uint8(x[0])
    assert img.shape == (64, 64, 3) and overlay(img, cams[0]).dtype == np.uint8


def test_sign_modes():
    torch.manual_seed(0)
    model = build_resnet50(weights=None).eval()
    x = torch.randn(2, 3, 64, 64)
    cam = GradCAM(model, get_layer(model, "layer4"))
    for sgn in ["fake", "real", "pred"]:
        c, p = cam(x, sign=sgn)
        assert c.shape == (2, 64, 64)
