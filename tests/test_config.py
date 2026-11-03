"""The config loads and all required keys / paths resolve."""
from src.config import get_paths, load_config


def test_config_loads_and_paths_resolve():
    cfg = load_config()
    assert cfg["project"]["seed"] == 42
    assert cfg["train_generator"] in [g["folder"] for g in cfg["generators"]]
    assert len(cfg["generators"]) == 8
    for env in ["colab", "local"]:
        P = get_paths(cfg, env=env)
        assert P.manifests.parent == P.project_root
    assert get_paths(cfg, "colab").genimage_root.as_posix() == "/content/drive/MyDrive/genimage"
    assert isinstance(cfg["training"]["resnet50"]["lr"], float)
