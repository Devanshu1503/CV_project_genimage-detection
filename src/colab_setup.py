"""Setup used by the first cell of every notebook."""
from __future__ import annotations

import os
import platform
import random
import shutil
import subprocess
import sys

import numpy as np

from .config import ensure_dirs, get_paths, is_colab, load_config, REPO_ROOT


def flush_drive() -> None:
    """Push pending Drive writes and unmount (Colab only), run before closing the tab."""
    if not is_colab():
        print("[flush] Not in Colab: nothing to flush.")
        return
    from google.colab import drive  # type: ignore
    drive.flush_and_unmount()
    print("[flush] Drive flushed and unmounted, safe to close the tab.")


def mount_drive() -> None:
    """Mount Drive (Colab only)."""
    if not is_colab():
        print("[setup] Not in Colab: skipping Drive mount.")
        return
    if os.path.isdir("/content/drive/MyDrive"):
        print("[setup] Drive already mounted.")
        return
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive")


# pip name -> import name
_IMPORT_NAMES = {"pyyaml": "yaml", "pillow": "PIL", "scikit-learn": "sklearn",
                 "opencv-python": "cv2", "imagehash": "imagehash", "open_clip_torch": "open_clip"}


def _requirement_names(req_file) -> list[str]:
    """Package names in a requirements file."""
    import re
    names = []
    for line in open(req_file, encoding="utf-8"):
        line = line.split("#", 1)[0].strip()
        if line:
            names.append(re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0])
    return names


def missing_packages(req_file) -> list[str]:
    """Requirements that aren't importable yet."""
    import importlib.util
    return [n for n in _requirement_names(req_file)
            if importlib.util.find_spec(_IMPORT_NAMES.get(n.lower(), n.replace("-", "_"))) is None]


def install_requirements(apt_packages: list[str] | None = None, quiet: bool = True) -> None:
    """pip install only missing packages (never upgrades), plus apt packages on Colab."""
    req = REPO_ROOT / ("requirements-colab.txt" if is_colab() else "requirements.txt")
    if req.exists():
        missing = missing_packages(req)
        if missing:
            print(f"[setup] pip install (missing only): {missing}")
            cmd = [sys.executable, "-m", "pip", "install", *(["-q"] if quiet else []), *missing]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print(res.stdout[-3000:], res.stderr[-3000:])
                raise RuntimeError("pip install failed (see output above).")
        else:
            print(f"[setup] all packages in {req.name} already installed (nothing reinstalled).")
    if apt_packages and is_colab():
        missing = [p for p in apt_packages if not _apt_installed(p)]
        if missing:
            print(f"[setup] apt-get install {' '.join(missing)} ...")
            subprocess.run(["apt-get", "-qq", "update"], check=False, capture_output=True)
            res = subprocess.run(["apt-get", "-qq", "install", "-y", *missing],
                                 capture_output=True, text=True)
            if res.returncode != 0:
                print(res.stdout[-3000:], res.stderr[-3000:])
                raise RuntimeError(f"apt-get install {missing} failed.")
        else:
            print(f"[setup] apt packages already installed: {apt_packages}")


def _apt_installed(pkg: str) -> bool:
    res = subprocess.run(["dpkg", "-s", pkg], capture_output=True, text=True)
    return res.returncode == 0


def set_seeds(seed: int) -> None:
    """Seed python, numpy, torch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def gpu_info() -> str:
    """GPU name/memory or "no GPU"."""
    try:
        import torch
    except ImportError:
        return "torch not installed"
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        return f"GPU: {p.name} | {p.total_memory / 1e9:.1f} GB | torch {torch.__version__}"
    return f"No GPU (CPU only) | torch {torch.__version__}"


def setup(apt_packages: list[str] | None = ("p7zip-full",), install: bool = True,
          config_path: str | None = None):
    """Mount, install, seed, make folders. Returns (cfg, P)."""
    print(f"[setup] Python {platform.python_version()} | Colab: {is_colab()} | repo: {REPO_ROOT}")
    mount_drive()
    if install:
        install_requirements(list(apt_packages) if apt_packages else None)
    cfg = load_config(config_path)
    set_seeds(cfg["project"]["seed"])
    P = get_paths(cfg)
    ensure_dirs(P)
    print(f"[setup] {gpu_info()}")
    print(f"[setup] seed = {cfg['project']['seed']}")
    print(f"[setup] Paths ({P.env}):")
    P.show()
    if not P.genimage_root.exists():
        print(f"[setup] WARNING: GenImage folder not found at {P.genimage_root}. "
              "In Colab, add a Drive shortcut named 'genimage' to MyDrive (see README).")
    du = shutil.disk_usage(str(P.work_dir))
    print(f"[setup] Free space on work disk: {du.free / 1e9:.1f} GB")
    return cfg, P
