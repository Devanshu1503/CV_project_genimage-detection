"""One-call environment setup used by the first cell of every notebook.

The notebook's first cell clones/pulls the repo (it has to, because src/ is not
importable before the repo exists), then calls `setup()` from here, which:
  1. mounts Google Drive (Colab only),
  2. installs Python deps from requirements.txt and apt packages (p7zip-full),
  3. sets all random seeds,
  4. prints GPU info,
  5. loads the config, resolves paths and creates the output folders.
"""
from __future__ import annotations

import os
import platform
import random
import shutil
import subprocess
import sys

import numpy as np

from .config import ensure_dirs, get_paths, is_colab, load_config, REPO_ROOT


def mount_drive() -> None:
    """Mount Google Drive at /content/drive (no-op outside Colab)."""
    if not is_colab():
        print("[setup] Not in Colab: skipping Drive mount.")
        return
    if os.path.isdir("/content/drive/MyDrive"):
        print("[setup] Drive already mounted.")
        return
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive")


def install_requirements(apt_packages: list[str] | None = None, quiet: bool = True) -> None:
    """pip install requirements.txt and (Colab only) apt packages."""
    req = REPO_ROOT / "requirements.txt"
    if req.exists():
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(req)]
        if quiet:
            cmd.insert(4, "-q")
        print("[setup] pip install -r requirements.txt ...")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stdout[-3000:], res.stderr[-3000:])
            raise RuntimeError("pip install failed (see output above).")
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
    """Fix python / numpy / torch seeds for reproducibility."""
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
    """Return a human-readable description of the available accelerator."""
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
    """Run the full setup. Returns (cfg, P)."""
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
