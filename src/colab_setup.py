"""One-call environment setup used by the first cell of every notebook.

The notebook's first cell clones/pulls the repo (it has to, because src/ is not
importable before the repo exists), then calls `setup()` from here, which:
  1. mounts Google Drive (Colab only),
  2. installs ONLY MISSING Python packages (requirements-colab.txt on Colab,
     requirements.txt locally; nothing is upgraded) and apt p7zip-full,
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


# pip name -> import name, for packages where the two differ.
_IMPORT_NAMES = {"pyyaml": "yaml", "pillow": "PIL", "scikit-learn": "sklearn",
                 "opencv-python": "cv2", "imagehash": "imagehash", "open_clip_torch": "open_clip"}


def _requirement_names(req_file) -> list[str]:
    """Package names from a requirements file (comments and version specifiers removed)."""
    import re
    names = []
    for line in open(req_file, encoding="utf-8"):
        line = line.split("#", 1)[0].strip()
        if line:
            names.append(re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0])
    return names


def missing_packages(req_file) -> list[str]:
    """Requirements that are NOT importable in this environment."""
    import importlib.util
    return [n for n in _requirement_names(req_file)
            if importlib.util.find_spec(_IMPORT_NAMES.get(n.lower(), n.replace("-", "_"))) is None]


def install_requirements(apt_packages: list[str] | None = None, quiet: bool = True) -> None:
    """Install only what is missing; never upgrade what is already there.

    Colab: requirements-colab.txt (torch, numpy, pandas, ... are preinstalled and are
    deliberately left alone, so Colab's CUDA build of torch is never replaced).
    Local: requirements.txt. Plus apt packages (Colab only).
    """
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
