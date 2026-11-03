"""Load config.yaml and resolve paths (Colab or local)."""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"


def is_colab() -> bool:
    """Running in Colab? (GENIMAGE_ENV can override)."""
    override = os.environ.get("GENIMAGE_ENV", "").strip().lower()
    if override in {"colab", "local"}:
        return override == "colab"
    return "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg


@dataclass
class Paths:
    """Project folders."""
    env: str
    repo_root: Path
    genimage_root: Path
    project_root: Path
    work_dir: Path
    manifests: Path
    raw_subset: Path
    processed: Path
    features: Path
    checkpoints: Path
    results: Path
    figures: Path
    logs: Path

    def show(self) -> None:
        for k, v in self.__dict__.items():
            print(f"  {k:<14} {v}")


def _resolve(p: str) -> Path:
    # "/..." counts as absolute on Windows too
    return Path(p) if (Path(p).is_absolute() or str(p).startswith("/")) else (REPO_ROOT / p)


def get_paths(cfg: dict, env: str | None = None) -> Paths:
    """Paths for the colab or local block."""
    env = env or ("colab" if is_colab() else "local")
    block = cfg["paths"][env]
    project_root = _resolve(block["project_root"])
    sub = {name: project_root / name for name in cfg["project_subdirs"]}
    return Paths(
        env=env,
        repo_root=REPO_ROOT,
        genimage_root=_resolve(block["genimage_root"]),
        project_root=project_root,
        work_dir=_resolve(block["work_dir"]),
        **sub,
    )


def ensure_dirs(P: Paths) -> None:
    """Create the output folders."""
    for name in ["project_root", "work_dir", "manifests", "raw_subset", "processed",
                 "features", "checkpoints", "results", "figures", "logs"]:
        getattr(P, name).mkdir(parents=True, exist_ok=True)


def generator_names(cfg: dict) -> dict:
    """folder -> display name."""
    return {g["folder"]: g["name"] for g in cfg["generators"]}


def get_logger(name: str, log_dir: Path | None = None) -> logging.Logger:
    """Logger that prints and writes logs/<name>.log."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_dir is not None:
        log_file = str(Path(log_dir) / f"{name}.log")
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_file)
                   for h in logger.handlers):
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger
