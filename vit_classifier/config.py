"""Config loading, with paths resolved against the package directory.

Relative paths in config.yaml resolve against this package rather than the
working directory, so `python -m vit_classifier` behaves identically whether
it's run from the repo root, from inside WAFT, or from a cron job with an
arbitrary cwd.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_DIR / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.is_absolute():
        path = PACKAGE_DIR / path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve(rel_path: str | Path) -> Path:
    p = Path(rel_path)
    return p if p.is_absolute() else PACKAGE_DIR / p


def classifier_kwargs(cfg: dict) -> dict:
    """Flatten the nested YAML into the flat dict RoiClassifier expects."""
    model = cfg.get("model", {})
    bank = cfg.get("bank", {})
    crops = cfg.get("crops", {})
    transitions = cfg.get("transitions", {})
    return {
        "name": model.get("name", "vit_small_patch14_dinov2.lvd142m"),
        "img_size": model.get("img_size", 224),
        "device": model.get("device", "auto"),
        "knn_k": bank.get("knn_k", 5),
        "min_similarity": bank.get("min_similarity", 0.55),
        "crop_pad_frac": crops.get("pad_frac", 0.15),
        "crop_square": crops.get("square", True),
        "crop_min_size_px": crops.get("min_size_px", 8),
        "background_class": transitions.get("background_class", "background"),
    }


def bank_paths(cfg: dict) -> tuple[Path, Path]:
    bank = cfg.get("bank", {})
    return (
        resolve(bank.get("exemplar_dir", "exemplars")),
        resolve(bank.get("cache", ".cache/exemplar_bank.npz")),
    )
