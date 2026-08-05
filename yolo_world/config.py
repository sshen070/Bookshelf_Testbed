"""Config loading, with paths resolved against the package directory.

Same contract as vit_classifier/config.py: relative paths in config.yaml
resolve against this package rather than the working directory, so the CLI
behaves identically from the repo root, from inside WAFT, or from a cron job
with an arbitrary cwd.

Imports cleanly without torch or ultralytics.
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


def detector_kwargs(cfg: dict) -> dict:
    """Flatten the nested YAML into the flat dict WorldDetector expects."""
    model = cfg.get("model", {})
    return {
        "name": model.get("name", "yolov8s-worldv2.pt"),
        "device": model.get("device", "auto"),
        "imgsz": int(model.get("imgsz", 640)),
        "conf": float(model.get("conf", 0.05)),
        "iou": float(model.get("iou", 0.7)),
    }


def events_dir(cfg: dict) -> Path:
    return resolve(cfg.get("events", {}).get(
        "events_dir", "../WAFT/WAFT/outputs/camera_events/events"))
