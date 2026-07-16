"""Load and resolve the pipeline/yolo YAML configs relative to the repo root."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(rel_path: str | Path) -> Path:
    """Resolve a path from a config file against the repo root, creating parent dirs."""
    p = Path(rel_path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def load_pipeline_config(path: str | Path = "configs/pipeline.yaml") -> dict[str, Any]:
    return load_yaml(path)


def load_yolo_config(path: str | Path = "configs/yolo.yaml") -> dict[str, Any]:
    return load_yaml(path)
