"""Split the auto-labeled pool into train/val/test and emit a YOLO data.yaml.

Splitting is grouped by capture session (the `<session>__` prefix autolabel.py
stamps on every pooled file), not by individual frame, so near-duplicate
frames from the same staged event can't leak between train and val/test and
inflate the validation score.
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import yaml

from src.utils.config import load_pipeline_config, resolve_path

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _session_of(pool_image_path: Path) -> str:
    return pool_image_path.stem.split("__", 1)[0]


def build_dataset(cfg_path: str = "configs/pipeline.yaml", out_root: str = "data/yolo_dataset") -> None:
    cfg = load_pipeline_config(cfg_path)
    split_cfg = cfg["dataset_split"]

    pool_images_dir = resolve_path(cfg["paths"]["labels_dir"]) / "images"
    pool_labels_dir = resolve_path(cfg["paths"]["labels_dir"]) / "labels"
    pool_images = sorted(p for p in pool_images_dir.glob("*") if p.suffix.lower() in IMG_EXTS)
    if not pool_images:
        print(f"No pooled images found in {pool_images_dir}. Run autolabel.py first.")
        return

    rng = random.Random(split_cfg["seed"])

    if split_cfg.get("group_by_session", True):
        sessions: dict[str, list[Path]] = {}
        for img_path in pool_images:
            sessions.setdefault(_session_of(img_path), []).append(img_path)
        session_names = list(sessions.keys())
        rng.shuffle(session_names)

        n = len(session_names)
        n_train = max(1, round(n * split_cfg["train"]))
        n_val = max(1 if n > 1 else 0, round(n * split_cfg["val"])) if n > 1 else 0
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)

        train_sessions = session_names[:n_train]
        val_sessions = session_names[n_train:n_train + n_val]
        test_sessions = session_names[n_train + n_val:]

        splits = {
            "train": [p for s in train_sessions for p in sessions[s]],
            "val": [p for s in val_sessions for p in sessions[s]],
            "test": [p for s in test_sessions for p in sessions[s]],
        }
    else:
        shuffled = pool_images[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = round(n * split_cfg["train"])
        n_val = round(n * split_cfg["val"])
        splits = {
            "train": shuffled[:n_train],
            "val": shuffled[n_train:n_train + n_val],
            "test": shuffled[n_train + n_val:],
        }

    n_sessions = len(sessions) if split_cfg.get("group_by_session", True) else None
    if not splits["train"] or not splits["val"]:
        detail = (
            f"{n_sessions} capture session(s) found ({len(pool_images)} pooled images)."
            if n_sessions is not None
            else f"{len(pool_images)} pooled images found."
        )
        raise ValueError(
            f"Split produced an empty train or val set -- {detail} "
            "YOLO training needs non-empty train AND val sets, and sessions are never split "
            "across them (see module docstring). Add more distinct capture sessions under "
            "data/raw/<session_name>/ (aim for several sessions per class of change, not just "
            "more frames within one session), re-run autolabel, then build-dataset again."
        )

    out_root_path = resolve_path(out_root)
    for split_name, images in splits.items():
        img_out = out_root_path / "images" / split_name
        lbl_out = out_root_path / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img_path in img_out.glob("*"):
            img_path.unlink()
        for lbl_path in lbl_out.glob("*"):
            lbl_path.unlink()

        for img_path in images:
            label_path = pool_labels_dir / f"{img_path.stem}.txt"
            shutil.copy2(img_path, img_out / img_path.name)
            if label_path.exists():
                shutil.copy2(label_path, lbl_out / label_path.name)
            else:
                (lbl_out / f"{img_path.stem}.txt").touch()

    data_yaml = {
        "path": str(out_root_path),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: cfg["labeling"]["class_name"]},
    }
    data_yaml_path = out_root_path / "data.yaml"
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    print({k: len(v) for k, v in splits.items()})
    print(f"data.yaml written to {data_yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split the pseudo-labeled pool into train/val/test for YOLO.")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--out-root", default="data/yolo_dataset")
    args = parser.parse_args()
    build_dataset(args.config, args.out_root)
