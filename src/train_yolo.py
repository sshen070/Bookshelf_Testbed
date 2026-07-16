"""Train the YOLO 'change' detector on the classical pipeline's pseudo-labels.

This model is a bootstrapped, noisy-label detector, not the final source of
truth -- its job is to be a fast, deployable stand-in for the SSIM pipeline.
Its detections get cross-validated against the classical pipeline's own
events.jsonl log in infer_and_crossvalidate.py before you trust it.
"""
from __future__ import annotations

import argparse

from ultralytics import YOLO

from src.utils.config import load_yolo_config, resolve_path


def train(cfg_path: str = "configs/yolo.yaml") -> None:
    cfg = load_yolo_config(cfg_path)
    data_yaml = resolve_path(cfg["data_yaml_out"])
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"{data_yaml} not found. Run src/autolabel.py then src/build_dataset.py first."
        )

    model = YOLO(cfg["model"]["base_weights"])
    model.train(
        data=str(data_yaml),
        imgsz=cfg["model"]["imgsz"],
        epochs=cfg["train"]["epochs"],
        patience=cfg["train"]["patience"],
        batch=cfg["train"]["batch"],
        lr0=cfg["train"]["lr0"],
        freeze=cfg["train"]["freeze"],
        hsv_h=cfg["train"]["hsv_h"],
        hsv_s=cfg["train"]["hsv_s"],
        hsv_v=cfg["train"]["hsv_v"],
        flipud=cfg["train"]["flipud"],
        fliplr=cfg["train"]["fliplr"],
        mosaic=cfg["train"]["mosaic"],
        degrees=cfg["train"]["degrees"],
        translate=cfg["train"]["translate"],
        scale=cfg["train"]["scale"],
        project=str(resolve_path(cfg["project"])),
        name=cfg["name"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the YOLO change detector.")
    parser.add_argument("--config", default="configs/yolo.yaml")
    args = parser.parse_args()
    train(args.config)
