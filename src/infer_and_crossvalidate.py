"""Cross-validate the trained YOLO detector against the classical pipeline.

This is the check that answers "did YOLO learn genuine structural change, or
did it just learn to reproduce the classical pipeline's noise?" -- run both
detectors on the same (ideally held-out, or fresh) images and measure how
often they agree, both at the image level (changed vs not) and at the box
level (IoU overlap of *where* they say the change is).

Persistent disagreement in a specific direction is itself diagnostic:
- YOLO flags things classical doesn't: it may be generalizing (good) or
  hallucinating (bad) -- inspect the crops.
- Classical flags things YOLO doesn't: likely a class of change under-
  represented in the pseudo-label training pool -- feed more of that back in.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.lighting import load_reference_bank, normalize_illumination, select_best_reference
from src.registration import align_to_reference
from src.ssim_diff import BBox, compute_dissimilarity_map, diss_map_to_boxes
from src.utils.config import load_pipeline_config, resolve_path
from src.utils.imio import IMAGE_EXTS as IMG_EXTS
from src.utils.imio import imread_any


def classical_detect(img, reference_bank, cfg) -> tuple[list[BBox], str]:
    ref_name, ref_img = select_best_reference(img, reference_bank, cfg["lighting"], full_pipeline_cfg=cfg)
    reg = align_to_reference(img, ref_img, cfg["registration"])
    if not reg.success:
        return [], "registration_failed"
    aligned_norm = normalize_illumination(reg.aligned, cfg["lighting"])
    ref_norm = normalize_illumination(ref_img, cfg["lighting"])
    diss_map = compute_dissimilarity_map(aligned_norm, ref_norm, cfg)
    boxes, _ = diss_map_to_boxes(diss_map, cfg)
    return boxes, "ok"


def yolo_detect(model: YOLO, img, conf: float = 0.25) -> list[BBox]:
    results = model.predict(img, conf=conf, verbose=False)
    boxes = []
    for r in results:
        for xyxy in r.boxes.xyxy.tolist():
            boxes.append(BBox(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])))
    return boxes


def match_boxes(classical: list[BBox], yolo: list[BBox], iou_threshold: float = 0.3) -> dict:
    matched_c, matched_y = set(), set()
    ious = []
    for i, c in enumerate(classical):
        best_j, best_iou = None, 0.0
        for j, y in enumerate(yolo):
            if j in matched_y:
                continue
            iou = c.iou(y)
            if iou > best_iou:
                best_j, best_iou = j, iou
        if best_j is not None and best_iou >= iou_threshold:
            matched_c.add(i)
            matched_y.add(best_j)
            ious.append(best_iou)
    return {
        "n_classical": len(classical),
        "n_yolo": len(yolo),
        "n_matched": len(matched_c),
        "n_only_classical": len(classical) - len(matched_c),
        "n_only_yolo": len(yolo) - len(matched_y),
        "mean_matched_iou": sum(ious) / len(ious) if ious else None,
    }


def run_crossvalidation(
    image_dir: str,
    yolo_weights: str,
    pipeline_cfg_path: str = "configs/pipeline.yaml",
    out_path: str = "data/processed/crossvalidation.jsonl",
    conf: float = 0.25,
) -> None:
    cfg = load_pipeline_config(pipeline_cfg_path)
    reference_bank = load_reference_bank(resolve_path(cfg["paths"]["reference_dir"]))
    model = YOLO(yolo_weights)

    image_paths = sorted(p for p in Path(image_dir).rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not image_paths:
        print(f"No images found under {image_dir}")
        return

    out_file = resolve_path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    agree_both_change = agree_both_nochange = only_classical = only_yolo = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for img_path in image_paths:
            img = imread_any(img_path)
            if img is None:
                continue
            classical_boxes, status = classical_detect(img, reference_bank, cfg)
            if status != "ok":
                continue
            yolo_boxes = yolo_detect(model, img, conf=conf)
            match_stats = match_boxes(classical_boxes, yolo_boxes)

            classical_changed = len(classical_boxes) > 0
            yolo_changed = len(yolo_boxes) > 0
            if classical_changed and yolo_changed:
                agree_both_change += 1
            elif not classical_changed and not yolo_changed:
                agree_both_nochange += 1
            elif classical_changed and not yolo_changed:
                only_classical += 1
            else:
                only_yolo += 1

            record = {
                "image": str(img_path),
                "classical_changed": classical_changed,
                "yolo_changed": yolo_changed,
                **match_stats,
            }
            f.write(json.dumps(record) + "\n")

    total = agree_both_change + agree_both_nochange + only_classical + only_yolo
    agreement_rate = (agree_both_change + agree_both_nochange) / total if total else 0.0
    print(f"Cross-validated {total} images.")
    print(f"  Both flag change:    {agree_both_change}")
    print(f"  Both flag no-change: {agree_both_nochange}")
    print(f"  Only classical:      {only_classical}  (candidates for more pseudo-label training data)")
    print(f"  Only YOLO:           {only_yolo}  (inspect for hallucination vs. generalization)")
    print(f"  Image-level agreement rate: {agreement_rate:.3f}")
    print(f"Per-image details written to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-validate trained YOLO against the classical pipeline.")
    parser.add_argument("--image-dir", required=True, help="Directory of images to evaluate (e.g. a held-out session).")
    parser.add_argument("--weights", required=True, help="Path to trained YOLO weights (e.g. runs/yolo/change_detector/weights/best.pt).")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--out", default="data/processed/crossvalidation.jsonl")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    run_crossvalidation(args.image_dir, args.weights, args.config, args.out, args.conf)
