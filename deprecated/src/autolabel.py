"""Turn raw shelf captures into YOLO pseudo-labels via the classical pipeline.

Per image: pick best-matching reference -> register -> normalize lighting ->
multi-scale SSIM diff -> bounding boxes -> YOLO-format label file. Every
image also gets one line in events.jsonl, which is the ground truth the
trained YOLO model will later be cross-validated against.

Images that fail registration are NOT silently labeled -- they're logged and
excluded from the training pool, because diffing against a misaligned
reference produces garbage boxes that would poison training.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
from tqdm import tqdm

from src.lighting import load_reference_bank, normalize_illumination, select_best_reference
from src.registration import align_to_reference
from src.ssim_diff import diss_map_to_boxes, compute_dissimilarity_map
from src.utils.config import load_pipeline_config, resolve_path
from src.utils.imio import IMAGE_EXTS as IMG_EXTS
from src.utils.imio import imread_any


def _write_yolo_label(label_path: Path, boxes, img_w: int, img_h: int, class_id: int = 0) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w", encoding="utf-8") as f:
        for box in boxes:
            cx, cy, w, h = box.to_yolo(img_w, img_h)
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def process_image(raw_path: Path, reference_bank, cfg: dict, out_dirs: dict) -> dict:
    session = raw_path.parent.name
    img = imread_any(raw_path)
    if img is None:
        return {"image": str(raw_path), "session": session, "status": "unreadable"}

    ref_name, ref_img = select_best_reference(img, reference_bank, cfg["lighting"], full_pipeline_cfg=cfg)
    reg = align_to_reference(img, ref_img, cfg["registration"])

    event = {
        "image": str(raw_path),
        "session": session,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reference_used": ref_name,
        "registration_method": reg.method,
        "reprojection_error_px": reg.reprojection_error_px,
    }

    if not reg.success:
        event["status"] = "registration_failed"
        return event

    aligned_norm = normalize_illumination(reg.aligned, cfg["lighting"])
    ref_norm = normalize_illumination(ref_img, cfg["lighting"])

    diss_map = compute_dissimilarity_map(aligned_norm, ref_norm, cfg)
    boxes, mask = diss_map_to_boxes(diss_map, cfg)

    stem = f"{session}__{raw_path.stem}"
    aligned_path = out_dirs["aligned"] / f"{stem}.jpg"
    diff_path = out_dirs["diffs"] / f"{stem}.png"
    pool_img_path = out_dirs["pool_images"] / f"{stem}.jpg"
    pool_label_path = out_dirs["pool_labels"] / f"{stem}.txt"

    cv2.imwrite(str(aligned_path), reg.aligned)
    cv2.imwrite(str(diff_path), (diss_map * 255).astype("uint8"))

    keep_background = cfg["labeling"]["keep_background_images"]
    if boxes or keep_background:
        cv2.imwrite(str(pool_img_path), reg.aligned)
        h, w = reg.aligned.shape[:2]
        _write_yolo_label(pool_label_path, boxes, w, h)

    event["status"] = "change" if boxes else "no_change"
    event["num_boxes"] = len(boxes)
    event["boxes_xyxy"] = [[b.x1, b.y1, b.x2, b.y2] for b in boxes]
    return event


def run_autolabel(cfg_path: str = "configs/pipeline.yaml") -> None:
    cfg = load_pipeline_config(cfg_path)
    reference_bank = load_reference_bank(resolve_path(cfg["paths"]["reference_dir"]))

    out_dirs = {
        "aligned": resolve_path(cfg["paths"]["aligned_dir"]),
        "diffs": resolve_path(cfg["paths"]["diff_dir"]),
        "pool_images": resolve_path(cfg["paths"]["labels_dir"]) / "images",
        "pool_labels": resolve_path(cfg["paths"]["labels_dir"]) / "labels",
    }
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    raw_paths = sorted(p for p in raw_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not raw_paths:
        print(f"No raw images found under {raw_dir}. Drop captures into per-session subfolders there first.")
        return

    events_path = resolve_path(cfg["paths"]["events_log"])
    events_path.parent.mkdir(parents=True, exist_ok=True)

    counts = {"change": 0, "no_change": 0, "registration_failed": 0, "unreadable": 0}
    with open(events_path, "w", encoding="utf-8") as ef:
        for raw_path in tqdm(raw_paths, desc="autolabeling"):
            event = process_image(raw_path, reference_bank, cfg, out_dirs)
            ef.write(json.dumps(event) + "\n")
            counts[event["status"]] = counts.get(event["status"], 0) + 1

    print(f"Done. {counts}")
    print(f"Events logged to {events_path}")
    print(f"Pseudo-labeled pool: {out_dirs['pool_images']} / {out_dirs['pool_labels']}")
    if counts["registration_failed"]:
        print(
            f"WARNING: {counts['registration_failed']} images failed registration and were "
            "excluded from the pool -- inspect these before trusting the dataset."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-label raw shelf images via the classical change-detection pipeline.")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()
    run_autolabel(args.config)
