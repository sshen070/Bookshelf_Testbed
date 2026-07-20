"""Grow the reference bank with a new confirmed-normal snapshot.

This is the ONLY sanctioned way to add to data/reference/ -- it's additive
only (never deletes or overwrites existing references) and requires a human
to explicitly confirm the image is genuinely normal before it's added.

Why manual, not automatic: the reference bank is the ground-truth anchor for
the whole system. If a live monitoring run could silently promote its own
captures into "normal", a real anomaly could quietly become the new baseline
the next time something drifts near it -- exactly the failure mode this
project exists to catch. Growing the bank is a deliberate, human-in-the-loop
action: typically done right after you've physically returned the shelf to
its settled state following a staged perturbation, when a few mm of
placement jitter relative to the founding reference(s) is expected and
normal rather than a genuine change. The founding references are never
touched, so long-timescale drift is still measurable against them if needed
later -- this only adds a closer match for day-to-day jitter.

Usage:
    python -m src.add_reference data/raw/some_session/frame_003.jpg
    python -m src.add_reference data/raw/some_session/frame_003.jpg --yes --no-preview   # non-interactive/headless
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import cv2

from src.lighting import load_reference_bank, normalize_illumination, select_best_reference
from src.registration import align_to_reference
from src.ssim_diff import compute_dissimilarity_map, diss_map_to_boxes
from src.utils.config import load_pipeline_config, resolve_path
from src.utils.imio import imread_any


def add_reference(
    image_path: str,
    cfg_path: str = "configs/pipeline.yaml",
    auto_confirm: bool = False,
    show_preview: bool = True,
) -> None:
    cfg = load_pipeline_config(cfg_path)
    reference_dir = resolve_path(cfg["paths"]["reference_dir"])
    reference_bank = load_reference_bank(reference_dir)

    img_path = Path(image_path)
    img = imread_any(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read {img_path}")

    ref_name, ref_img = select_best_reference(img, reference_bank, cfg["lighting"], full_pipeline_cfg=cfg)
    reg = align_to_reference(img, ref_img, cfg["registration"])
    if not reg.success:
        print(
            f"WARNING: could not register {img_path} against any existing reference "
            f"(method={reg.method}). Adding it anyway would give the bank an unaligned "
            "entry that would corrupt every future diff against it -- refusing."
        )
        return

    aligned_norm = normalize_illumination(reg.aligned, cfg["lighting"])
    ref_norm = normalize_illumination(ref_img, cfg["lighting"])
    diss_map = compute_dissimilarity_map(aligned_norm, ref_norm, cfg)
    boxes, _ = diss_map_to_boxes(diss_map, cfg)

    print(f"Closest existing reference: {ref_name}")
    print(f"Residual dissimilarity against it: mean={diss_map.mean():.4f}, {len(boxes)} candidate box(es) remain.")
    if boxes:
        print(
            "NOTE: the classical pipeline's own threshold still flags a change against the "
            "closest existing reference. Make sure this is really just settling jitter, not "
            "an unstaged change, before adding it as 'normal'."
        )

    if show_preview:
        h, w = img.shape[:2]
        side_by_side = cv2.hconcat(
            [cv2.resize(ref_img, (w // 2, h // 2)), cv2.resize(reg.aligned, (w // 2, h // 2))]
        )
        cv2.imshow("closest reference (left) vs new candidate, aligned (right) -- press any key", side_by_side)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if not auto_confirm:
        answer = input(f"Add {img_path.name} to the reference bank at {reference_dir}? [y/N] ").strip().lower()
        if answer != "y":
            print("Not added.")
            return

    out_name = f"confirmed_{datetime.now().strftime('%Y%m%dT%H%M%S')}{img_path.suffix.lower()}"
    out_path = reference_dir / out_name
    shutil.copy2(img_path, out_path)
    print(f"Added {out_path}. The bank now has {len(reference_bank) + 1} reference image(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add a confirmed-normal image to the reference bank (additive only).")
    parser.add_argument("image", help="Path to the candidate normal-state image.")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--yes", dest="auto_confirm", action="store_true", help="Skip the interactive y/n prompt.")
    parser.add_argument(
        "--no-preview", dest="show_preview", action="store_false", help="Skip the cv2.imshow preview (headless/SSH)."
    )
    args = parser.parse_args()
    add_reference(args.image, args.config, args.auto_confirm, args.show_preview)
