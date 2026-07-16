"""Wipe derived pipeline artifacts to start clean for a new round of sessions.

By default this only clears what autolabel/build_dataset/train regenerate
from your raw captures: aligned frames, diff maps, the pseudo-label pool,
the yolo_dataset train/val/test split, and the event logs. It deliberately
leaves data/raw/ (your captured photos) and data/reference/ (your curated
baseline bank) untouched, since those are irreplaceable and everything else
can be rebuilt from them with `python run_pipeline.py all`.

--raw and --runs additionally wipe raw captures and trained model weights
respectively -- both require --yes since neither is recoverable from what
remains.

Usage:
    python -m src.reset_pipeline --dry-run          # see what would be cleared
    python -m src.reset_pipeline                    # clear derived artifacts only
    python -m src.reset_pipeline --raw --yes        # also wipe data/raw/ captures
    python -m src.reset_pipeline --runs --yes       # also wipe trained model weights
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.utils.config import load_pipeline_config, resolve_path


def _clear_dir_contents(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    for child in path.iterdir():
        n += 1
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    return n


def reset(
    cfg_path: str = "configs/pipeline.yaml",
    wipe_raw: bool = False,
    wipe_runs: bool = False,
    dry_run: bool = False,
) -> None:
    cfg = load_pipeline_config(cfg_path)
    paths = cfg["paths"]
    events_log = resolve_path(paths["events_log"])

    dir_targets = [
        resolve_path(paths["aligned_dir"]),
        resolve_path(paths["diff_dir"]),
        resolve_path(paths["labels_dir"]) / "images",
        resolve_path(paths["labels_dir"]) / "labels",
        resolve_path("data/yolo_dataset/images"),
        resolve_path("data/yolo_dataset/labels"),
    ]
    file_targets = [
        events_log,
        events_log.parent / "live_events.jsonl",
        events_log.parent / "crossvalidation.jsonl",
        resolve_path("data/yolo_dataset/data.yaml"),
    ]

    if wipe_raw:
        dir_targets.append(resolve_path(paths["raw_dir"]))
    if wipe_runs:
        dir_targets.append(resolve_path("runs/yolo"))

    label = "DRY RUN -- would clear" if dry_run else "Clearing"
    print(f"{label}:")
    for d in dir_targets:
        print(f"  dir : {d}")
    for f in file_targets:
        print(f"  file: {f}")

    if dry_run:
        return

    for d in dir_targets:
        _clear_dir_contents(d)
        d.mkdir(parents=True, exist_ok=True)
    for f in file_targets:
        if f.exists():
            f.unlink()

    print("Done.")
    if not wipe_raw:
        print(f"Left untouched: {resolve_path(paths['raw_dir'])} (raw captures)")
    if not wipe_runs:
        print(f"Left untouched: {resolve_path('runs/yolo')} (trained model weights)")
    print(f"Always left untouched: {resolve_path(paths['reference_dir'])} (reference bank)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Wipe derived pipeline artifacts (aligned/diff/pool/dataset/logs) to reset for a new round of sessions."
    )
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--raw", action="store_true", help="Also wipe data/raw/ captures (irreversible -- requires --yes).")
    parser.add_argument("--runs", action="store_true", help="Also wipe runs/yolo/ trained models (irreversible -- requires --yes).")
    parser.add_argument("--yes", action="store_true", help="Required to actually delete anything when --raw or --runs is set.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting anything.")
    args = parser.parse_args()

    if (args.raw or args.runs) and not args.yes and not args.dry_run:
        parser.error("--raw/--runs delete irreplaceable data -- pass --yes to confirm, or --dry-run to preview first.")

    reset(args.config, wipe_raw=args.raw, wipe_runs=args.runs, dry_run=args.dry_run)
