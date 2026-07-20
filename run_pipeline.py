"""Single entry point chaining the shelf change-detection pipeline stages.

Usage:
    python run_pipeline.py autolabel                 # raw images -> pseudo-labels + events.jsonl
    python run_pipeline.py build-dataset              # pool -> train/val/test + data.yaml
    python run_pipeline.py train                      # train YOLO on the dataset
    python run_pipeline.py crossvalidate --image-dir data/raw/holdout_session --weights runs/yolo/change_detector/weights/best.pt
    python run_pipeline.py live --pi-host 192.168.1.42 --weights runs/yolo/change_detector/weights/best.pt
    python run_pipeline.py add-reference data/raw/some_session/frame_003.jpg   # grow the reference bank (see src/add_reference.py)
    python run_pipeline.py reset --dry-run             # preview what a reset would clear
    python run_pipeline.py reset                       # clear derived artifacts, keep raw/reference/runs
    python run_pipeline.py waft-boxes --waft-python /path/to/waft-venv/python.exe --frames a.jpg b.jpg
    python run_pipeline.py all                        # autolabel -> build-dataset -> train
"""
from __future__ import annotations

import argparse

from src.add_reference import add_reference
from src.autolabel import run_autolabel
from src.build_dataset import build_dataset
from src.infer_and_crossvalidate import run_crossvalidation
from src.live_infer import run_live
from src.reset_pipeline import reset
from src.train_yolo import train
from src.waft_bridge import run_waft_boxes


def main() -> None:
    parser = argparse.ArgumentParser(description="Shelf change-detection pipeline orchestrator.")
    parser.add_argument(
        "stage",
        choices=["autolabel", "build-dataset", "train", "crossvalidate", "live", "add-reference", "reset", "waft-boxes", "all"],
    )
    parser.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    parser.add_argument("--yolo-config", default="configs/yolo.yaml")
    parser.add_argument("--image-dir", help="Required for crossvalidate: directory of images to evaluate.")
    parser.add_argument("--weights", help="Required for crossvalidate/live: path to trained YOLO weights.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--pi-host", default="192.168.1.197", help="Raspberry Pi's IP address or hostname (default: 192.168.1.197).")
    parser.add_argument("--pi-port", type=int, default=8000)
    parser.add_argument("--headless", action="store_true", help="live: no display window (for running over SSH).")
    parser.add_argument("image", nargs="?", help="Required for add-reference: path to the candidate normal-state image.")
    parser.add_argument("--yes", action="store_true", help="add-reference: skip confirmation prompt. reset: confirm --raw/--runs.")
    parser.add_argument("--no-preview", dest="show_preview", action="store_false", help="add-reference: skip the cv2.imshow preview.")
    parser.add_argument("--raw", action="store_true", help="reset: also wipe data/raw/ captures (irreversible -- requires --yes).")
    parser.add_argument("--runs", action="store_true", help="reset: also wipe runs/yolo/ trained models (irreversible -- requires --yes).")
    parser.add_argument("--dry-run", action="store_true", help="reset: show what would be deleted without deleting.")
    parser.add_argument("--waft-python", help="Required for waft-boxes: Python executable in WAFT's own environment.")
    parser.add_argument("--waft-dir", default="WAFT/WAFT", help="waft-boxes: path to the WAFT repo root.")
    parser.add_argument("--frames", nargs="+", help="Required for waft-boxes: ordered frame paths; consecutive pairs are processed.")
    parser.add_argument("--waft-cfg", default="config/a2/twins/chairs-things.json", help="waft-boxes: WAFT model config, relative to --waft-dir.")
    parser.add_argument("--waft-ckpt", default="checkpoints/a2/twins/zero-shot.pth", help="waft-boxes: WAFT checkpoint, relative to --waft-dir.")
    parser.add_argument("--waft-output-dir", default="outputs/flow_boxes", help="waft-boxes: where to write boxes/overlays.")
    parser.add_argument("--waft-magnitude-threshold", type=float, default=2.0, help="waft-boxes: flow magnitude (px) to count as motion.")
    args = parser.parse_args()

    if args.stage in ("autolabel", "all"):
        run_autolabel(args.pipeline_config)
    if args.stage in ("build-dataset", "all"):
        build_dataset(args.pipeline_config)
    if args.stage in ("train", "all"):
        train(args.yolo_config)
    if args.stage == "crossvalidate":
        if not args.image_dir or not args.weights:
            parser.error("crossvalidate requires --image-dir and --weights")
        run_crossvalidation(args.image_dir, args.weights, args.pipeline_config, conf=args.conf)
    if args.stage == "live":
        if not args.pi_host or not args.weights:
            parser.error("live requires --pi-host and --weights")
        run_live(args.pi_host, args.pi_port, args.weights, args.pipeline_config, conf=args.conf, headless=args.headless)
    if args.stage == "add-reference":
        if not args.image:
            parser.error("add-reference requires an image path")
        add_reference(args.image, args.pipeline_config, auto_confirm=args.yes, show_preview=args.show_preview)
    if args.stage == "reset":
        if (args.raw or args.runs) and not args.yes and not args.dry_run:
            parser.error("reset --raw/--runs delete irreplaceable data -- pass --yes to confirm, or --dry-run to preview first.")
        reset(args.pipeline_config, wipe_raw=args.raw, wipe_runs=args.runs, dry_run=args.dry_run)
    if args.stage == "waft-boxes":
        if not args.waft_python or not args.frames:
            parser.error("waft-boxes requires --waft-python and --frames")
        result = run_waft_boxes(
            args.waft_python,
            args.waft_dir,
            args.frames,
            args.waft_cfg,
            args.waft_ckpt,
            args.waft_output_dir,
            extra_args=["--magnitude-threshold", str(args.waft_magnitude_threshold)],
        )
        for pair in result["pairs"]:
            print(f"pair {pair['pair_index']}: {pair['num_boxes']} boxes -> {pair['overlay']}")


if __name__ == "__main__":
    main()
