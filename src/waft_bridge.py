"""Bridge from this pipeline into WAFT's flow-based box extractor.

WAFT lives in its own environment (torch/timm/xformers -- see
WAFT/WAFT/README.md), separate from this project's lightweight opencv/
scikit-image/ultralytics venv (WAFT needs timm, which isn't and shouldn't be
a dependency of this project). So this bridges via subprocess, invoking
WAFT/WAFT/flow_boxes_demo.py with whatever Python executable WAFT's own
environment provides, rather than importing WAFT's modules directly.

IMPORTANT -- what WAFT's boxes represent is DIFFERENT from this project's
main "change" boxes. WAFT measures MOTION between two temporally-close
frames (something actively moving), not a settled STATE DIFFERENCE against a
fixed reference captured at an arbitrary earlier time -- which is what
src/ssim_diff.py measures, and what the trained YOLO model learns. Don't
blindly merge WAFT's output into build_dataset.py's "change" pseudo-label
pool: "something moved between frame a and frame b" and "there is now a
lasting structural difference from the reference" are not the same claim.
This bridge writes WAFT's boxes to their own location for inspection/cross-
reference rather than silently treating them as more "change" ground truth.
A sensible use: feed WAFT's motion trigger frames into data/raw/<session>/
as new capture events, then let autolabel.py's classical pipeline (which
already knows how to judge a frame against the reference bank) decide
whether it's a lasting change -- same principle as live_infer.py already
uses for the Pi's continuous feed, just with a motion-aware trigger instead
of always-diff-against-reference.

Usage:
    python -m src.waft_bridge --waft-python /path/to/waft-venv/python.exe \\
        --frames data/raw/some_session/frame_001.jpg data/raw/some_session/frame_002.jpg \\
        --output-dir outputs/flow_boxes
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run_waft_boxes(
    waft_python: str,
    waft_dir: str,
    frame_paths: list[str],
    cfg: str,
    ckpt: str,
    output_dir: str,
    extra_args: list[str] | None = None,
) -> dict:
    """Run WAFT's flow_boxes_demo.py as a subprocess and return its per-pair results.

    frame_paths must be given in temporal order; consecutive pairs are
    processed (frame 0->1, 1->2, ...), matching flow_boxes_demo.py's pairing.
    """
    waft_dir_path = Path(waft_dir)
    script = waft_dir_path / "flow_boxes_demo.py"
    if not script.exists():
        raise FileNotFoundError(
            f"{script} not found -- is --waft-dir pointing at the WAFT repo root (e.g. WAFT/WAFT)?"
        )

    abs_frames = [str(Path(p).resolve()) for p in frame_paths]
    abs_output_dir = str(Path(output_dir).resolve())

    cmd = [
        waft_python,
        "flow_boxes_demo.py",
        "--cfg", cfg,
        "--ckpt", ckpt,
        "--output-dir", abs_output_dir,
        "--frames", *abs_frames,
    ]
    if extra_args:
        cmd.extend(extra_args)

    print("[waft_bridge] Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(waft_dir_path), check=True)

    summary_path = Path(abs_output_dir) / "summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    for r in results:
        with open(r["boxes_json"], "r", encoding="utf-8") as f:
            r["boxes"] = json.load(f)
    return {"output_dir": abs_output_dir, "pairs": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run WAFT flow-based box extraction on a sequence of frames via subprocess."
    )
    parser.add_argument(
        "--waft-python",
        required=True,
        help="Path to the Python executable in WAFT's own environment (see WAFT/WAFT/README.md).",
    )
    parser.add_argument("--waft-dir", default="WAFT/WAFT", help="Path to the WAFT repo root.")
    parser.add_argument("--frames", nargs="+", required=True, help="Ordered frame paths; consecutive pairs are processed.")
    parser.add_argument("--cfg", default="config/a2/twins/chairs-things.json")
    parser.add_argument("--ckpt", default="checkpoints/a2/twins/zero-shot.pth")
    parser.add_argument("--output-dir", default="outputs/flow_boxes")
    parser.add_argument("--magnitude-threshold", type=float, default=2.0, help="Flow magnitude (px) to count as motion.")
    args = parser.parse_args()

    result = run_waft_boxes(
        args.waft_python,
        args.waft_dir,
        args.frames,
        args.cfg,
        args.ckpt,
        args.output_dir,
        extra_args=["--magnitude-threshold", str(args.magnitude_threshold)],
    )
    for pair in result["pairs"]:
        print(f"pair {pair['pair_index']}: {pair['num_boxes']} boxes -> {pair['overlay']}")
