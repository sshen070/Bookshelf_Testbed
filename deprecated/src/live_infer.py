"""Live inference against the Pi's MJPEG camera stream -- runs ON THE PC.

Connects to pi/stream_server.py, and on every frame runs the same two
detectors as infer_and_crossvalidate.py (imported directly, not
reimplemented, so live and batch cross-validation can't silently drift
apart): the classical registration+SSIM pipeline, and the trained YOLO model.

Side effect that matters for the bigger goal: every frame either detector
flags as changed gets saved into a fresh data/raw/live_<timestamp>/ session
folder. A monitoring run doubles as a data-collection run -- feed the result
back through `python run_pipeline.py autolabel` to grow the training set with
real (not staged) events over time, which is exactly the long-timescale
behavior that's supposed to transfer to the dam.

A new session directory is started whenever a change is detected after a
quiet gap (--new-session-gap seconds) with no change, so a single monitoring
run can still contain several independent "events" that get properly
separated for train/val/test splitting later -- consistent with the
one-folder-per-event rule in build_dataset.py.

Usage:
    python -m src.live_infer --pi-host 192.168.1.42 --weights runs/yolo/change_detector/weights/best.pt
    python -m src.live_infer --pi-host 192.168.1.42 --weights ...\\best.pt --headless   # for running over SSH, no display
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.infer_and_crossvalidate import classical_detect, match_boxes, yolo_detect
from src.lighting import load_reference_bank
from src.ssim_diff import BBox
from src.utils.config import load_pipeline_config, resolve_path

RECONNECT_DELAY_S = 3.0


def _open_stream(url: str) -> cv2.VideoCapture:
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG)


def _draw_boxes(img, boxes: list[BBox], color: tuple[int, int, int], label: str) -> None:
    for b in boxes:
        cv2.rectangle(img, (b.x1, b.y1), (b.x2, b.y2), color, 2)
        cv2.putText(img, label, (b.x1, max(0, b.y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def run_live(
    pi_host: str,
    pi_port: int,
    yolo_weights: str,
    pipeline_cfg_path: str = "configs/pipeline.yaml",
    conf: float = 0.25,
    headless: bool = False,
    save_changes: bool = True,
    min_save_interval_s: float = 5.0,
    new_session_gap_s: float = 30.0,
) -> None:
    cfg = load_pipeline_config(pipeline_cfg_path)
    reference_bank = load_reference_bank(resolve_path(cfg["paths"]["reference_dir"]))
    model = YOLO(yolo_weights)

    url = f"http://{pi_host}:{pi_port}/stream.mjpg"
    events_path = resolve_path(cfg["paths"]["events_log"]).parent / "live_events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])

    session_dir: Path | None = None
    last_change_t = 0.0
    last_save_t = 0.0

    print(f"Connecting to {url} ...")
    cap = _open_stream(url)

    with open(events_path, "a", encoding="utf-8") as ef:
        while True:
            if not cap.isOpened():
                print(f"Stream not open, retrying in {RECONNECT_DELAY_S}s...")
                time.sleep(RECONNECT_DELAY_S)
                cap = _open_stream(url)
                continue

            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"Lost frame, reconnecting in {RECONNECT_DELAY_S}s...")
                cap.release()
                time.sleep(RECONNECT_DELAY_S)
                cap = _open_stream(url)
                continue

            classical_boxes, status = classical_detect(frame, reference_bank, cfg)
            yolo_boxes = yolo_detect(model, frame, conf=conf) if status == "ok" else []
            changed = status == "ok" and (bool(classical_boxes) or bool(yolo_boxes))
            now = time.time()

            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "classical_changed": bool(classical_boxes),
                "yolo_changed": bool(yolo_boxes),
                **(match_boxes(classical_boxes, yolo_boxes) if status == "ok" else {}),
            }
            ef.write(json.dumps(record) + "\n")
            ef.flush()

            if changed:
                if session_dir is None or (now - last_change_t) > new_session_gap_s:
                    session_dir = raw_dir / f"live_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
                    session_dir.mkdir(parents=True, exist_ok=True)
                last_change_t = now
                if save_changes and (now - last_save_t) >= min_save_interval_s:
                    out_path = session_dir / f"{datetime.now().strftime('%H%M%S_%f')}.jpg"
                    cv2.imwrite(str(out_path), frame)
                    last_save_t = now
                    print(f"Change detected -> saved {out_path}")

            if not headless:
                annotated = frame.copy()
                _draw_boxes(annotated, classical_boxes, (0, 165, 255), "classical")
                _draw_boxes(annotated, yolo_boxes, (0, 255, 0), "yolo")
                status_text = "CHANGE" if changed else ("REG. FAILED" if status != "ok" else "normal")
                color = (0, 0, 255) if changed else (0, 200, 0)
                cv2.putText(annotated, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                cv2.imshow("Live shelf monitor", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    if not headless:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live inference against the Pi's MJPEG camera stream.")
    parser.add_argument("--pi-host", required=True, help="Raspberry Pi's IP address or hostname.")
    parser.add_argument("--pi-port", type=int, default=8000)
    parser.add_argument("--weights", required=True, help="Path to trained YOLO weights.")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--headless", action="store_true", help="No display window (for running over SSH).")
    parser.add_argument("--no-save", dest="save_changes", action="store_false", help="Don't save changed frames to data/raw/.")
    parser.add_argument("--min-save-interval", type=float, default=5.0, help="Seconds between saved frames within one event.")
    parser.add_argument("--new-session-gap", type=float, default=30.0, help="Seconds of no-change after which the next change starts a new session folder.")
    args = parser.parse_args()
    run_live(
        args.pi_host,
        args.pi_port,
        args.weights,
        args.config,
        args.conf,
        args.headless,
        args.save_changes,
        args.min_save_interval,
        args.new_session_gap,
    )
