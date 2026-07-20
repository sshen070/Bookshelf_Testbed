"""Live, on-device motion-box detection for the Raspberry Pi / Jetson Nano.

Runs entirely on the edge device using classical Farneback optical flow
(cv2.calcOpticalFlowFarneback) -- no torch, no WAFT, no ultralytics. Farneback
at a small processing resolution (default 160x120) costs a few milliseconds
per frame, which is what makes continuous on-device operation feasible on a
Pi's CPU or a Jetson Nano. WAFT's deep model needs real GPU/CPU budget this
class of device doesn't have -- that's why the WAFT integration
(src/waft_bridge.py) stays a PC-side, offline/batch tool instead.

Box extraction mirrors the same pipeline used elsewhere in this project
(src/ssim_diff.py for the classical SSIM path, WAFT/WAFT/flow_boxes_demo.py
for WAFT's flow path): threshold flow magnitude -> morphological cleanup ->
connected components -> solidity filter (rejects thin/sprawling shapes, e.g.
a swaying cable, that aren't a compact moving object) -> merge overlapping ->
cap count. Deliberately re-implemented here with only cv2+numpy rather than
imported from src/, so this script has zero dependency on the rest of the
project and can be copied to the Pi/Jetson entirely on its own.

This process OWNS the camera, like stream_server.py does -- run this
INSTEAD of stream_server.py, not alongside it, since most camera backends
(picamera2, a USB webcam via V4L2) only support one open handle at a time.
It still serves the live feed over HTTP, annotated with detected boxes, plus
a JSON endpoint with the latest detections, so a PC can watch or poll it the
same way it would stream_server.py's plain feed.

Detected-motion frames are also saved locally (events/<session_timestamp>/)
so this can run autonomously and accumulate candidate sessions even with no
PC connected. Copy/rsync that directory into this project's data/raw/ later
and run `python run_pipeline.py autolabel` as usual -- the classical pipeline
judges each frame against the reference bank the same way regardless of
whether it arrived via a manual capture or an on-device motion trigger.

Usage (on the Pi/Jetson):
    python3 motion_boxes.py --port 8000
    python3 motion_boxes.py --usb-cam --port 8000
Then from a PC/browser: http://<device-ip>:8000/stream.mjpg (annotated view)
                         http://<device-ip>:8000/boxes.json (latest detection)
"""
from __future__ import annotations

import argparse
import json
import os
import socketserver
import threading
import time
from datetime import datetime
from http import server

import cv2
import numpy as np

try:
    from picamera2 import Picamera2

    _HAS_PICAMERA2 = True
except ImportError:
    _HAS_PICAMERA2 = False


# ---------------------------------------------------------------------------
# Box extraction -- see module docstring: intentionally duplicated from
# src/ssim_diff.py / WAFT/WAFT/flow_boxes_demo.py, not imported.
# ---------------------------------------------------------------------------

def _iou(a, b):
    ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, a["x2"] - a["x1"]) * max(0, a["y2"] - a["y1"])
    area_b = max(0, b["x2"] - b["x1"]) * max(0, b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _merge_overlapping(boxes, iou_threshold):
    merged = []
    used = [False] * len(boxes)
    for i, box in enumerate(boxes):
        if used[i]:
            continue
        group = [box]
        used[i] = True
        for j in range(i + 1, len(boxes)):
            if not used[j] and _iou(box, boxes[j]) >= iou_threshold:
                group.append(boxes[j])
                used[j] = True
        merged.append(
            {
                "x1": min(b["x1"] for b in group),
                "y1": min(b["y1"] for b in group),
                "x2": max(b["x2"] for b in group),
                "y2": max(b["y2"] for b in group),
            }
        )
    return merged


def magnitude_to_boxes(magnitude, args):
    """magnitude: 2D float32 array (proc_h, proc_w) of per-pixel flow magnitude."""
    mask = (magnitude >= args.magnitude_threshold).astype(np.uint8) * 255
    if args.open_kernel > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((args.open_kernel, args.open_kernel), np.uint8))
    if args.close_kernel > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((args.close_kernel, args.close_kernel), np.uint8))

    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = magnitude.shape[:2]

    boxes = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < args.min_area:
            continue
        x, y, bw, bh = (int(v) for v in stats[label, cv2.CC_STAT_LEFT : cv2.CC_STAT_LEFT + 4])

        if args.min_solidity > 0:
            crop = (labels_im[y : y + bh, x : x + bw] == label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                hull_area = cv2.contourArea(cv2.convexHull(contour))
                solidity = cv2.contourArea(contour) / hull_area if hull_area > 0 else 1.0
                if solidity < args.min_solidity:
                    continue

        pad_x, pad_y = int(bw * args.pad_frac), int(bh * args.pad_frac)
        boxes.append(
            {
                "x1": max(0, x - pad_x),
                "y1": max(0, y - pad_y),
                "x2": min(w, x + bw + pad_x),
                "y2": min(h, y + bh + pad_y),
            }
        )

    boxes = _merge_overlapping(boxes, args.merge_iou_threshold)
    if args.max_boxes > 0 and len(boxes) > args.max_boxes:
        boxes.sort(key=lambda b: (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]), reverse=True)
        boxes = boxes[: args.max_boxes]
    return boxes


def scale_boxes(boxes, from_size, to_size):
    fw, fh = from_size
    tw, th = to_size
    sx, sy = tw / fw, th / fh
    return [
        {"x1": int(b["x1"] * sx), "y1": int(b["y1"] * sy), "x2": int(b["x2"] * sx), "y2": int(b["y2"] * sy)}
        for b in boxes
    ]


def draw_boxes(frame_bgr, boxes, color=(0, 0, 255)):
    out = frame_bgr.copy()
    for b in boxes:
        cv2.rectangle(out, (b["x1"], b["y1"]), (b["x2"], b["y2"]), color, 2)
    return out


# ---------------------------------------------------------------------------
# Camera capture + live motion-box processing, single owning thread.
# ---------------------------------------------------------------------------

class MotionBoxWorker:
    def __init__(self, args):
        self.args = args
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._latest_jpeg: bytes | None = None
        self._latest_boxes: dict = {"boxes": [], "timestamp": None, "num_boxes": 0}
        self._cap = None
        self._picam = None

        if args.usb_cam or not _HAS_PICAMERA2:
            self._cap = cv2.VideoCapture(args.cam_index)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            self._cap.set(cv2.CAP_PROP_FPS, args.fps)
            if not self._cap.isOpened():
                raise RuntimeError(f"Could not open camera index {args.cam_index}")
        else:
            self._picam = Picamera2()
            config = self._picam.create_video_configuration(main={"size": (args.width, args.height), "format": "RGB888"})
            self._picam.configure(config)
            self._picam.start()

        self._session_dir = None
        self._last_motion_t = 0.0
        self._last_save_t = 0.0
        os.makedirs(args.events_dir, exist_ok=True)

    def _read_frame_bgr(self):
        if self._picam is not None:
            frame_rgb = self._picam.capture_array()
            return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        ok, frame = self._cap.read()
        return frame if ok else None

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def latest_boxes(self) -> dict:
        with self._lock:
            return dict(self._latest_boxes)

    def start(self) -> None:
        threading.Thread(target=self._loop, daemon=True).start()

    def _maybe_save_event(self, frame_bgr, boxes_full_res):
        now = time.time()
        if not boxes_full_res:
            return
        if self._session_dir is None or (now - self._last_motion_t) > self.args.new_session_gap:
            session_name = f"motion_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
            self._session_dir = os.path.join(self.args.events_dir, session_name)
            os.makedirs(self._session_dir, exist_ok=True)
        self._last_motion_t = now

        if (now - self._last_save_t) < self.args.min_save_interval:
            return
        frame_name = f"{datetime.now().strftime('%H%M%S_%f')}.jpg"
        frame_path = os.path.join(self._session_dir, frame_name)
        cv2.imwrite(frame_path, frame_bgr)
        boxes_path = os.path.join(self._session_dir, frame_name.replace(".jpg", "_boxes.json"))
        with open(boxes_path, "w", encoding="utf-8") as f:
            json.dump({"boxes_xyxy": [[b["x1"], b["y1"], b["x2"], b["y2"]] for b in boxes_full_res]}, f)
        self._last_save_t = now
        print(f"[motion] saved {frame_path} ({len(boxes_full_res)} boxes)")

    def _loop(self) -> None:
        proc_size = (self.args.proc_width, self.args.proc_height)
        full_size = (self.args.width, self.args.height)
        prev_gray = None
        period = 1.0 / self.args.fps if self.args.fps > 0 else 0.0

        while not self._stop.is_set():
            t0 = time.time()
            frame_bgr = self._read_frame_bgr()
            if frame_bgr is None:
                time.sleep(0.05)
                continue

            proc_frame = cv2.resize(frame_bgr, proc_size)
            gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)

            boxes_proc = []
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                )
                magnitude = np.linalg.norm(flow, axis=2).astype(np.float32)
                boxes_proc = magnitude_to_boxes(magnitude, self.args)
            prev_gray = gray

            boxes_full = scale_boxes(boxes_proc, proc_size, full_size)
            annotated = draw_boxes(frame_bgr, boxes_full) if boxes_full else frame_bgr
            ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])

            with self._lock:
                if ok:
                    self._latest_jpeg = buf.tobytes()
                self._latest_boxes = {
                    "boxes": [[b["x1"], b["y1"], b["x2"], b["y2"]] for b in boxes_full],
                    "num_boxes": len(boxes_full),
                    "timestamp": time.time(),
                }

            self._maybe_save_event(frame_bgr, boxes_full)

            if period > 0:
                time.sleep(max(0.0, period - (time.time() - t0)))

    def stop(self) -> None:
        self._stop.set()
        if self._cap is not None:
            self._cap.release()
        if self._picam is not None:
            self._picam.stop()


_worker: MotionBoxWorker | None = None


class Handler(server.BaseHTTPRequestHandler):
    def log_message(self, format, *args) -> None:
        pass

    def do_GET(self) -> None:
        if self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            try:
                while True:
                    frame = _worker.latest_jpeg()
                    if frame is None:
                        time.sleep(0.05)
                        continue
                    self.wfile.write(b"--FRAME\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(frame)))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path == "/boxes.json":
            body = json.dumps(_worker.latest_boxes()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><img src='/stream.mjpg'></body></html>")
        else:
            self.send_response(404)
            self.end_headers()


class ThreadingHTTPServer(socketserver.ThreadingMixIn, server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> None:
    global _worker
    parser = argparse.ArgumentParser(description="Live, on-device Farneback motion-box detection for the Pi/Jetson.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--width", type=int, default=1280, help="Capture/stream resolution width.")
    parser.add_argument("--height", type=int, default=960, help="Capture/stream resolution height.")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--usb-cam", action="store_true", help="Force OpenCV/USB webcam capture instead of picamera2.")
    parser.add_argument("--cam-index", type=int, default=0)

    parser.add_argument("--proc-width", type=int, default=160, help="Downscaled width Farneback actually runs at.")
    parser.add_argument("--proc-height", type=int, default=120, help="Downscaled height Farneback actually runs at.")
    parser.add_argument("--magnitude-threshold", type=float, default=1.5, help="Flow magnitude (proc-res px) to count as motion.")
    parser.add_argument("--open-kernel", type=int, default=3)
    parser.add_argument("--close-kernel", type=int, default=5)
    parser.add_argument("--min-area", type=int, default=20, help="Min blob area in proc-res px (proc frame is much smaller than capture).")
    parser.add_argument("--min-solidity", type=float, default=0.3, help="0 disables the shape filter.")
    parser.add_argument("--pad-frac", type=float, default=0.08)
    parser.add_argument("--merge-iou-threshold", type=float, default=0.3)
    parser.add_argument("--max-boxes", type=int, default=20)

    parser.add_argument("--events-dir", default="events", help="Local directory to save motion-triggered frames into.")
    parser.add_argument("--min-save-interval", type=float, default=5.0, help="Seconds between saved frames within one event.")
    parser.add_argument("--new-session-gap", type=float, default=30.0, help="Seconds of no motion after which the next event starts a new session folder.")
    args = parser.parse_args()

    _worker = MotionBoxWorker(args)
    _worker.start()

    httpd = ThreadingHTTPServer(("", args.port), Handler)
    print(f"[INFO] Motion-box detection running. Stream: http://0.0.0.0:{args.port}/stream.mjpg  Boxes: /boxes.json")
    print(f"[INFO] Processing at {args.proc_width}x{args.proc_height}, capture at {args.width}x{args.height}")
    print(f"[INFO] Events saved to {os.path.abspath(args.events_dir)}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _worker.stop()


if __name__ == "__main__":
    main()