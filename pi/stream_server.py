"""MJPEG camera stream server -- runs ON THE RASPBERRY PI.

The Pi's only job is "grab a frame, serve a frame". All the heavy lifting
(registration, SSIM diffing, YOLO inference) happens on the PC, in
src/live_infer.py, which connects to this server over the network. Keeping
the Pi this dumb means it can run headless indefinitely without ever needing
torch/ultralytics installed on it.

Captures via picamera2 (official Pi Camera Module) if available, otherwise
falls back to OpenCV's camera capture (USB webcam) -- pass --usb-cam to force
the fallback even if picamera2 is installed.

Setup on the Pi (Raspberry Pi OS, camera module):
    sudo apt install -y python3-picamera2 python3-opencv
    python3 stream_server.py --port 8000

Setup on the Pi (USB webcam only, no picamera2 needed):
    pip install -r requirements-pi.txt
    python3 stream_server.py --usb-cam --port 8000

Then from the PC, point src/live_infer.py at this Pi's IP:port.
"""
from __future__ import annotations

import argparse
import socketserver
import threading
import time
from http import server

import cv2

try:
    from picamera2 import Picamera2

    _HAS_PICAMERA2 = True
except ImportError:
    _HAS_PICAMERA2 = False


class FrameGrabber:
    """Continuously grabs frames in a background thread so HTTP requests
    always get served the latest frame instead of blocking on capture."""

    def __init__(self, width: int, height: int, fps: int, use_usb_cam: bool, cam_index: int):
        self.fps = fps
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._stop = threading.Event()
        self._cap = None
        self._picam = None

        if use_usb_cam or not _HAS_PICAMERA2:
            self._cap = cv2.VideoCapture(cam_index)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._cap.set(cv2.CAP_PROP_FPS, fps)
            if not self._cap.isOpened():
                raise RuntimeError(f"Could not open camera index {cam_index}")
        else:
            self._picam = Picamera2()
            config = self._picam.create_video_configuration(main={"size": (width, height), "format": "RGB888"})
            self._picam.configure(config)
            self._picam.start()

    def _read_frame(self):
        if self._picam is not None:
            frame = self._picam.capture_array()
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, frame = self._cap.read()
        return frame if ok else None

    def start(self) -> None:
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        period = 1.0 / self.fps
        while not self._stop.is_set():
            t0 = time.time()
            frame = self._read_frame()
            if frame is not None:
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    with self._lock:
                        self._jpeg = buf.tobytes()
            time.sleep(max(0.0, period - (time.time() - t0)))

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def stop(self) -> None:
        self._stop.set()
        if self._cap is not None:
            self._cap.release()
        if self._picam is not None:
            self._picam.stop()


_grabber: FrameGrabber | None = None


class StreamHandler(server.BaseHTTPRequestHandler):
    def log_message(self, format, *args) -> None:
        pass  # keep the Pi's console quiet

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
                    frame = _grabber.latest_jpeg()
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
                pass  # client disconnected -- normal, not an error
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
    global _grabber
    parser = argparse.ArgumentParser(description="MJPEG camera stream server for the Raspberry Pi.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--usb-cam", action="store_true", help="Force OpenCV/USB webcam capture instead of picamera2.")
    parser.add_argument("--cam-index", type=int, default=0, help="OpenCV camera index, only used with --usb-cam.")
    args = parser.parse_args()

    _grabber = FrameGrabber(args.width, args.height, args.fps, args.usb_cam, args.cam_index)
    _grabber.start()

    httpd = ThreadingHTTPServer(("", args.port), StreamHandler)
    print(f"Serving MJPEG stream on http://0.0.0.0:{args.port}/stream.mjpg  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _grabber.stop()


if __name__ == "__main__":
    main()
