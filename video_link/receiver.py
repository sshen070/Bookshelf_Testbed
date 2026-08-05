"""Video receiver -- runs ON THE SERVER (Jetson Xavier), decodes the stream.

Standalone: imports nothing from the rest of this repo.

    python receiver.py --serve 8080         # view it in a browser (SSH-friendly)
    python receiver.py --record out.mkv     # save what arrives
    python receiver.py                      # just prove frames are arriving
    python receiver.py --display            # a real screen, or ssh -X

VIEWING IT OVER SSH
-------------------
`--serve` decodes the stream and re-serves it as MJPEG over HTTP, so it is
viewable in an ordinary browser with no client software and no X server. On a
headless box reached over SSH, forward the port from your laptop:

    ssh -L 8080:localhost:8080 user@server      # on your laptop
    # then open http://localhost:8080/

This beats `ssh -X` with a video sink, which ships uncompressed frames over the
SSH channel and stutters badly even on a LAN.

Unlike the sender, the decode side CAN use NVIDIA hardware: `nvv4l2decoder`
exists on the Xavier and, unusually, on the Orin Nano too -- that SKU lost its
encoder but kept NVDEC. `--decoder auto` prefers it and falls back to the
software avdec_h264, so the same script runs on either box.

The receiver is the LISTENER and the sender is the caller, so start this first.
That direction also means only the server needs a reachable port; the Jetson can
sit behind NAT.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

BOUNDARY = "frameboundary"


class LatestFrame:
    """Holds the most recent JPEG, and wakes HTTP threads when it changes.

    Only the newest frame is kept -- a viewer that falls behind should skip to
    the present rather than replay a backlog, which is the whole point of a live
    view. A Condition rather than a poll loop so an idle viewer costs nothing.
    """

    def __init__(self):
        self._condition = threading.Condition()
        self._jpeg: bytes | None = None
        self._sequence = 0
        self._last_time = 0.0
        self._bytes = 0

    def publish(self, jpeg: bytes) -> None:
        with self._condition:
            self._jpeg = jpeg
            self._sequence += 1
            self._last_time = time.time()
            self._bytes += len(jpeg)
            self._condition.notify_all()

    def wait_for_next(self, last_seen: int, timeout: float = 5.0):
        """Block until a frame newer than `last_seen` exists. -> (jpeg, sequence)"""
        with self._condition:
            if self._sequence <= last_seen:
                self._condition.wait(timeout)
            return self._jpeg, self._sequence

    def status(self) -> dict:
        """What the page needs to explain itself when the picture is blank."""
        with self._condition:
            age = time.time() - self._last_time if self._last_time else None
            return {
                "frames": self._sequence,
                "bytes": self._bytes,
                # A stream that stopped 10s ago is a different problem from one
                # that never started, and the page says so.
                "age_seconds": round(age, 1) if age is not None else None,
                "live": age is not None and age < 2.0,
            }


INDEX_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Camera feed</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#0d0d0d; color:#c3c2b7;
         font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
         display:flex; flex-direction:column; min-height:100vh; }
  header { padding:10px 16px; font-size:13px; border-bottom:1px solid #2c2c2a;
           display:flex; gap:10px; align-items:center; }
  .dot { font-size:11px; }
  .live { color:#0ca30c; } .stalled { color:#fab219; } .none { color:#d03b3b; }
  main { flex:1; display:flex; align-items:center; justify-content:center;
         padding:16px; position:relative; }
  img { max-width:100%; max-height:100%; object-fit:contain; background:#000; }
  #overlay { position:absolute; text-align:center; max-width:38em; padding:20px;
             background:rgba(13,13,13,0.94); border:1px solid #2c2c2a; border-radius:8px; }
  #overlay h2 { margin:0 0 10px; font-size:15px; color:#fff; }
  #overlay p { margin:6px 0; font-size:13px; line-height:1.5; }
  code { background:#1a1a19; padding:1px 5px; border-radius:3px; font-size:12px; }
  .hidden { display:none; }
</style></head>
<body>
  <header>
    <span id="dot" class="dot none">&#9679;</span>
    <span id="label">connecting&hellip;</span>
    <span style="color:#898781">&middot; <code>/stream.mjpg</code></span>
  </header>
  <main>
    <img id="feed" src="/stream.mjpg" alt="live camera feed">
    <div id="overlay">
      <h2>No frames received yet</h2>
      <p>The receiver is running and this page is served by it, but nothing has
         arrived on the video port.</p>
      <p><strong>The sender is the other half.</strong> On the Jetson, run:</p>
      <p><code>python3 sender.py --host &lt;this-server-ip&gt; --port 8890 --source test</code></p>
      <p style="color:#898781">If it is already running, check that its
         <code>--host</code> is this machine and that UDP is not firewalled.</p>
    </div>
  </main>
<script>
// The <img> alone cannot distinguish "no sender" from "sender died" from
// "still starting" -- they all render as a black rectangle. Poll the receiver
// for what it actually knows and say so.
async function poll() {
  try {
    const r = await fetch('/status.json', {cache: 'no-store'});
    const s = await r.json();
    const dot = document.getElementById('dot');
    const label = document.getElementById('label');
    const overlay = document.getElementById('overlay');
    if (s.live) {
      dot.className = 'dot live';
      label.textContent = s.frames + ' frames received';
      overlay.classList.add('hidden');
    } else if (s.frames > 0) {
      dot.className = 'dot stalled';
      label.textContent = 'stream stopped ' + s.age_seconds + 's ago (' + s.frames + ' frames)';
      overlay.classList.remove('hidden');
      overlay.querySelector('h2').textContent = 'Stream stopped';
    } else {
      dot.className = 'dot none';
      label.textContent = 'waiting for sender';
      overlay.classList.remove('hidden');
    }
  } catch (e) {
    document.getElementById('label').textContent = 'receiver unreachable';
  }
}
poll(); setInterval(poll, 1000);
</script>
</body></html>"""


def make_handler(frames: LatestFrame):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = INDEX_PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/status.json":
                body = json.dumps(frames.status()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/stream.mjpg":
                self._stream()
            else:
                self.send_error(404)

        def _stream(self):
            self.send_response(200)
            # multipart/x-mixed-replace is the MJPEG contract: each part
            # replaces the last, so a plain <img> tag animates with no client JS.
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.end_headers()
            seen = 0
            try:
                while True:
                    jpeg, seen = frames.wait_for_next(seen)
                    if jpeg is None:
                        continue
                    self.wfile.write(
                        f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                        f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                    )
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass  # viewer closed the tab; entirely normal

        def log_message(self, *args):
            pass  # one line per frame otherwise

    return Handler


def pick_decoder(preference: str) -> str:
    if preference == "software":
        return "avdec_h264"
    if preference == "hardware":
        return "nvv4l2decoder"
    registry = Gst.Registry.get()
    return "nvv4l2decoder" if registry.find_feature("nvv4l2decoder", Gst.ElementFactory.__gtype__) else "avdec_h264"


def build_pipeline(args, decoder: str) -> str:
    if args.transport == "srt":
        # `mode` is a property, not a URI query parameter -- see sender.py.
        source = (
            f'srtsrc uri="srt://:{args.port}" mode=listener latency={args.latency} '
            f"! tsdemux ! h264parse"
        )
    else:
        source = (
            f"udpsrc port={args.port} "
            f'caps="application/x-rtp,media=video,encoding-name=H264,payload=96" ! '
            f"rtph264depay ! h264parse"
        )

    if args.record:
        # Remux without re-encoding: no reason to spend CPU recompressing what
        # already arrived compressed.
        return f"{source} ! queue ! matroskamux ! filesink location={args.record} sync=false"

    # nvv4l2decoder outputs NVMM memory and needs nvvidconv to reach anything
    # else; avdec_h264 outputs system memory and needs videoconvert.
    convert = "nvvidconv" if decoder == "nvv4l2decoder" else "videoconvert"

    if args.serve:
        scale = f"! videoscale ! video/x-raw,width={args.serve_width} " if args.serve_width else ""
        # sync=false and max-buffers=1/drop=true keep the browser on the newest
        # frame: a slow viewer should skip ahead, never accumulate lag.
        return (
            f"{source} ! {decoder} ! {convert} ! video/x-raw,format=I420 {scale}"
            f"! jpegenc quality={args.serve_quality} "
            f"! appsink name=out emit-signals=true max-buffers=1 drop=true sync=false"
        )

    if args.display:
        return f"{source} ! {decoder} ! {convert} ! autovideosink sync=false"

    return f"{source} ! {decoder} ! {convert} ! fpsdisplaysink name=sink video-sink=fakesink text-overlay=false sync=false"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Receive the Jetson's H.264 stream.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--port", type=int, default=8890, help="port to listen on for the video stream")
    p.add_argument("--transport", choices=["udp", "srt"], default="udp",
                   help="must match the sender; udp is the verified path")
    p.add_argument("--latency", type=int, default=200, help="SRT latency budget, ms; must match the sender")
    p.add_argument("--serve", type=int, metavar="HTTPPORT",
                   help="re-serve as MJPEG on this port; view at http://localhost:<port>/")
    p.add_argument("--serve-quality", type=int, default=80, help="JPEG quality for --serve")
    p.add_argument("--serve-width", type=int, help="downscale before serving, e.g. 640 over a slow tunnel")
    p.add_argument("--record", metavar="FILE", help="write the stream to a .mkv instead")
    p.add_argument("--display", action="store_true", help="open a window (needs a real display or ssh -X)")
    p.add_argument("--decoder", choices=["auto", "hardware", "software"], default="auto")
    p.add_argument("--print-pipeline", action="store_true")
    args = p.parse_args(argv)

    if sum(bool(x) for x in (args.serve, args.record, args.display)) > 1:
        print("Pick one of --serve, --record, --display.", file=sys.stderr)
        return 2

    Gst.init(None)
    decoder = pick_decoder(args.decoder)
    description = build_pipeline(args, decoder)
    if args.print_pipeline:
        print(description)
        return 0

    print(f"decoder: {decoder}")
    print(f"pipeline: {description}\n", flush=True)
    try:
        pipeline = Gst.parse_launch(description)
    except GLib.Error as e:
        print(f"Could not build pipeline: {e}", file=sys.stderr)
        return 1

    loop = GLib.MainLoop()
    state = {"first": False}

    httpd = None
    if args.serve:
        frames = LatestFrame()

        def on_new_sample(sink):
            sample = sink.emit("pull-sample")
            if sample is None:
                return Gst.FlowReturn.OK
            buffer = sample.get_buffer()
            ok, info = buffer.map(Gst.MapFlags.READ)
            if ok:
                try:
                    frames.publish(bytes(info.data))
                finally:
                    buffer.unmap(info)
                if not state["first"]:
                    state["first"] = True
                    print("FIRST FRAME DECODED -- open the page now", flush=True)
            return Gst.FlowReturn.OK

        pipeline.get_by_name("out").connect("new-sample", on_new_sample)
        httpd = ThreadingHTTPServer(("0.0.0.0", args.serve), make_handler(frames))
        httpd.daemon_threads = True
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        print(f"MJPEG server on :{args.serve}")
        print(f"  over SSH:  ssh -L {args.serve}:localhost:{args.serve} <user>@<this-host>")
        print(f"  then open: http://localhost:{args.serve}/\n", flush=True)

        # A silent receiver looks identical whether the sender is missing or the
        # decode is broken. Say which, on the console as well as the page.
        def nag():
            snapshot = frames.status()
            if snapshot["frames"] == 0:
                print(f"  ...still no frames on :{args.port}. Is sender.py running "
                      f"with --host pointing here?", flush=True)
            return True

        GLib.timeout_add_seconds(5, nag)

    def on_message(_bus, message):
        if message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"ERROR: {err}", file=sys.stderr)
            if debug:
                print(f"  {debug}", file=sys.stderr)
            loop.quit()
        elif message.type == Gst.MessageType.EOS:
            print("sender disconnected (EOS)")
            loop.quit()
        elif message.type == Gst.MessageType.ELEMENT:
            structure = message.get_structure()
            if structure and structure.get_name() == "fps-measurements":
                if not state["first"]:
                    state["first"] = True
                    print("FIRST FRAMES ARRIVING -- link is up", flush=True)
                print(f"  {structure.get_value('fps-display'):.1f} fps  "
                      f"(avg {structure.get_value('fps-average'):.1f})", flush=True)
        return True

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message)

    # Ctrl-C must reach the mainloop, not kill the process outright: with
    # --record, an .mkv whose muxer never finalised has no index and most
    # players refuse to open it. See the EOS handling in the finally block.
    def on_sigint():
        print("\nstopping", flush=True)
        loop.quit()
        return GLib.SOURCE_REMOVE

    GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGINT, on_sigint)

    if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        print("Pipeline failed to start.", file=sys.stderr)
        return 1

    print(f"listening on :{args.port} ({args.transport}) -- start the sender now", flush=True)
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        if args.record:
            pipeline.send_event(Gst.Event.new_eos())
            pipeline.get_bus().timed_pop_filtered(3 * Gst.SECOND, Gst.MessageType.EOS)
        pipeline.set_state(Gst.State.NULL)
        if httpd is not None:
            httpd.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
