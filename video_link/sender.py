"""Video sender -- runs ON THE JETSON ORIN NANO, streams H.264 to the server.

Standalone: imports nothing from the rest of this repo.

    python sender.py --host 10.11.1.106 --source test    # prove the link
    python sender.py --host 10.11.1.106 --source csi     # the real camera

WHY x264enc AND NOT nvv4l2h264enc
---------------------------------
The Orin Nano has **no hardware video encoder**. NVIDIA shipped NVDEC but
dropped NVENC from this SKU, so `nvv4l2h264enc`, `nvv4l2h265enc`, `omxh264enc`
and `v4l2h264enc` are all absent -- NVIDIA's nvvideo4linux2 plugin provides only
`nvv4l2decoder` here. Any pipeline copied from an AGX/NX/Xavier example dies
with "no element nvv4l2h264enc". Encoding is therefore on the CPU via x264enc,
measured on this board at 93 fps for 720p unthrottled, so 30 fps has room.

This is not purely a loss. x264enc honours a `bitrate` change on a live PLAYING
pipeline (measured: 4359 kbps -> 527 kbps within a second of the property set,
no teardown, no stream gap), which is precisely what the NVIDIA encoder elements
are unreliable about. The adaptive-bitrate controller this is groundwork for
wants exactly that.

THE CAMERA IS SINGLE-CONSUMER
-----------------------------
The CSI sensor emits raw Bayer (RG10) and can only be debayered through Argus,
i.e. `nvarguscamerasrc` -- which is also why ffmpeg is not an option for capture
here: pointed at /dev/video0 it sees Bayer it cannot process. WAFT's
camera_event_detector.py already holds that camera. Two processes cannot both
open it, so `--source csi` WILL FAIL, or fight, while WAFT is running. Use
`--source test` to commission the network path, and see the README for feeding
frames from WAFT's existing capture loop via appsrc instead.
"""
from __future__ import annotations

import argparse
import base64
import json
import signal
import sys
import time

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402


def build_pipeline(args) -> str:
    if args.source == "csi":
        # NVMM caps keep the frames in hardware memory until nvvidconv; the
        # sensor cannot produce anything x264enc understands without Argus.
        source = (
            f"nvarguscamerasrc sensor-id={args.sensor_id} ! "
            f"video/x-raw(memory:NVMM),width={args.width},height={args.height},"
            f"framerate={args.fps}/1 ! "
            f"nvvidconv ! video/x-raw,format=I420"
        )
    else:
        source = (
            f"videotestsrc is-live=true pattern={args.pattern} ! "
            f"video/x-raw,width={args.width},height={args.height},"
            f"framerate={args.fps}/1,format=I420"
        )

    # key-int-max bounds how long a receiver that joins late waits for a
    # keyframe; config-interval=-1 repeats SPS/PPS ahead of each one. Without
    # both, a receiver that connects mid-stream or reconnects after a drop sits
    # on a black frame until the next random IDR. Under event-triggered
    # streaming that "joins late" case is now the NORMAL case, so this matters
    # more here than it did for a continuous stream.
    encode = (
        f"x264enc name=enc bitrate={args.bitrate} speed-preset={args.preset} "
        f"tune=zerolatency key-int-max={args.fps} ! "
        f"h264parse config-interval=-1"
    )

    # The pre-roll buffer. With the gate shut, encoded frames pile up here and
    # the oldest are dropped (leaky=downstream), so it always holds the last
    # `preroll` seconds. Opening the gate flushes that backlog first, which is
    # what makes the stream start BEFORE the moment of the trigger -- the
    # footage of a rock starting to fall, not just the aftermath.
    #
    # Buffering encoded H.264 rather than raw frames is what makes this cheap:
    # 5s at 2 Mbps is ~1.2 MB, where 5s of raw 720p is ~1.4 GB.
    # Bounded by BUFFERS, not by time. max-size-time alone did not bound this in
    # testing -- blocked for 8s with a 4s time limit, the queue held all 8s, so
    # a camera that is never approved would grow without limit until it is
    # killed. A frame count is enforced reliably: measured 85 frames (~2.8s)
    # retained over an 8s block, versus unbounded growth with the time limit.
    preroll_frames = max(1, int(args.preroll * args.fps))
    preroll = (
        f"queue name=preroll max-size-buffers={preroll_frames} "
        f"max-size-time=0 max-size-bytes=0 leaky=downstream"
    )

    # The gate. Shut, nothing reaches the network at all; open, it flows. The
    # same valve serves two masters -- the allocator shuts it to shed a camera,
    # and the event logic shuts it when no approved stream is running -- so
    # whichever wants it closed wins. Dropping here rather than upstream of the
    # encoder keeps the pipeline running and timestamps continuous, so opening
    # is a property set rather than a renegotiation.
    # drop-mode matters as much as drop. The default, drop-all, discards sticky
    # events (caps, segment) along with buffers, so downstream never negotiates
    # -- and a valve shut at startup then back-pressures the whole pipeline,
    # including the OTHER branch of the tee. Measured: with drop-all the
    # snapshot branch produced 0 frames in 6s; with forward-sticky-events, 6.
    # Starts open: this valve now serves only the allocator's shed decision, and
    # nothing is shed at startup. Event-gating is the blocking probe instead --
    # see _apply_gate for why the two cannot be the same mechanism.
    gate = "valve name=gate drop=false drop-mode=forward-sticky-events"

    # async=false is REQUIRED whenever the gate can be shut. A sink defaults to
    # holding the pipeline's state change until it has prerolled a buffer, but
    # behind a closed valve no buffer ever arrives -- so the sink sticks at
    # READY, the pipeline sits in PAUSED, and a live source in PAUSED produces
    # nothing at all. That starves the snapshot branch through the tee as well,
    # which is how this showed up: zero thumbnails, and the cause three elements
    # away from the symptom. Measured: default -> state=paused, 0 snapshots;
    # async=false -> state=playing, 6 snapshots in 6s.
    no_preroll = "async=false" if args.trigger_mode else ""

    if args.transport == "srt":
        # `mode` is a PROPERTY, not a URI query parameter. GStreamer 1.20's
        # srtsink documents its uri as plain "srt://address:port" and silently
        # ignores "?mode=caller", leaving the default -- which is how you end up
        # with both ends as callers and a link that never connects.
        sink = (
            f'srtsink name=sink uri="srt://{args.host}:{args.port}" '
            f"mode=caller latency={args.latency} {no_preroll}"
        )
        mux = "mpegtsmux"
    else:
        # RTP over UDP: no handshake, no retransmit, no link stats -- but it is
        # the path verified working on this board, so it is the default.
        sink = f"udpsink name=sink host={args.host} port={args.port} sync=false {no_preroll}"
        mux = "rtph264pay config-interval=1 pt=96"

    # This queue only decouples the tee's branches; it must not become a second,
    # unaccounted pre-roll buffer. Left at its defaults (200 buffers ~= 6.7s at
    # 30fps) it silently added its own backlog on top of the real pre-roll
    # queue, so a 5s pre-roll delivered ~26s of video.
    decouple = "queue max-size-buffers=5 max-size-time=0 max-size-bytes=0 leaky=downstream"
    video_branch = f"{decouple} ! {encode} ! {preroll} ! {gate} ! {mux} ! {sink}"

    if not args.trigger_mode:
        return f"{source} ! {video_branch}"

    # Second branch: a low-rate thumbnail the camera can attach to an event so
    # the server has something to verify BEFORE granting the expensive stream.
    # Throttled to 1 fps and scaled down, so it costs almost nothing to run
    # continuously -- and it must run continuously, because a thumbnail is only
    # useful if it already exists at the instant the trigger fires.
    # Height is computed rather than left to negotiation: videoscale given only
    # a width will happily letterbox or distort, and a stretched thumbnail is a
    # bad input to a verifier that may be running a classifier on it. Rounded to
    # even because I420 chroma is subsampled 2x and odd dimensions are invalid.
    snapshot_height = max(2, int(round(args.height * args.snapshot_width / args.width / 2)) * 2)
    snapshot_branch = (
        f"queue leaky=downstream max-size-buffers=2 ! videorate ! "
        f"video/x-raw,framerate=1/1 ! videoscale ! "
        f"video/x-raw,width={args.snapshot_width},height={snapshot_height} ! "
        f"jpegenc quality=60 ! appsink name=snap emit-signals=true "
        f"max-buffers=1 drop=true sync=false"
    )
    return f"{source} ! tee name=split  split. ! {video_branch}  split. ! {snapshot_branch}"


class BitrateControl:
    """Applies the server's bandwidth grant to the live encoder.

    The whole scheme rests on one measured fact: x264enc honours a `bitrate`
    change on a PLAYING pipeline (4359 -> 527 kbps within a second, no teardown,
    no stream gap). No restart, no reconnect, no hole in the feed.

    Announce -> grant -> apply -> report, over MQTT:

        video/<id>/announce   (sender, retained)  what this camera can use
        video/<id>/bitrate    (server)            its grant, in kbps
        video/<id>/stats      (sender)            what it is actually sending

    The announce is retained so an allocator that starts later still learns the
    camera's bounds without waiting for it to restart.
    """

    def __init__(self, encoder, gate, preroll_pad, camera_id: str, args):
        self.encoder = encoder
        self.gate = gate
        self.preroll_pad = preroll_pad
        self.block_probe = None
        self.camera_id = camera_id
        self.args = args
        self.client = None
        self.current_kbps = args.bitrate
        self.shed = False
        self.grants_applied = 0
        self._last_bytes = 0
        self._last_time = time.time()

        # Event-triggered state. `streaming` is "the server has approved a
        # window that has not expired"; in trigger mode the gate stays shut
        # until that is true, so the default is NOT sending.
        self.streaming = not args.trigger_mode
        self.stream_until = 0.0
        self.pending_event = None
        self.latest_snapshot = None
        self._last_event_sent = 0.0

    def connect(self, host: str, port: int) -> bool:
        import paho.mqtt.client as mqtt

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"video-{self.camera_id}")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        try:
            self.client.connect(host, port, keepalive=60)
        except OSError as e:
            print(f"Control channel: could not reach broker {host}:{port}: {e}", file=sys.stderr)
            return False
        self.client.loop_start()
        return True

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        if reason_code.is_failure:
            print(f"Control channel refused: {reason_code}", file=sys.stderr)
            return
        # Subscribe from on_connect so an automatic reconnect re-subscribes; a
        # resumed session that never re-subscribed would silently stop receiving
        # grants and sit at whatever bitrate it last had.
        topic = f"video/{self.camera_id}/bitrate"
        client.subscribe(topic, qos=1)
        # stream: the server's verdict on an event we raised.
        # trigger: how a local detector (WAFT) tells us something happened --
        #   a topic rather than a socket so the detector needs no knowledge of
        #   this process, only of the broker it already talks to.
        client.subscribe(f"video/{self.camera_id}/stream", qos=1)
        client.subscribe(f"video/{self.camera_id}/trigger", qos=1)
        client.publish(
            f"video/{self.camera_id}/announce",
            json.dumps({
                "camera_id": self.camera_id,
                "weight": self.args.weight,
                "min_kbps": self.args.min_bitrate,
                "max_kbps": self.args.max_bitrate,
                "width": self.args.width,
                "height": self.args.height,
                "fps": self.args.fps,
            }),
            qos=1, retain=True,
        )
        print(f"control: connected, listening for grants on {topic}", flush=True)

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload)
        except json.JSONDecodeError as e:
            print(f"control: ignoring malformed message ({e})", file=sys.stderr)
            return

        if msg.topic.endswith("/trigger"):
            self.raise_event(payload if isinstance(payload, dict) else {})
            return
        if msg.topic.endswith("/stream"):
            if isinstance(payload, dict):
                self._handle_stream_decision(payload)
            return

        try:
            if isinstance(payload, dict):
                kbps = int(payload["kbps"])
                shed = bool(payload.get("shed", False))
            else:
                kbps, shed = int(payload), False
        except (KeyError, TypeError, ValueError) as e:
            print(f"control: ignoring malformed grant ({e})", file=sys.stderr)
            return

        # Shedding is a separate decision from bitrate, and must be handled
        # before the clamp below. A shed camera is told kbps=0, which the clamp
        # would raise back to min_bitrate -- leaving it streaming at its floor
        # and freeing none of the bandwidth shedding was supposed to reclaim.
        if shed != self.shed:
            self.shed = shed
            GLib.idle_add(self._set_gate, shed)
        if shed:
            return

        # Clamp. The allocator should respect the announced bounds, but a sender
        # that trusts a remote number unconditionally can be told to emit
        # 100 Mbps by one buggy or stale publisher.
        kbps = max(self.args.min_bitrate, min(self.args.max_bitrate, kbps))
        if kbps == self.current_kbps:
            return

        previous = self.current_kbps
        self.current_kbps = kbps
        self.grants_applied += 1

        # paho delivers on its own network thread. Marshal the property set onto
        # the GLib main loop rather than touching a PLAYING pipeline from an
        # arbitrary thread.
        def apply():
            self.encoder.set_property("bitrate", kbps)
            print(f"control: bitrate {previous} -> {kbps} kbps", flush=True)
            return False

        GLib.idle_add(apply)

    def _set_gate(self, shed: bool) -> bool:
        self.shed = shed
        self._apply_gate()
        print("control: SHED -- gate closed" if shed else "control: un-shed", flush=True)
        return False

    def _apply_gate(self) -> None:
        """Two gates, closed for different reasons and behaving differently.

        SHED (allocator: no bandwidth) uses the valve, which DROPS. Buffering
        through a shed would hand back a burst of stale video at the exact
        moment bandwidth became available again -- the opposite of what
        shedding was for.

        NOT-YET-APPROVED (no live event) uses a blocking pad probe, which
        HOLDS. The leaky queue upstream then keeps the most recent
        `--preroll` seconds and discards the rest, so releasing the block
        flushes footage from *before* the trigger. That is the whole point of
        pre-roll, and it is why a valve cannot serve here: a valve in drop mode
        consumes buffers, so nothing ever accumulates behind it.
        """
        self.gate.set_property("drop", self.shed)

        want_flowing = self.streaming and not self.shed
        if want_flowing and self.block_probe is not None:
            self.preroll_pad.remove_probe(self.block_probe)
            self.block_probe = None
            # The flush starts at an arbitrary point in a GOP, so the receiver
            # has nothing decodable until the next IDR. Force one immediately.
            self.encoder.send_event(
                Gst.Event.new_custom(
                    Gst.EventType.CUSTOM_UPSTREAM, Gst.Structure.new_empty("GstForceKeyUnit")
                )
            )
        elif not want_flowing and self.block_probe is None and self.args.trigger_mode:
            self.block_probe = self.preroll_pad.add_probe(
                Gst.PadProbeType.BLOCK_DOWNSTREAM, lambda pad, info: Gst.PadProbeReturn.OK
            )

    # ---- event-triggered streaming --------------------------------------

    def on_snapshot(self, sink) -> int:
        """Keep the newest thumbnail, so one is ready the instant a trigger fires."""
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK
        buffer = sample.get_buffer()
        ok, info = buffer.map(Gst.MapFlags.READ)
        if ok:
            try:
                first = self.latest_snapshot is None
                self.latest_snapshot = bytes(info.data)
                if first:
                    # Worth saying once: an event with no thumbnail still gets
                    # judged, just on metadata alone, and that is easy to miss.
                    print(f"snapshot: thumbnails ready ({len(self.latest_snapshot)} bytes)",
                          flush=True)
            finally:
                buffer.unmap(info)
        return Gst.FlowReturn.OK

    def raise_event(self, payload: dict) -> None:
        """Report a detection and ask permission to stream.

        Deliberately does NOT open the gate. The camera proposes; the server
        disposes. Streaming on local detection alone is what floods a
        constrained link with false positives -- which is the whole reason for
        a verification step.
        """
        if self.client is None:
            return
        now = time.time()
        if now - self._last_event_sent < self.args.event_min_interval:
            return  # local rate limit; the server has its own, stricter one
        self._last_event_sent = now

        event_id = f"{self.camera_id}-{int(now * 1000)}"
        self.pending_event = event_id
        message = {
            "event_id": event_id,
            "camera_id": self.camera_id,
            "timestamp": now,
            "trigger": payload.get("trigger", "unknown"),
            "confidence": payload.get("confidence"),
            "boxes": payload.get("boxes"),
            "labels": payload.get("labels"),
            "requested_seconds": payload.get("requested_seconds", self.args.stream_seconds),
            "preroll_seconds": self.args.preroll,
            "streaming_already": self.streaming,
        }
        if self.latest_snapshot is not None:
            message["snapshot_jpeg_b64"] = base64.b64encode(self.latest_snapshot).decode()
            message["snapshot_bytes"] = len(self.latest_snapshot)
        self.client.publish(f"video/{self.camera_id}/event", json.dumps(message), qos=1)
        print(f"event: {event_id} raised ({message['trigger']}, "
              f"conf={message['confidence']}), awaiting verification", flush=True)

    def _handle_stream_decision(self, payload: dict) -> None:
        approved = bool(payload.get("approved"))
        reason = payload.get("reason", "")
        if not approved:
            print(f"event: DENIED by server ({reason})", flush=True)
            return

        seconds = float(payload.get("duration_seconds", self.args.stream_seconds))
        self.stream_until = time.time() + seconds
        if not self.streaming:
            self.streaming = True
            GLib.idle_add(self._open_for_stream, reason, seconds)
        else:
            # Already live: a re-approval extends the window rather than
            # restarting the stream, so continuous motion does not produce a
            # sequence of stop/start gaps in the recording.
            print(f"event: APPROVED, window extended to {seconds:.0f}s", flush=True)

    def _open_for_stream(self, reason: str, seconds: float) -> bool:
        self._apply_gate()
        print(f"event: APPROVED ({reason}) -- streaming {seconds:.0f}s "
              f"with {self.args.preroll}s pre-roll", flush=True)
        return False

    def check_stream_expiry(self) -> bool:
        """Close the gate when the granted window runs out."""
        if self.streaming and time.time() >= self.stream_until:
            self.streaming = False
            self._apply_gate()
            print("event: window expired -- gate closed", flush=True)
            if self.client is not None:
                self.client.publish(
                    f"video/{self.camera_id}/stream_ended",
                    json.dumps({"camera_id": self.camera_id, "event_id": self.pending_event}),
                    qos=1,
                )
        return True

    def publish_stats(self, sink) -> bool:
        """Report what is actually going out, so the allocator can see reality.

        The grant is a target; what the encoder emits varies around it with
        scene complexity. A static wall may use half its grant, and the
        allocator can only reclaim that if somebody measures it.
        """
        if self.client is None:
            return True
        now = time.time()
        total = sink.get_property("bytes-served") if sink.find_property("bytes-served") else None
        measured = None
        if total is not None:
            elapsed = now - self._last_time
            if elapsed > 0:
                measured = round((total - self._last_bytes) * 8 / elapsed / 1000)
            self._last_bytes, self._last_time = total, now
        self.client.publish(
            f"video/{self.camera_id}/stats",
            json.dumps({
                "camera_id": self.camera_id,
                "granted_kbps": 0 if self.shed else self.current_kbps,
                "measured_kbps": 0 if self.shed else measured,
                "shed": self.shed,
                "grants_applied": self.grants_applied,
                "timestamp": now,
            }),
            qos=0,
        )
        return True

    def close(self):
        if self.client is not None:
            self.client.disconnect()
            self.client.loop_stop()


def report_stats(sink, encoder) -> bool:
    """Print the SRT link stats that the adaptive controller will later act on.

    Send-buffer depth is the signal that matters: it grows when the encoder is
    outproducing what the link can drain, and it does so *before* loss appears.
    Loss is a lagging indicator -- by the time it shows up, frames are already
    gone.
    """
    stats = sink.get_property("stats")
    if stats is None:
        return True
    buffered = stats.get_value("send-buffer-level-ms") or stats.get_value("bytes-sent") or 0
    lost = stats.get_value("packets-sent-lost") or 0
    rtt = stats.get_value("rtt-ms") or 0
    print(
        f"  bitrate={encoder.get_property('bitrate')}kbps  "
        f"rtt={rtt}ms  lost={lost}  buffer={buffered}",
        flush=True,
    )
    return True


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Stream H.264 from the Jetson to the server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host", required=True, help="server address")
    p.add_argument("--port", type=int, default=8890, help="server port")
    p.add_argument("--source", choices=["test", "csi"], default="test",
                   help="'test' proves the link without touching the camera WAFT owns")
    p.add_argument("--transport", choices=["udp", "srt"], default="udp",
                   help="udp is verified working here; srt is not yet (see README)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--bitrate", type=int, default=2000, help="kbps (x264enc units)")
    p.add_argument("--preset", default="ultrafast",
                   help="x264 speed preset; ultrafast/superfast keep CPU away from WAFT")
    p.add_argument("--latency", type=int, default=200, help="SRT latency budget, ms")
    p.add_argument("--sensor-id", type=int, default=0)
    p.add_argument("--pattern", default="ball", help="videotestsrc pattern for --source test")
    p.add_argument("--stats", action="store_true", help="print link stats once a second")
    p.add_argument("--print-pipeline", action="store_true", help="show the pipeline and exit")

    control = p.add_argument_group("server-coordinated bandwidth")
    control.add_argument("--control-host", help="MQTT broker the allocator publishes grants on")
    control.add_argument("--control-port", type=int, default=1883)
    control.add_argument("--camera-id", help="identifies this feed to the allocator (default: hostname)")
    control.add_argument("--weight", type=float, default=1.0,
                         help="how much this feed matters, relative to the others")
    control.add_argument("--min-bitrate", type=int, default=300,
                         help="kbps below which this feed is not worth sending at all")
    control.add_argument("--max-bitrate", type=int, default=8000,
                         help="kbps beyond which extra bits buy nothing")

    trigger = p.add_argument_group("event-triggered streaming")
    trigger.add_argument("--trigger-mode", action="store_true",
                         help="send nothing until an event is raised AND the server approves")
    trigger.add_argument("--preroll", type=float, default=5.0,
                         help="seconds of pre-trigger footage to buffer and send first")
    trigger.add_argument("--stream-seconds", type=float, default=30.0,
                         help="how long to stream per approval, unless the server says otherwise")
    trigger.add_argument("--event-min-interval", type=float, default=2.0,
                         help="seconds between events this camera will raise (local rate limit)")
    trigger.add_argument("--snapshot-width", type=int, default=320,
                         help="width of the thumbnail attached to each event for verification")
    args = p.parse_args(argv)

    if args.trigger_mode and not args.control_host:
        print("--trigger-mode needs --control-host: approval arrives over MQTT, and without\n"
              "a broker the gate would never open and nothing would ever be sent.",
              file=sys.stderr)
        return 2

    if args.min_bitrate > args.max_bitrate:
        print("--min-bitrate cannot exceed --max-bitrate", file=sys.stderr)
        return 2
    args.bitrate = max(args.min_bitrate, min(args.max_bitrate, args.bitrate))

    Gst.init(None)
    description = build_pipeline(args)
    if args.print_pipeline:
        print(description)
        return 0

    print(f"pipeline: {description}\n", flush=True)
    try:
        pipeline = Gst.parse_launch(description)
    except GLib.Error as e:
        print(f"Could not build pipeline: {e}", file=sys.stderr)
        return 1

    loop = GLib.MainLoop()

    def on_message(_bus, message):
        if message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"ERROR: {err}", file=sys.stderr)
            if debug:
                print(f"  {debug}", file=sys.stderr)
            loop.quit()
        elif message.type == Gst.MessageType.EOS:
            print("end of stream")
            loop.quit()
        return True

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message)

    # GLib.MainLoop.run() blocks inside C, so Python's own SIGINT handler never
    # runs and Ctrl-C would kill the process without unwinding -- taking the
    # muxer's finalisation with it. This routes the signal through GLib instead.
    def on_sigint():
        print("\nstopping", flush=True)
        loop.quit()
        return GLib.SOURCE_REMOVE

    GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGINT, on_sigint)

    if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        print("Pipeline failed to start.", file=sys.stderr)
        if args.source == "csi":
            print("With --source csi this usually means WAFT already holds the camera.", file=sys.stderr)
        return 1

    print(f"streaming to {args.host}:{args.port} over {args.transport} "
          f"({args.width}x{args.height}@{args.fps}, {args.bitrate}kbps)", flush=True)

    sink, encoder = pipeline.get_by_name("sink"), pipeline.get_by_name("enc")

    if args.stats and args.transport == "srt":
        GLib.timeout_add_seconds(1, report_stats, sink, encoder)

    control = None
    if args.control_host:
        import socket as _socket

        preroll_element = pipeline.get_by_name("preroll")
        control = BitrateControl(
            encoder, pipeline.get_by_name("gate"),
            preroll_element.get_static_pad("src") if preroll_element else None,
            args.camera_id or _socket.gethostname(), args,
        )
        if args.trigger_mode:
            # Engage the block before anything can flow, so a triggered camera
            # never leaks video it was not granted.
            control._apply_gate()
        if control.connect(args.control_host, args.control_port):
            print(f"control: camera-id={control.camera_id} weight={args.weight} "
                  f"range={args.min_bitrate}-{args.max_bitrate}kbps", flush=True)
            GLib.timeout_add_seconds(2, control.publish_stats, sink)
            if args.trigger_mode:
                snap = pipeline.get_by_name("snap")
                if snap is not None:
                    snap.connect("new-sample", control.on_snapshot)
                # Checked once a second rather than scheduled at grant time: a
                # timer set when the window opens would be wrong the moment the
                # server extended it, and polling a deadline cannot drift.
                GLib.timeout_add_seconds(1, control.check_stream_expiry)
                print(f"trigger mode: gate SHUT, {args.preroll}s pre-roll buffering. "
                      f"Waiting for an event on video/{control.camera_id}/trigger", flush=True)
        else:
            # Deliberately not fatal: a camera that cannot reach the allocator
            # should keep streaming at its starting bitrate rather than go dark.
            print("control: continuing without coordination", file=sys.stderr)
            control = None

    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        if control is not None:
            control.close()
        pipeline.set_state(Gst.State.NULL)
    return 0


if __name__ == "__main__":
    sys.exit(main())
