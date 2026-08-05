# Video Link

Streams H.264 from the Jetson Orin Nano to the server. Standalone — imports
nothing from `vit_classifier/`, `WAFT/`, `device_status/` or `src/`.

- `sender.py` runs **on the Nano**.
- `receiver.py` runs **on the server** (Xavier).

## The finding that shapes everything: this Nano has no hardware encoder

NVIDIA shipped NVDEC but **dropped NVENC from the Orin Nano SKU**. Verified on
this board — NVIDIA's `nvvideo4linux2` GStreamer plugin registers exactly one
element here:

```
$ gst-inspect-1.0 nvvideo4linux2
  nvv4l2decoder: NVIDIA v4l2 video decoder      <- decode only, no encoder
```

`nvv4l2h264enc`, `nvv4l2h265enc`, `omxh264enc` and `v4l2h264enc` are all absent.
**Any pipeline copied from an AGX/NX/Xavier example fails immediately** with
`no element "nvv4l2h264enc"`. Encoding on the Nano is CPU-side via `x264enc`.

Measured on this board, 1280×720:

| Encoder | Throughput | Notes |
|---|---|---|
| `x264enc` ultrafast/zerolatency | **93 fps** | 3× headroom at 30 fps |
| `nvjpegenc` (HW JPEG) | 149 fps | MJPEG only, far higher bitrate |
| `jpegenc` (CPU JPEG) | 140 fps | HW JPEG is barely faster here |

The receiver is unaffected — the Xavier has full NVENC/NVDEC, and `receiver.py`
auto-selects `nvv4l2decoder` when present, falling back to `avdec_h264`.

### This is good news for adaptive bitrate

The usual warning about NVIDIA encoder elements is that they don't reliably
honour a bitrate change mid-stream. `x264enc` does. Measured here on a live
`PLAYING` pipeline, no teardown and no stream gap:

```
phase 1 (4000 kbps target):     4359 kbps measured
phase 2 ( 500 kbps target):      527 kbps measured
VERDICT: runtime mutation WORKS
```

So `enc.set_property("bitrate", n)` is a sound basis for the controller.

## Why GStreamer, not FFmpeg

The decisive reason is the camera, not the codec. The CSI sensor emits **raw
10-bit Bayer (`RG10`)** and can only be debayered through NVIDIA's Argus ISP,
i.e. `nvarguscamerasrc`. FFmpeg pointed at `/dev/video0` gets Bayer it cannot
process, and FFmpeg cannot drive Argus at all. That settles it before codecs or
bitrate control enter the argument.

Supporting reasons: the codebase already uses GStreamer (`WAFT/WAFT/camera_sources.py`
builds an `nvarguscamerasrc → BGR appsink` pipeline), and runtime bitrate
mutation needs in-process control that an FFmpeg subprocess can't give you.

FFmpeg is still the right tool for **file** work — clip concatenation with
`-c copy`, thumbnails, and `ffprobe`. Note it is **not currently installed** on
this Jetson.

## The camera is single-consumer

WAFT's `camera_event_detector.py` already holds the CSI camera. Two processes
cannot both open it, so **`--source csi` will fail or fight while WAFT is
running**. Use `--source test` to commission the network path — that is why it
is the default.

For production, the stream must be fed from frames WAFT already has, not from a
second capture. Either `tee` after `nvarguscamerasrc` inside the detector
process, or push its existing BGR numpy frames into an `appsrc`. The second is
less invasive, since `camera_sources.py` already hands frames to OpenCV.

## Usage

Start the receiver first — it listens, and the sender connects to it. That
direction also means only the server needs a reachable port.

```bash
# on the server
python receiver.py --port 8890 --serve 8080         # watch it in a browser
python receiver.py --port 8890 --record out.mkv     # save the stream
python receiver.py --port 8890                      # just prove frames arrive

# on the Nano
python sender.py --host <server-ip> --port 8890 --source test
python sender.py --host <server-ip> --port 8890 --source csi --bitrate 4000
```

`--print-pipeline` shows the constructed pipeline without running it.

## Watching the feed when the server is headless / SSH-only

`--serve PORT` decodes the incoming H.264 and re-serves it as **MJPEG over
HTTP**, so any browser can view it with no client software and no X server.
Forward the port from your laptop:

```bash
# on the server
python receiver.py --port 8890 --serve 8080

# on your laptop
ssh -L 8080:localhost:8080 user@server
# then open http://localhost:8080/
```

Prefer this to `ssh -X` with a video sink: X11 forwarding ships *uncompressed*
frames over the SSH channel and stutters badly even on a LAN. `--display` is
still there if you have a real screen.

### MJPEG is bandwidth-hungry — downscale for a tunnel

Every frame is an independent JPEG, so there is no inter-frame compression.
Measured here at 30 fps (`videotestsrc pattern=snow`, the worst case):

| Setting | Bandwidth |
|---|---|
| `--serve` (1280×720, quality 80) | **25.6 Mbps** |
| `--serve-width 640 --serve-quality 70` | **12.5 Mbps** (51% less, still 30 fps) |

Fine on a LAN, painful through a tunnel on a slow uplink. Real camera footage
compresses far better than `snow`; use it as an upper bound, not a forecast.
Drop `--serve-width` to 640 and quality to ~60 if the view stutters.

Only the *viewing* path pays this — the Nano→server link stays H.264 at
whatever `--bitrate` you set. MJPEG exists solely so a browser can render it.

Verified end to end: 90 JPEG frames in 3 s (exactly 30 fps, matching source),
each decoding to a valid 1280×720 image.

Requires `python3-gi` and `gstreamer1.0-plugins-{good,bad,ugly}` (`x264enc` is
in `-ugly`). Both were already present here.

## Server-coordinated bandwidth allocation

The server divides one link budget across all cameras by importance, and pushes
each camera its share. Cameras apply it to the **running** encoder — no restart,
no reconnect, no gap in the feed.

```bash
# server
python3 allocator.py --broker localhost --total 10000 --http-port 8090

# each Nano
python3 sender.py --host <server> --port 8890 \
    --control-host <broker> --camera-id spillway --weight 5 \
    --min-bitrate 500 --max-bitrate 6000
```

Three MQTT topics, reusing the broker you already run:

```
video/<id>/announce   camera -> server   bounds and weight   (retained)
video/<id>/bitrate    server -> camera   its grant, in kbps  (retained)
video/<id>/stats      camera -> server   what it is actually sending
```

Nothing is configured server-side: a camera exists because it announced, so
adding a Jetson needs no change to the allocator. The announce is **retained**,
so an allocator that starts later still learns each camera's bounds without
waiting for a restart.

### The policy

Lives in [`bandwidth.py`](bandwidth.py) as pure functions — no MQTT, no
GStreamer — so it can be reasoned about and tested on its own:

```bash
python3 bandwidth.py     # 15 checks across 7 scenarios
```

Budget is divided in proportion to weight, subject to each camera's
`[min, max]`. Two refinements matter more than the proportional split:

- **A camera at its cap hands back the surplus.** Pushing 4 Mbps at a static
  wall is waste; the excess is redistributed to cameras that can still use it.
- **Below its floor a camera is shed, not starved.** H.264 at 200 kbps cannot
  resolve a hairline crack, so those bits buy *nothing*. Better to run three
  cameras properly than five uselessly.

Redistribution is iterative (progressive filling): clamp, freeze the clamped,
re-divide the remainder among the rest, repeat.

Only 90% of the link is allocated. The encoder's bitrate is a target its output
varies around, not a hard ceiling, and RTP/UDP/IP headers add ~5–8% on top —
measured traffic ran ~1% above grant, comfortably inside that margin.

### Event-driven priority

`event_weight()` raises a camera's weight when WAFT reports motion, then decays
it back over `--decay` seconds. The decay matters as much as the boost: without
it, the first camera to ever trigger holds priority forever and the allocation
reflects history rather than what is happening now.

```bash
curl -X POST -d '{"camera_id":"spillway"}' http://server:8090/api/event
```

That is the hook for WAFT — call it when `camera_event_detector.py` fires.

### Dashboard and control API

`--http-port` serves a dashboard (live grants, weights, measured throughput) and:

| Route | Purpose |
|---|---|
| `POST /api/total` `{"kbps":5000}` | change the link budget |
| `POST /api/weight` `{"camera_id":"x","weight":5}` | re-prioritise a camera |
| `POST /api/event` `{"camera_id":"x"}` | mark motion; boosts then decays |
| `GET /api/state` | everything as JSON |

Over SSH: `ssh -L 8090:localhost:8090 user@server`.

### Verified end to end

Three cameras at weights 5:2:1 against a real broker, with traffic measured by
an **independent UDP counter** rather than trusting the senders' own reports:

| Link budget | spillway | crest | access |
|---|---|---|---|
| 10 Mbps | 5625 granted / **5664 measured** | 2250 / **2275** | 1125 / **1120** |
| 1.2 Mbps | 580 | 500 | **SHED — 0 kbps on the wire** |
| back to 9 Mbps | 5063 | 2025 | 1012 (resumed, keyframe forced) |

Event boost on `access`, the *lowest*-weight camera: it jumped to 2581 kbps
(above `crest`) and decayed back to 1012 over 30 s, with the others smoothly
reclaiming.

### Shedding needs a valve, not a bitrate of zero

A shed camera is told `kbps: 0`, but the sender clamps incoming grants into its
announced bounds as a safety measure — which silently raised that 0 back to
`min_bitrate`, leaving the camera streaming at its floor and freeing **none** of
the bandwidth shedding was supposed to reclaim. This was live in the first
implementation and caught only because the traffic was measured independently.

The fix is a `valve` element after `h264parse`: shedding sets `drop=true` and
nothing reaches the network. Dropping *there* rather than upstream of the
encoder is deliberate — bandwidth is the scarce resource, not CPU, and an
encoder starved of input needs a full renegotiation to come back. Resuming
re-opens the valve and forces a keyframe, since everything the receiver buffered
before the gate closed is undecodable.

### Known limits

- **Bitrate is the only knob.** Per the degradation note below, framerate should
  drop before resolution for defect inspection. Framerate changes need caps
  renegotiation on a live pipeline — not wired up.
- **The allocator trusts its `--total`.** It does not *measure* available
  bandwidth; it divides the number you give it. Discovering the real link
  capacity is a separate problem, and on a LAN there is nothing to discover.
- **No feedback loop yet.** Cameras report measured throughput, but the
  allocator does not yet act on it — a camera persistently undershooting its
  grant could have the surplus reclaimed automatically.

## Transport status

| Transport | State |
|---|---|
| `udp` (RTP/H.264) | **verified working end to end** — default |
| `srt` | **not working yet** — code is in place, see below |

Verified UDP round trip: 9.87 s of 1280×720 H.264 Constrained Baseline received
and decoded, recorded to a playable `.mkv`.

SRT is the better eventual choice — it carries RTT, loss and send-buffer depth,
which is the controller input the adaptive loop wants, for free. It does not yet
connect on this GStreamer 1.20 build. One real cause is already fixed in the
code: **`mode` is a property, not a URI query parameter.** The 1.20 `srtsink`
documents its URI as plain `srt://address:port` and silently ignores
`?mode=caller`, leaving both ends as callers so the link never forms. With that
corrected the listener still exits without binding, which needs more
investigation — hence UDP as the default for now.

## Notes for when you wire up the controller

- **Control on send-buffer depth, not packet loss.** Buffer growth means the
  encoder is outproducing what the link can drain, and it appears *before* loss
  does. Loss is a lagging indicator — by the time you see it, frames are gone.
  This is why SRT is worth finishing.
- **`key-int-max` and `config-interval=-1` are already set.** Without them a
  receiver joining mid-stream — or reconnecting after a drop — sits on a black
  frame until the next random keyframe.
- **Degrade framerate before resolution.** For defect inspection, spatial detail
  is the whole point and motion smoothness is nearly irrelevant, so 30→10→5 fps
  costs little diagnostically while dropping resolution destroys the ability to
  resolve a hairline crack. A naive bitrate-only controller does the opposite:
  it lets the encoder blur spatial detail to preserve temporal smoothness. Worth
  encoding as a staged policy rather than a single bitrate scalar.
- **`videotestsrc pattern=ball` compresses to ~200 kbps** regardless of the
  bitrate you ask for — the scene is nearly static. Don't read that as the
  bitrate setting being broken; use `pattern=snow` or the real camera to see the
  encoder actually work.

## What already exists in the codebase

Do not rebuild these:

| Capability | Where | State |
|---|---|---|
| GStreamer CSI capture → BGR frames | `WAFT/WAFT/camera_sources.py` | done |
| Pre-event ring buffer (the "delayed" part) | `camera_event_detector.py` (`saved_history` deque, `--event-pre-frames`) | done |
| Per-frame JPEG encode | `camera_event_detector.py` (`cv2.imencode`) | done |
| Offline clip encode to file | `waft_overlay_video.py`, `video_farneback_demo.py` (`cv2.VideoWriter`) | done |
| **Network streaming encode/transport** | this package | new |
| **Server-side decode** | this package | new |

The pre-roll buffer is worth emphasising: the "delayed stream" requirement is
**already solved** in `camera_event_detector.py`, which keeps a deque of frames
from before each trigger. What is missing is the transport, not the buffering.
