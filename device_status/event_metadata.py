"""Build the detection-event document the server ranks video feeds by.

This is the join between the two producers, and it is deliberately pure: it
reads their files, returns a dict, and touches no network. `event_client.py`
does the sending, `status_server.py` the receiving. Run this file directly for
a self-check against a real event directory:

    python3 event_metadata.py --event-dir <waft-event-dir>
    python3 event_metadata.py --self-test

## Where each field comes from

WAFT's `camera_event_detector.py` writes, per event:

    event.json                  trigger mode, both method scores, boxes, frames
    farneback_grid_trigger.json per-cell optical-flow magnitude (pixels/frame)
    change_grid_trigger.json    per-cell colour change (mean |RGB delta|, 0-255)
    boxes_trigger.json          trigger-frame boxes, {x1,y1,x2,y2}
    frames/                     the unannotated burst

`vit_classifier` later writes `vit_result.json` beside it, with a before/after
label per box. It is optional here on purpose -- the ViT runs seconds after the
burst lands and may not have run at all, and an event that reaches the server
late is worth less than a motion-only event that arrives now.

## Nothing is imported from WAFT or vit_classifier

Only their on-disk formats are known here. `device_status/` stays copyable to a
box that has neither, which is the same property the health half already has.
The cost is that a producer changing its JSON breaks this silently, so every
read is defensive and a missing part degrades the document instead of raising.

## The two magnitudes, and why both travel

`farneback` is optical flow -- how far pixels *moved*, in pixels/frame.
`change` is mean absolute RGB difference -- how much the picture *recoloured*,
in intensity levels 0-255. They disagree in ways that matter to a server
deciding where to spend a link:

- flow high, colour low   -> something moved across a similar background
- colour high, flow low   -> a light turned on, a shadow swept, exposure shifted
- both high               -> a real object entered or left

Sending only one leaves the server unable to tell a genuine intrusion from the
sun coming out. Both are cheap -- twelve floats.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

SCHEMA_VERSION = "1.0"

# Namespace for the deterministic event ids below. Any fixed UUID works; this
# one is arbitrary and must simply never change, or every id changes with it.
ID_NAMESPACE = uuid.UUID("6f6b7368-656c-5f74-6573-746265640001")

# WAFT stamps frame names with datetime.now() in this format -- local time, no
# offset, microseconds after the seconds field.
WAFT_TIMESTAMP_FMT = "%Y%m%d_%H%M%S_%f"

DEFAULT_THUMBNAIL_WIDTH = 320
DEFAULT_THUMBNAIL_QUALITY = 60

# vit_result.json transition words -> the event_type the server sees. These are
# the complete set `classifier.describe_transition` can return; anything else
# would be a producer change, and falls back to plain motion.
#
# Listed most significant first: with several regions disagreeing, the most
# significant transition names the event. A frame where one object was removed
# and another merely shifted is a removal.
TRANSITION_EVENT_TYPES = {
    "removed": "object_removed",
    "placed": "object_placed",
    "substituted": "object_substituted",
    "moved_or_same_class": "object_moved",
    "uncertain": "uncertain",
}

# What an unclassified event is called. Distinct from "uncertain", which means
# the classifier *did* run and could not name the transition -- the server
# grants on classified events only, so conflating the two would make an
# un-run classifier look like a completed one.
UNCLASSIFIED_EVENT_TYPE = "motion"


# ---- small readers ------------------------------------------------------


def _read_json(path: Path, default=None):
    """Read JSON, or return `default` if it is missing or unreadable.

    Both producers rewrite their manifests in place with a truncating write, so
    a read can land mid-write. That is "not ready", never "broken" -- the
    watcher will be back in a second.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return default


def parse_waft_time(value) -> float | None:
    """A WAFT timestamp -> epoch seconds, or None.

    Accepts both forms the detector writes: `20260723_160545_043662` on frames
    and `2026-07-23T16:05:45` in `created_at`. Both are naive local time --
    WAFT calls datetime.now() -- so they are interpreted as local, which is
    what `.timestamp()` does with a naive datetime. Producing the epoch here
    rather than shipping the string means the server never has to guess the
    device's timezone.
    """
    if not isinstance(value, str) or not value:
        return None
    for parse in (
        lambda v: datetime.strptime(v, WAFT_TIMESTAMP_FMT),
        lambda v: datetime.fromisoformat(v.replace("Z", "+00:00")),
    ):
        try:
            return parse(value).timestamp()
        except ValueError:
            continue
    return None


def to_xywh(box) -> list[int] | None:
    """`{x1,y1,x2,y2}` or `[x1,y1,x2,y2]` -> `[x, y, w, h]`.

    The sample schema is xywh; WAFT works in xyxy throughout. Converting once
    here beats every consumer guessing which convention a `bbox` follows.
    """
    if isinstance(box, dict):
        try:
            x1, y1, x2, y2 = (box["x1"], box["y1"], box["x2"], box["y2"])
        except KeyError:
            return None
    elif isinstance(box, (list, tuple)) and len(box) == 4:
        x1, y1, x2, y2 = box
    else:
        return None
    try:
        return [int(x1), int(y1), int(x2) - int(x1), int(y2) - int(y1)]
    except (TypeError, ValueError):
        return None


def _round(value, digits=3):
    return round(float(value), digits) if isinstance(value, (int, float)) else None


# ---- motion block -------------------------------------------------------


def grid_digest(cells) -> dict | None:
    """Reduce a 3x3 magnitude grid to the four numbers worth transmitting.

    The full grid is nine cells x five fields; a server ranking cameras needs
    the peak and the spread, not the map. `peak_topk` is the dominant cell's
    top-k mean, which is what the detector itself thresholds against, and
    `frame_mean` is the average across all cells -- their ratio is how
    localised the activity is. A whole frame brightening scores high on both;
    one object moving scores high only on the peak.
    """
    if not isinstance(cells, list) or not cells:
        return None
    scored = [c for c in cells if isinstance(c, dict) and isinstance(c.get("topk_score"), (int, float))]
    if not scored:
        return None
    dominant = max(scored, key=lambda c: c["topk_score"])
    return {
        "dominant_cell": dominant.get("label"),
        "peak_topk": _round(dominant.get("topk_score")),
        "peak_max": _round(dominant.get("max_score")),
        "frame_mean": _round(sum(c.get("mean_score", 0.0) for c in scored) / len(scored)),
    }


def motion_block(event: dict, farneback_cells, change_cells) -> dict:
    """Both detectors' magnitudes, plus why the event actually fired.

    `window` is included because the trigger-frame scores frequently do *not*
    explain the trigger on their own: the detector fires on N hits within a
    sliding window, so a real event can carry a dominant score below its own
    threshold. A server that saw only the scores would read those as noise.
    """
    farneback = event.get("farneback") if isinstance(event.get("farneback"), dict) else {}
    change = event.get("change") if isinstance(event.get("change"), dict) else {}
    window = event.get("window") if isinstance(event.get("window"), dict) else {}
    boxes = event.get("boxes") if isinstance(event.get("boxes"), dict) else {}
    flow_reference, colour_reference = trigger_reference(event)

    return {
        # Optical flow: displacement in pixels per frame.
        "flow": {
            "units": "px/frame",
            "threshold": _round(farneback.get("threshold")),
            # What this magnitude should be judged against -- not always the
            # grid threshold above. See `trigger_reference`.
            "reference": _round(flow_reference),
            "dominant_cell": farneback.get("dominant_cell"),
            "dominant_score": _round(farneback.get("dominant_score")),
            "selected_cells": farneback.get("selected_cells") or [],
            "grid": grid_digest(farneback_cells),
        },
        # Colour shift: mean |RGB delta| across channels, 0-255 intensity levels.
        "colour": {
            "units": "mean_abs_rgb_delta_0_255",
            "threshold": _round(change.get("threshold")),
            "reference": _round(colour_reference),
            "dominant_cell": change.get("dominant_cell"),
            "dominant_score": _round(change.get("dominant_score")),
            "selected_cells": change.get("selected_cells") or [],
            "grid": grid_digest(change_cells),
        },
        "trigger_mode": event.get("trigger_mode"),
        "box_trigger": bool(boxes.get("enabled")) and not farneback.get("selected_cells"),
        "magnitude_threshold": _round(boxes.get("magnitude_threshold")),
        "num_boxes": boxes.get("num_boxes"),
        "trigger_cells": event.get("trigger_cells") or [],
        "window": {
            "size": window.get("window_size"),
            "hit_frames": window.get("hit_frames"),
            "global_hit": window.get("global_hit"),
        },
    }


def soft_normalise(value, reference) -> float:
    """`value / (value + reference)` -- 0 at rest, 0.5 at the reference, →1 beyond.

    Replaces `min(1, value/threshold)`, which was measured on this rig to be
    unusable: against WAFT's own thresholds the flow ratio saturated at 1.0 on
    12% of *all* frames including idle ones, leaving no dynamic range in the
    region that decides anything, while the colour ratio sat below 0.5 on 92%
    of frames. `max()` over the two then picked flow on 100% of frames, so the
    colour channel contributed nothing at all.

    This curve never reaches its cap, so it stays strictly ordered however
    large the input, and it degrades gracefully when the reference is a poor
    one -- which matters here, because these thresholds are WAFT's defaults
    rather than anything calibrated against this scene.
    """
    if not isinstance(value, (int, float)) or not isinstance(reference, (int, float)):
        return 0.0
    if reference <= 0 or value <= 0:
        return 0.0
    return value / (value + reference)


def motion_strength(flow_score, flow_reference, colour_score, colour_reference) -> float:
    """Both magnitudes on one [0, 1) scale.

    A **mean**, not a max. The two measure different things -- displacement
    against recolouring -- and taking the larger meant that whichever channel
    happened to sit closer to its own threshold silently became the only one
    that counted. Measured on this rig, that was always flow.

    Weighted evenly because there is no evidence here to justify preferring
    one, not because equality is known to be correct. Both are in the same
    place so re-weighting is one edit once real data exists.
    """
    return round(
        0.5 * soft_normalise(flow_score, flow_reference)
        + 0.5 * soft_normalise(colour_score, colour_reference),
        4,
    )


def trigger_reference(event: dict) -> tuple[float | None, float | None]:
    """The two references an event's magnitudes should be judged against.

    Which threshold actually fired the event depends on the mode, and getting
    this wrong was a real defect: under `--box-trigger` a frame is "hot" when
    it holds at least one qualifying *box*, so the trigger test is
    `magnitude_threshold` against the flow map and the 3x3 grid thresholds are
    never consulted. Events routinely carry `selected_cells: []` for both
    methods for exactly that reason -- dividing by a grid threshold there
    measures distance to a bar nothing was jumping.
    """
    boxes = event.get("boxes") if isinstance(event.get("boxes"), dict) else {}
    farneback = event.get("farneback") if isinstance(event.get("farneback"), dict) else {}
    change = event.get("change") if isinstance(event.get("change"), dict) else {}

    flow_reference = farneback.get("threshold")
    # box-trigger leaves both grids unselected; the box threshold is the one
    # the frame was actually judged on.
    if not farneback.get("selected_cells") and isinstance(boxes.get("magnitude_threshold"), (int, float)):
        flow_reference = boxes["magnitude_threshold"]

    return flow_reference, change.get("threshold")


def motion_confidence(event: dict) -> float:
    """A frame-level motion score in [0, 1) -- NOT a per-box confidence.

    WAFT's boxes are connected components of the flow-magnitude mask and carry
    no score of their own (`box_extraction.magnitude_to_boxes` returns
    coordinates only). So when no classifier has run there is nothing per-box
    to report, and the honest move is one frame-level number repeated across
    the boxes rather than a fabricated per-box one.

    Note the server does **not** grant a video feed on this number alone -- an
    unclassified event is never grant-eligible, because motion cannot tell a
    bird from a person. It orders unclassified events against each other and
    nothing more.
    """
    farneback = event.get("farneback") if isinstance(event.get("farneback"), dict) else {}
    change = event.get("change") if isinstance(event.get("change"), dict) else {}
    flow_reference, colour_reference = trigger_reference(event)
    return motion_strength(
        farneback.get("dominant_score"), flow_reference,
        change.get("dominant_score"), colour_reference,
    )


# ---- detections ---------------------------------------------------------


def build_detections(boxes_xyxy, vit_regions, fallback_confidence: float) -> list[dict]:
    """The flat `detections[]` of the schema: one class + confidence per box.

    Kept deliberately generic -- `{class, confidence, bbox}` and nothing else --
    so the server can rank a camera without knowing anything about optical flow
    or exemplar banks. The project-specific detail lives alongside in `motion`
    and `regions`, which a generic consumer ignores.

    The ViT's `score` is cosine similarity against an exemplar bank, not a
    softmax probability. It shares a range with a probability and orders the
    same way, which is why it can sit in `confidence`, but a threshold tuned on
    a detector's confidence will not transfer -- the bank's own unknown cutoff
    is 0.35 (`vit_classifier/config.yaml`). `regions[].after.score` carries the
    same number unrounded for anyone who needs to be exact about it.
    """
    regions = vit_regions if isinstance(vit_regions, list) else []
    detections = []

    for index, raw_box in enumerate(boxes_xyxy or []):
        bbox = to_xywh(raw_box)
        if bbox is None:
            continue

        # Positional pairing: vit_result.json lists its regions in the same
        # order as the boxes it was handed. It repeats box_xyxy per region, so
        # prefer matching on that and fall back to position -- a mismatch would
        # otherwise attach a label to the wrong object, which is worse than
        # reporting plain motion.
        region = None
        for candidate in regions:
            if isinstance(candidate, dict) and to_xywh(candidate.get("box_xyxy")) == bbox:
                region = candidate
                break
        if region is None and index < len(regions) and isinstance(regions[index], dict):
            if "box_xyxy" not in regions[index]:
                region = regions[index]

        label, confidence = "motion", fallback_confidence
        if region is not None:
            after = region.get("after") if isinstance(region.get("after"), dict) else region
            if isinstance(after, dict) and isinstance(after.get("label"), str):
                label = after["label"]
                score = after.get("score")
                # Cosine similarity is [-1, 1]; a negative one means "less than
                # nothing in common", so clamping at 0 loses no information a
                # consumer could act on.
                if isinstance(score, (int, float)):
                    confidence = round(max(0.0, min(1.0, float(score))), 3)

        detections.append({"class": label, "confidence": confidence, "bbox": bbox})

    return detections


def stable_id(node_id: str, event_name: str, kind: str = "event") -> str:
    """A UUID that is the same every time this event is described.

    Two independent reasons it must not be random:

    - WAFT's own `event_id` is a per-run counter that restarts at 1 with the
      detector, so two runs on one node emit colliding ids and a server keyed
      on them would treat the second run's first event as a duplicate of the
      first run's. Scoping by node and event name fixes that without
      coordination between nodes.
    - An event is published twice -- once on motion, again once the classifier
      has labelled it (see `revision`). Both sends must carry the same id or
      the update reads as a second detection and the camera looks twice as
      active as it is.

    Deterministic also means a restarted publisher re-sending an event it
    already sent is recognised as the same one rather than duplicated.
    """
    return str(uuid.uuid5(ID_NAMESPACE, f"{kind}/{node_id}/{event_name}"))


def derive_event_type(vit_regions) -> str:
    """The event word the server routes on.

    `motion` when no classifier has run -- which is every revision-1 event,
    since the motion pipeline can only report that something moved, never what.
    Once `vit_result.json` exists this becomes the most significant transition
    across all regions; the full set stays in `regions[]`.
    """
    if not isinstance(vit_regions, list):
        return UNCLASSIFIED_EVENT_TYPE

    found = {
        region.get("event") for region in vit_regions if isinstance(region, dict)
    }
    # Dict order is the significance order, so the first match is the winner.
    for transition, event_type in TRANSITION_EVENT_TYPES.items():
        if transition in found:
            return event_type

    # Regions exist but name no transition this version knows. The classifier
    # ran, so this is not the same as unclassified.
    return "uncertain" if vit_regions else UNCLASSIFIED_EVENT_TYPE


# ---- thumbnail ----------------------------------------------------------


def encode_thumbnail(
    path: Path,
    width: int = DEFAULT_THUMBNAIL_WIDTH,
    quality: int = DEFAULT_THUMBNAIL_QUALITY,
) -> str | None:
    """A small JPEG of one frame, base64'd, or None if it cannot be made.

    Optional by construction. WAFT's frames are PNG and re-encoding needs an
    image library, but `device_status/` deliberately depends on nothing but
    paho-mqtt so either half can be copied to a bare box. cv2 then Pillow are
    tried and absence is not an error -- the event still carries every number,
    which is what the ranking decision actually runs on. A Jetson running WAFT
    has cv2 regardless.

    Width matters more than quality for size: this is a thumbnail for a human
    to confirm the event was real, not the video feed being requested.
    """
    if not path or not Path(path).exists():
        return None

    try:
        import cv2

        image = cv2.imread(str(path))
        if image is None:
            return None
        height = max(1, int(image.shape[0] * width / image.shape[1]))
        small = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        ok, buffer = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return base64.b64encode(buffer.tobytes()).decode() if ok else None
    except ImportError:
        pass

    try:
        import io

        from PIL import Image

        with Image.open(path) as image:
            image = image.convert("RGB")
            height = max(1, int(image.height * width / image.width))
            image = image.resize((width, height), Image.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality)
        return base64.b64encode(buffer.getvalue()).decode()
    except (ImportError, OSError):
        return None


def pick_thumbnail_frame(event: dict, event_dir: Path) -> Path | None:
    """The trigger frame, which is the one that caused the event.

    Frame paths in event.json are recorded relative to the WAFT repo root, so
    they break as soon as an event folder moves. Resolved by basename against
    `<event_dir>/frames/` instead, which survives relocation.

    Never `frames_annotated/` -- those have boxes painted on, and a thumbnail
    is meant to show an operator the scene, not WAFT's opinion of it.
    """
    frames_dir = Path(event_dir) / "frames"
    entries = event.get("frames") if isinstance(event.get("frames"), list) else []

    candidates = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        recorded = entry.get("event_path") or entry.get("source_path")
        if not recorded:
            continue
        resolved = frames_dir / Path(recorded).name
        if resolved.exists():
            candidates.append((entry.get("label", ""), resolved))

    for label, resolved in candidates:
        if label.startswith("trigger"):
            return resolved
    if candidates:
        return candidates[0][1]

    on_disk = sorted(frames_dir.glob("*.png")) if frames_dir.is_dir() else []
    return on_disk[0] if on_disk else None


# ---- device health ------------------------------------------------------


def health_digest(status: dict | None) -> dict:
    """The health subset small enough to ride on every event.

    A full heartbeat is ~1.7 KB and already goes to the same server on its own
    topic; repeating it per event would be pure duplication. What stays is what
    changes an *allocation* decision rather than a maintenance one: a node at
    90C is about to throttle and will not deliver the feed you grant it, and a
    node with no disk cannot buffer a pre-roll. Names mirror the sample schema.
    """
    if not isinstance(status, dict):
        return {}

    digest = {}
    temperatures = status.get("temperatures_c")
    if isinstance(temperatures, dict) and temperatures:
        readings = [v for v in temperatures.values() if isinstance(v, (int, float))]
        if readings:
            # The hottest zone, not a named one: which zone leads varies by
            # workload, and an alerting rule wants the worst case.
            digest["soc_temp_c"] = _round(max(readings), 1)

    disk = status.get("disk")
    if isinstance(disk, dict) and isinstance(disk.get("used_pct"), (int, float)):
        digest["disk_free_pct"] = _round(100.0 - disk["used_pct"], 1)

    for source, key, field in (
        ("cpu", "used_pct", "cpu_pct"),
        ("gpu", "load_pct", "gpu_pct"),
        ("memory", "used_pct", "mem_used_pct"),
    ):
        block = status.get(source)
        if isinstance(block, dict) and isinstance(block.get(key), (int, float)):
            digest[field] = _round(block[key], 1)

    power = status.get("power")
    if isinstance(power, dict) and isinstance(power.get("total_watts"), (int, float)):
        digest["power_w"] = _round(power["total_watts"], 2)

    return digest


# ---- the document -------------------------------------------------------


def build_document(
    event_dir: str | Path,
    node_id: str,
    site_id: str,
    camera_id: str | None = None,
    device_status: dict | None = None,
    model_version: str | None = None,
    requested_seconds: float = 30.0,
    preroll_seconds: float = 5.0,
    thumbnail_width: int = DEFAULT_THUMBNAIL_WIDTH,
    thumbnail_quality: int = DEFAULT_THUMBNAIL_QUALITY,
    include_thumbnail: bool = True,
) -> dict | None:
    """Join one WAFT event directory into the wire document.

    Returns None only when there is no readable `event.json` -- everything else
    degrades. An event with no boxes, no classifier result and no thumbnail is
    still worth sending: it says this camera saw something, which is the
    question the server is asking.
    """
    event_dir = Path(event_dir)
    event = _read_json(event_dir / "event.json")
    if not isinstance(event, dict):
        return None

    farneback_cells = _read_json(event_dir / "farneback_grid_trigger.json", [])
    change_cells = _read_json(event_dir / "change_grid_trigger.json", [])

    boxes = event.get("boxes") if isinstance(event.get("boxes"), dict) else {}
    boxes_xyxy = boxes.get("boxes_xyxy")
    if not boxes_xyxy:
        # Grid-triggered events carry no boxes in event.json; the sidecar is
        # the only other place they appear.
        boxes_xyxy = _read_json(event_dir / "boxes_trigger.json", []) or []

    vit = _read_json(event_dir / "vit_result.json")
    vit_regions = vit.get("regions") if isinstance(vit, dict) else None

    timestamp = (
        parse_waft_time(event.get("trigger_timestamp"))
        or parse_waft_time(event.get("created_at"))
        or Path(event_dir / "event.json").stat().st_mtime
    )

    thumbnail = None
    if include_thumbnail:
        frame = pick_thumbnail_frame(event, event_dir)
        if frame is not None:
            thumbnail = encode_thumbnail(frame, thumbnail_width, thumbnail_quality)

    camera_id = camera_id or node_id
    event_name = event.get("event_name") or event_dir.name
    classified = vit_regions is not None

    document = {
        "event_id": stable_id(node_id, event_name, "event"),
        # 1 = motion only, sent the moment the burst completes. 2 = the same
        # event re-sent once the classifier has labelled it. The server keeps
        # the highest revision per event_id; see `stable_id`.
        "revision": 2 if classified else 1,
        "schema_version": SCHEMA_VERSION,
        "site_id": site_id,
        "node_id": node_id,
        "camera_id": camera_id,
        "timestamp_utc": round(timestamp, 3),
        "event_type": derive_event_type(vit_regions),
        "detections": build_detections(boxes_xyxy, vit_regions, motion_confidence(event)),
        "model_version": model_version,
        # The coordinate space every bbox in this document is expressed in.
        # WAFT detects at --width/--height, not sensor resolution, so a
        # consumer drawing these on a full-res frame without scaling gets boxes
        # in the wrong place.
        "frame_size": boxes.get("processing_size"),
        "clip": {
            "clip_id": stable_id(node_id, event_name, "clip"),
            "status": "pending",
            "transport": "rtp/h264",
            "endpoint": None,
            "duration_s": requested_seconds,
            "preroll_s": preroll_seconds,
            # How the server acts on its own decision. video_link's sender is
            # already subscribed here, so granting a feed is one publish and
            # this module needs no knowledge of that process beyond the name.
            "control_topic": f"video/{camera_id}/trigger",
        },
        "thumbnail_b64": thumbnail,
        "device_health": health_digest(device_status),
        # ---- beyond the generic schema; a plain consumer ignores these ----
        "motion": motion_block(event, farneback_cells, change_cells),
        "regions": vit_regions or [],
        "source": {
            "event_name": event_name,
            "event_dir": str(event_dir),
            "waft_status": (event.get("waft") or {}).get("status") if isinstance(event.get("waft"), dict) else None,
            "classified": classified,
        },
    }
    return document


# ---- self-check ---------------------------------------------------------


def _self_test() -> int:
    """Checks the conversions that are easy to get quietly wrong."""
    checks, failures = 0, 0

    def check(name, got, want):
        nonlocal checks, failures
        checks += 1
        if got != want:
            failures += 1
            print(f"  FAIL {name}: got {got!r}, want {want!r}")

    check("xyxy dict -> xywh", to_xywh({"x1": 938, "y1": 381, "x2": 960, "y2": 440}), [938, 381, 22, 59])
    check("xyxy list -> xywh", to_xywh([10, 20, 110, 140]), [10, 20, 100, 120])
    check("bad box -> None", to_xywh({"x1": 1}), None)
    check("bad box type -> None", to_xywh("nope"), None)

    stamp = parse_waft_time("20260723_160545_043662")
    check("waft stamp parses", isinstance(stamp, float), True)
    check("iso stamp parses", isinstance(parse_waft_time("2026-07-23T16:05:45"), float), True)
    check("junk stamp -> None", parse_waft_time("not-a-time"), None)
    check("agree within a second", abs(stamp - parse_waft_time("2026-07-23T16:05:45")) < 1.0, True)

    check("soft: at the reference -> 0.5", soft_normalise(5.0, 5.0), 0.5)
    check("soft: half the reference", round(soft_normalise(2.5, 5.0), 4), 0.3333)
    check("soft: 10x the reference still under 1", soft_normalise(50.0, 5.0) < 1.0, True)
    # The old min(1, x/t) saturated on 12% of all frames including idle ones,
    # so two very different events scored identically. This must not.
    check("soft: never saturates, stays ordered",
          soft_normalise(50.0, 5.0) < soft_normalise(500.0, 5.0), True)
    check("soft: zero reference is not a divide", soft_normalise(1.0, 0), 0.0)
    check("soft: zero value", soft_normalise(0, 5.0), 0.0)

    # Both channels must move the result. Under the old max() the colour term
    # was picked on 0% of measured frames, so it contributed nothing at all.
    quiet_colour = motion_strength(2.791, 5.0, 0.0, 100.0)
    loud_colour = motion_strength(2.791, 5.0, 90.0, 100.0)
    check("colour actually changes the score", loud_colour > quiet_colour, True)
    check("flow actually changes the score",
          motion_strength(40.0, 5.0, 15.0, 100.0) > motion_strength(2.0, 5.0, 15.0, 100.0), True)
    check("no motion at all", motion_strength(0, 5.0, 0, 100.0), 0.0)

    # The reference must be the threshold that actually fired the event. Under
    # --box-trigger neither grid threshold is consulted -- selected_cells is
    # empty on both -- and the box magnitude_threshold is the real bar.
    box_event = {
        "farneback": {"threshold": 5.0, "selected_cells": []},
        "change": {"threshold": 100.0},
        "boxes": {"magnitude_threshold": 3.0},
    }
    check("box-trigger uses the box threshold", trigger_reference(box_event)[0], 3.0)
    grid_event = {
        "farneback": {"threshold": 5.0, "selected_cells": ["r1c1"]},
        "change": {"threshold": 100.0},
        "boxes": {"magnitude_threshold": 3.0},
    }
    check("grid trigger uses the grid threshold", trigger_reference(grid_event)[0], 5.0)
    check("colour reference is the change threshold", trigger_reference(box_event)[1], 100.0)
    check("confidence with no methods", motion_confidence({}), 0.0)

    cells = [
        {"label": "r1c1", "mean_score": 3.0, "topk_score": 9.9, "max_score": 33.6},
        {"label": "r1c2", "mean_score": 1.8, "topk_score": 6.6, "max_score": 27.0},
    ]
    check("grid picks the peak cell", grid_digest(cells)["dominant_cell"], "r1c1")
    check("grid means across cells", grid_digest(cells)["frame_mean"], 2.4)
    check("empty grid -> None", grid_digest([]), None)
    check("garbage grid -> None", grid_digest([{"label": "x"}]), None)

    detections = build_detections(
        [{"x1": 0, "y1": 0, "x2": 10, "y2": 10}],
        [{"box_xyxy": [0, 0, 10, 10], "after": {"label": "books", "score": 0.87}}],
        0.42,
    )
    check("vit label wins", detections[0]["class"], "books")
    check("vit score becomes confidence", detections[0]["confidence"], 0.87)

    unmatched = build_detections([{"x1": 0, "y1": 0, "x2": 10, "y2": 10}], [], 0.42)
    check("no vit -> motion", unmatched[0]["class"], "motion")
    check("no vit -> frame-level confidence", unmatched[0]["confidence"], 0.42)

    # A region whose box does not match must not be attached to this box.
    mismatched = build_detections(
        [{"x1": 0, "y1": 0, "x2": 10, "y2": 10}],
        [{"box_xyxy": [500, 500, 600, 600], "after": {"label": "plushie", "score": 0.9}}],
        0.42,
    )
    check("mismatched region is not borrowed", mismatched[0]["class"], "motion")

    negative = build_detections(
        [{"x1": 0, "y1": 0, "x2": 10, "y2": 10}],
        [{"box_xyxy": [0, 0, 10, 10], "after": {"label": "unknown", "score": -0.2}}],
        0.42,
    )
    check("negative cosine clamps to 0", negative[0]["confidence"], 0.0)

    check("id is stable across calls", stable_id("n1", "ev1"), stable_id("n1", "ev1"))
    check("id differs per node", stable_id("n1", "ev1") != stable_id("n2", "ev1"), True)
    check("id differs per event", stable_id("n1", "ev1") != stable_id("n1", "ev2"), True)
    check("clip id differs from event id", stable_id("n1", "ev1", "clip") != stable_id("n1", "ev1", "event"), True)

    check("placed promotes event_type", derive_event_type([{"event": "placed"}]), "object_placed")
    check("removed promotes event_type", derive_event_type([{"event": "removed"}]), "object_removed")
    # These two are real describe_transition outputs that used to fall through
    # to "motion", making a classified event indistinguishable from an
    # unclassified one -- which now decides whether a feed is granted at all.
    check("substituted is handled", derive_event_type([{"event": "substituted"}]), "object_substituted")
    check("moved is handled", derive_event_type([{"event": "moved_or_same_class"}]), "object_moved")
    check("uncertain is classified, not motion", derive_event_type([{"event": "uncertain"}]), "uncertain")
    check("no regions is unclassified", derive_event_type(None), "motion")
    check("empty regions is unclassified", derive_event_type([]), "motion")
    check("most significant transition wins",
          derive_event_type([{"event": "moved_or_same_class"}, {"event": "removed"}]), "object_removed")
    check("unknown transition word is still classified",
          derive_event_type([{"event": "some_future_word"}]), "uncertain")

    health = health_digest({
        "temperatures_c": {"cpu-thermal": 41.9, "tj-thermal": 61.2},
        "disk": {"used_pct": 39.7},
        "cpu": {"used_pct": 19.0},
        "memory": {"used_pct": 42.0},
    })
    check("health takes the hottest zone", health["soc_temp_c"], 61.2)
    check("health inverts disk used", health["disk_free_pct"], 60.3)
    check("health tolerates nothing", health_digest(None), {})

    print(f"\n{checks - failures}/{checks} checks passed.")
    return 1 if failures else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Build the detection-event document from a WAFT event directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--event-dir", help="a WAFT event directory (containing event.json)")
    p.add_argument("--node-id", default="node-01")
    p.add_argument("--site-id", default="site-01")
    p.add_argument("--camera-id", default=None)
    p.add_argument("--no-thumbnail", action="store_true", help="skip the JPEG, print the rest")
    p.add_argument("--self-test", action="store_true", help="run the built-in checks and exit")
    args = p.parse_args(argv)

    if args.self_test:
        return _self_test()
    if not args.event_dir:
        p.error("need --event-dir or --self-test")

    document = build_document(
        args.event_dir,
        node_id=args.node_id,
        site_id=args.site_id,
        camera_id=args.camera_id,
        include_thumbnail=not args.no_thumbnail,
    )
    if document is None:
        print(f"{args.event_dir} has no readable event.json", file=sys.stderr)
        return 1

    # The thumbnail would drown the rest on a terminal; report its size instead.
    printable = dict(document)
    if printable.get("thumbnail_b64"):
        printable["thumbnail_b64"] = f"<{len(printable['thumbnail_b64'])} b64 chars>"
    print(json.dumps(printable, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())