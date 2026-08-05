"""Read WAFT's on-disk output formats.

Two producers, both handled here so the classifier never has to know WAFT's
directory conventions:

1. `camera_event_detector.py` writes an *event burst* per trigger:
       <event_dir>/event.json          metadata + boxes + frame manifest
       <event_dir>/frames/*.png        the unannotated burst (pre/trigger/post)
       <event_dir>/frames_annotated/   the same frames with boxes drawn on
       <event_dir>/boxes_trigger.json  trigger-frame boxes as dicts

2. `flow_boxes_demo.py` writes a *pair directory* per consecutive frame pair:
       <pair_dir>/boxes.json  boxes in flow resolution, plus flow stats

Two things worth knowing about the event format:

- `event.json` frame paths are recorded relative to the WAFT repo root, not to
  the event directory, so they break the moment an event folder is copied
  anywhere. Frames are resolved here by basename against `<event_dir>/frames/`
  instead, which survives relocation.
- Never read `frames_annotated/`. Those PNGs have rectangles painted on them;
  feeding one to the classifier means classifying a drawn-on box edge. WAFT
  keeps `frames/` unannotated for exactly this reason (its optical flow has the
  same problem) and this module only ever resolves into `frames/`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .boxes import Box, imread, parse_boxes_payload

# The burst order camera_event_detector.py writes. Sorting by this rather than
# alphabetically keeps pre_01 < trigger < post_01, which "post" < "pre" <
# "trigger" alphabetically would not.
_BURST_ORDER = {"pre": 0, "trigger": 1, "post": 2}


@dataclass
class EventFrame:
    label: str
    path: Path
    frame_index: int | None = None
    timestamp: str | None = None

    @property
    def phase(self) -> str:
        """'pre', 'trigger' or 'post' -- the label without its ordinal."""
        return self.label.split("_")[0]

    def load(self) -> np.ndarray | None:
        return imread(self.path)


@dataclass
class WaftEvent:
    event_dir: Path
    boxes: list[Box]
    processing_size: tuple[int, int] | None
    frames: list[EventFrame]
    event_id: int | None = None
    event_name: str | None = None
    trigger_mode: str | None = None

    def frames_in_phase(self, phase: str) -> list[EventFrame]:
        return [f for f in self.frames if f.phase == phase]

    def pair_for_transition(self) -> tuple[EventFrame, EventFrame] | None:
        """The most informative (before, after) frame pair in the burst.

        Earliest pre-frame vs. latest post-frame: the widest interval in the
        burst, so a slow placement or removal has actually finished by the
        'after' side. Falls back to first-vs-last when the burst has no
        pre/post split (e.g. --event-pre-frames 0), and returns None when
        there is only one frame and no transition can be formed.
        """
        pre = self.frames_in_phase("pre")
        post = self.frames_in_phase("post")
        if pre and post:
            return pre[0], post[-1]
        if len(self.frames) >= 2:
            return self.frames[0], self.frames[-1]
        return None


def _frame_sort_key(frame: EventFrame) -> tuple[int, int, str]:
    return (_BURST_ORDER.get(frame.phase, 99), frame.frame_index or 0, frame.label)


def load_event(event_dir: str | Path) -> WaftEvent:
    """Load a `camera_event_detector.py` event burst."""
    event_dir = Path(event_dir)
    manifest_path = event_dir / "event.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{event_dir} has no event.json -- expected a directory written by "
            "camera_event_detector.py, e.g. outputs/camera_events/events/<event_name>/"
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    boxes, processing_size = [], None
    if isinstance(manifest.get("boxes"), dict):
        boxes, processing_size = parse_boxes_payload(manifest["boxes"])
    if not boxes:
        # Grid-triggered events (no --boxes) carry no boxes in event.json; the
        # trigger-frame sidecar is the only other place they can appear.
        sidecar = event_dir / "boxes_trigger.json"
        if sidecar.exists():
            boxes, sidecar_size = parse_boxes_payload(json.load(open(sidecar, encoding="utf-8")))
            processing_size = processing_size or sidecar_size

    frames_dir = event_dir / "frames"
    frames = []
    for entry in manifest.get("frames", []):
        recorded = entry.get("event_path")
        if not recorded:
            continue
        # Resolve by basename -- the recorded path is relative to the WAFT repo
        # root and breaks if the event folder was copied elsewhere.
        path = frames_dir / Path(recorded).name
        if not path.exists():
            continue
        frames.append(
            EventFrame(
                label=entry.get("label", path.stem),
                path=path,
                frame_index=entry.get("frame_index"),
                timestamp=entry.get("timestamp"),
            )
        )

    if not frames and frames_dir.is_dir():
        # Manifest missing or unusable -- fall back to whatever is on disk.
        frames = [EventFrame(label=p.stem.split("_")[0], path=p) for p in sorted(frames_dir.glob("*.png"))]

    frames.sort(key=_frame_sort_key)
    return WaftEvent(
        event_dir=event_dir,
        boxes=boxes,
        processing_size=processing_size,
        frames=frames,
        event_id=manifest.get("event_id"),
        event_name=manifest.get("event_name"),
        trigger_mode=manifest.get("trigger_mode"),
    )


def find_events(events_dir: str | Path) -> list[Path]:
    """Every event directory under an `outputs/camera_events/events/` root."""
    events_dir = Path(events_dir)
    return sorted(p for p in events_dir.iterdir() if p.is_dir() and (p / "event.json").exists())


def load_flow_pair(pair_dir: str | Path) -> tuple[list[Box], tuple[int, int] | None]:
    """Load boxes from one `flow_boxes_demo.py` pair directory.

    Note the returned boxes are in *flow* resolution (`--width`/`--height`),
    which is usually smaller than the source frames. boxes.json does not record
    the frame size, so the caller must scale them against whatever image it is
    about to crop -- see boxes.scale_boxes.
    """
    pair_dir = Path(pair_dir)
    boxes_path = pair_dir / "boxes.json"
    if not boxes_path.exists():
        raise FileNotFoundError(f"{pair_dir} has no boxes.json (expected flow_boxes_demo.py output).")
    with open(boxes_path, "r", encoding="utf-8") as f:
        return parse_boxes_payload(json.load(f))
