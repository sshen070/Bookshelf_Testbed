"""Read WAFT event bursts. Standalone -- no vit_classifier import.

This duplicates a little of vit_classifier/waft_events.py on purpose: that
package is a peer, not a dependency, and a detector that cannot run without
the classifier installed defeats the point of evaluating whether it could
replace it.

Two gotchas are handled here, both inherited from how the detector writes
events:

  * `event.json` frame paths are relative to the WAFT repo root and break the
    moment an event folder is copied, so frames are resolved by BASENAME
    against <event_dir>/frames/.
  * `frames_annotated/` is never read -- those PNGs have boxes painted on
    them, and feeding a detector its own previous output is a good way to
    detect rectangles.

Imports cleanly without torch or ultralytics.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

PHASE_ORDER = {"pre": 0, "trigger": 1, "post": 2}


@dataclass
class EventFrame:
    label: str
    path: Path
    frame_index: int

    @property
    def phase(self) -> str:
        return self.label.split("_")[0]


@dataclass
class WaftEvent:
    event_dir: Path
    event_name: str
    status: str
    frames: list[EventFrame] = field(default_factory=list)
    boxes_xyxy: list[list[int]] = field(default_factory=list)
    processing_size: tuple[int, int] | None = None

    @property
    def ready(self) -> bool:
        """The detector sets this only once the last post-frame is on disk."""
        return self.status == "ready_for_waft"

    def frame(self, label_or_phase: str) -> EventFrame | None:
        """Exact label first ('post_02'), then first frame of a phase ('post')."""
        for f in self.frames:
            if f.label == label_or_phase:
                return f
        matches = [f for f in self.frames if f.phase == label_or_phase]
        return matches[0] if matches else None


def load_event(event_dir: str | Path) -> WaftEvent:
    event_dir = Path(event_dir)
    manifest_path = event_dir / "event.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No event.json in {event_dir}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    frames_dir = event_dir / "frames"
    frames = []
    for entry in manifest.get("frames", []):
        recorded = entry.get("event_path") or ""
        resolved = frames_dir / Path(recorded).name
        if resolved.is_file():
            frames.append(EventFrame(
                label=entry.get("label", "?"),
                path=resolved,
                frame_index=int(entry.get("frame_index", -1)),
            ))
    frames.sort(key=lambda f: (PHASE_ORDER.get(f.phase, 9), f.frame_index))

    boxes_cfg = manifest.get("boxes", {}) or {}
    size = boxes_cfg.get("processing_size")
    return WaftEvent(
        event_dir=event_dir,
        event_name=manifest.get("event_name", event_dir.name),
        status=manifest.get("status", ""),
        frames=frames,
        boxes_xyxy=[list(map(int, b)) for b in boxes_cfg.get("boxes_xyxy", [])],
        processing_size=tuple(size) if size else None,
    )


def find_events(events_dir: str | Path) -> list[Path]:
    events_dir = Path(events_dir)
    if not events_dir.is_dir():
        return []
    return sorted(p for p in events_dir.iterdir() if (p / "event.json").is_file())


def iou(a: list[int], b: list[int]) -> float:
    """IoU of two xyxy boxes. Used to ask whether two systems mean the same region."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def scale_box(box: list[int], from_size, to_size) -> list[int]:
    """WAFT reports boxes at the FLOW resolution, not the saved frame's.

    vit_classifier's README calls this the scaling trap: cropping a 320x240
    box out of a 960x540 image silently addresses the wrong third of the
    frame. The same applies to comparing boxes across the two systems.
    """
    fw, fh = from_size
    tw, th = to_size
    sx, sy = tw / fw, th / fh
    x1, y1, x2, y2 = box
    return [int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)]
