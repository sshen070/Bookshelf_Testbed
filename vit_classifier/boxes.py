"""Boxes, coordinate scaling, and cropping -- everything before the ViT sees pixels.

Deliberately depends on cv2 + numpy only (no torch, no timm), so the parts of
this package that read WAFT output and cut crops can be imported, tested, and
scripted on a machine with no model stack installed.

THE SCALING TRAP: WAFT reports boxes in the resolution the flow was computed
at, not the resolution the frames were saved at. `flow_boxes_demo.py --width
320 --height 240` writes boxes in 320x240 while the PNGs on disk may be
960x540; `camera_event_detector.py` records the resolution it used in
`event.json` as `boxes.processing_size`. Cropping a 320x240 box out of a
960x540 image silently addresses the wrong third of the frame and the
classifier confidently labels the wrong object. Always route through
`scale_boxes` when the two differ -- `crop` cannot detect the mismatch for you.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


@dataclass(frozen=True)
class Box:
    """An axis-aligned box in pixel coordinates, xyxy, x2/y2 exclusive."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]

    def to_dict(self) -> dict[str, int]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


def scale_box(box: Box, from_size: tuple[int, int], to_size: tuple[int, int]) -> Box:
    """Rescale a box between two image resolutions. Sizes are (width, height)."""
    fw, fh = from_size
    tw, th = to_size
    if fw <= 0 or fh <= 0:
        raise ValueError(f"from_size must be positive, got {from_size}")
    if (fw, fh) == (tw, th):
        return box
    sx, sy = tw / fw, th / fh
    return Box(
        x1=int(round(box.x1 * sx)),
        y1=int(round(box.y1 * sy)),
        x2=int(round(box.x2 * sx)),
        y2=int(round(box.y2 * sy)),
    )


def scale_boxes(boxes: list[Box], from_size: tuple[int, int], to_size: tuple[int, int]) -> list[Box]:
    return [scale_box(b, from_size, to_size) for b in boxes]


def crop(
    img: np.ndarray,
    box: Box,
    pad_frac: float = 0.15,
    square: bool = True,
    min_size_px: int = 8,
) -> np.ndarray | None:
    """Cut `box` out of `img`, optionally padded and squared up.

    Squaring matters: the ViT transform resizes to a fixed square input, so a
    tall thin box (a book spine, a person's arm) would otherwise be stretched
    into a shape the pretrained backbone has never seen. Growing the short side
    to match the long one preserves the aspect ratio at the cost of a little
    extra surrounding context -- usually helpful for identification anyway.

    Returns None when the result would be smaller than `min_size_px` on a side
    or falls entirely outside the image, rather than handing the model a 2px
    smear it will confidently misclassify.
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2

    if pad_frac > 0:
        pad_x = int((x2 - x1) * pad_frac)
        pad_y = int((y2 - y1) * pad_frac)
        x1, y1, x2, y2 = x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y

    if square:
        bw, bh = x2 - x1, y2 - y1
        if bw > bh:
            grow = (bw - bh) // 2
            y1, y2 = y1 - grow, y2 + grow
        elif bh > bw:
            grow = (bh - bw) // 2
            x1, x2 = x1 - grow, x2 + grow

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < min_size_px or y2 - y1 < min_size_px:
        return None
    return img[y1:y2, x1:x2]


def imread(path: str | Path) -> np.ndarray | None:
    """Read an image as BGR uint8. Returns None if unreadable."""
    return cv2.imread(str(path))


# ---- WAFT JSON parsing --------------------------------------------------


def parse_boxes_payload(payload) -> tuple[list[Box], tuple[int, int] | None]:
    """Parse any of the box shapes WAFT emits.

    Three formats in the wild, all handled here so callers don't branch:
      1. `flow_boxes_demo.py` boxes.json  -> {"boxes_xyxy": [[x1,y1,x2,y2], ...], ...}
      2. `camera_event_detector.py` boxes_trigger.json -> [{"x1":..,"y1":..,"x2":..,"y2":..}, ...]
      3. the `boxes` block inside event.json -> as (1) plus "processing_size": [w, h]

    Returns (boxes, processing_size) where processing_size is None if the
    payload didn't record it -- in which case the caller must know the flow
    resolution some other way before cropping. See the module docstring.
    """
    processing_size = None

    if isinstance(payload, dict):
        if "boxes" in payload and isinstance(payload["boxes"], dict):
            payload = payload["boxes"]  # event.json wraps it one level down
        raw_size = payload.get("processing_size")
        if raw_size and len(raw_size) == 2:
            processing_size = (int(raw_size[0]), int(raw_size[1]))
        raw_boxes = payload.get("boxes_xyxy", [])
    elif isinstance(payload, list):
        raw_boxes = payload
    else:
        raise ValueError(f"Unrecognized box payload of type {type(payload).__name__}")

    boxes = []
    for item in raw_boxes:
        if isinstance(item, dict):
            boxes.append(Box(int(item["x1"]), int(item["y1"]), int(item["x2"]), int(item["y2"])))
        elif isinstance(item, (list, tuple)) and len(item) == 4:
            boxes.append(Box(int(item[0]), int(item[1]), int(item[2]), int(item[3])))
        else:
            raise ValueError(f"Unrecognized box entry: {item!r}")
    return boxes, processing_size


def load_boxes_json(path: str | Path) -> tuple[list[Box], tuple[int, int] | None]:
    with open(path, "r", encoding="utf-8") as f:
        return parse_boxes_payload(json.load(f))
