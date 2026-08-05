"""Box parsing, scaling, cropping, and WAFT event reading.

No torch, no timm, no weight download -- these all run on a bare machine.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vit_classifier.boxes import Box, crop, load_boxes_json, parse_boxes_payload, scale_box, scale_boxes
from vit_classifier.waft_events import find_events, load_event


# ---- Box / scaling ------------------------------------------------------


def test_box_geometry():
    b = Box(10, 20, 40, 60)
    assert (b.width, b.height, b.area) == (30, 40, 1200)
    assert b.to_list() == [10, 20, 40, 60]
    assert b.to_dict() == {"x1": 10, "y1": 20, "x2": 40, "y2": 60}


def test_scale_box_upsamples():
    """The trap this guards: flow-resolution boxes cropped from full-res frames."""
    b = Box(80, 60, 160, 120)
    scaled = scale_box(b, (320, 240), (960, 720))
    assert scaled == Box(240, 180, 480, 360)


def test_scale_box_is_identity_when_sizes_match():
    b = Box(938, 381, 960, 440)
    assert scale_box(b, (960, 540), (960, 540)) is b


def test_scale_box_roundtrips_within_rounding():
    b = Box(100, 50, 200, 150)
    there = scale_box(b, (320, 240), (960, 720))
    back = scale_box(there, (960, 720), (320, 240))
    assert back == b


def test_scale_box_rejects_zero_source():
    with pytest.raises(ValueError):
        scale_box(Box(0, 0, 10, 10), (0, 240), (960, 720))


def test_scale_boxes_maps_all():
    boxes = [Box(0, 0, 10, 10), Box(10, 10, 20, 20)]
    assert scale_boxes(boxes, (100, 100), (200, 200)) == [Box(0, 0, 20, 20), Box(20, 20, 40, 40)]


# ---- payload parsing ----------------------------------------------------


def test_parse_flow_boxes_demo_format():
    """flow_boxes_demo.py: {"boxes_xyxy": [[...]]}, no processing_size."""
    boxes, size = parse_boxes_payload({"pair_index": 0, "num_boxes": 2, "boxes_xyxy": [[1, 2, 3, 4], [5, 6, 7, 8]]})
    assert boxes == [Box(1, 2, 3, 4), Box(5, 6, 7, 8)]
    assert size is None


def test_parse_boxes_trigger_format():
    """boxes_trigger.json: a bare list of dicts."""
    boxes, size = parse_boxes_payload([{"x1": 938, "y1": 381, "x2": 960, "y2": 440}])
    assert boxes == [Box(938, 381, 960, 440)]
    assert size is None


def test_parse_event_json_boxes_block():
    """event.json nests the boxes one level down and records processing_size."""
    boxes, size = parse_boxes_payload(
        {"boxes": {"enabled": True, "num_boxes": 1, "boxes_xyxy": [[938, 381, 960, 440]], "processing_size": [960, 540]}}
    )
    assert boxes == [Box(938, 381, 960, 440)]
    assert size == (960, 540)


def test_parse_empty_payload():
    boxes, size = parse_boxes_payload({"boxes_xyxy": []})
    assert boxes == [] and size is None


def test_parse_rejects_garbage():
    with pytest.raises(ValueError):
        parse_boxes_payload({"boxes_xyxy": [[1, 2, 3]]})
    with pytest.raises(ValueError):
        parse_boxes_payload("not a payload")


def test_load_boxes_json(tmp_path):
    p = tmp_path / "boxes.json"
    p.write_text(json.dumps({"boxes_xyxy": [[1, 2, 3, 4]]}))
    boxes, _ = load_boxes_json(p)
    assert boxes == [Box(1, 2, 3, 4)]


# ---- cropping -----------------------------------------------------------


def test_crop_plain():
    img = np.zeros((100, 200, 3), np.uint8)
    c = crop(img, Box(10, 20, 60, 70), pad_frac=0.0, square=False)
    assert c.shape[:2] == (50, 50)


def test_crop_squares_a_tall_box():
    img = np.zeros((400, 400, 3), np.uint8)
    c = crop(img, Box(180, 100, 220, 300), pad_frac=0.0, square=True)
    h, w = c.shape[:2]
    assert abs(h - w) <= 1, f"expected square-ish, got {w}x{h}"


def test_crop_pads():
    img = np.zeros((400, 400, 3), np.uint8)
    plain = crop(img, Box(100, 100, 200, 200), pad_frac=0.0, square=False)
    padded = crop(img, Box(100, 100, 200, 200), pad_frac=0.2, square=False)
    assert padded.shape[0] > plain.shape[0] and padded.shape[1] > plain.shape[1]


def test_crop_clamps_at_edges():
    """A box on the frame edge -- WAFT produces these (x2 == frame width)."""
    img = np.zeros((540, 960, 3), np.uint8)
    c = crop(img, Box(938, 381, 960, 440), pad_frac=0.15, square=True)
    assert c is not None and c.size > 0
    assert c.shape[0] <= 540 and c.shape[1] <= 960


def test_crop_rejects_degenerate_and_tiny():
    img = np.zeros((100, 100, 3), np.uint8)
    assert crop(img, Box(50, 50, 50, 50)) is None
    assert crop(img, Box(200, 200, 260, 260)) is None          # fully outside
    assert crop(img, Box(10, 10, 14, 14), pad_frac=0.0, min_size_px=8) is None  # too small


def test_crop_min_size_is_configurable():
    img = np.zeros((100, 100, 3), np.uint8)
    assert crop(img, Box(10, 10, 14, 14), pad_frac=0.0, square=False, min_size_px=2) is not None


# ---- WAFT event reading -------------------------------------------------


def _make_event(tmp_path: Path, *, with_boxes=True, frames=("pre_01", "trigger", "post_01")) -> Path:
    event_dir = tmp_path / "event_20260723_160545_000001_frame_00000035"
    (event_dir / "frames").mkdir(parents=True)
    (event_dir / "frames_annotated").mkdir(parents=True)

    manifest_frames = []
    for i, label in enumerate(frames):
        fname = f"{label}_2026072316054{i}_frame_0000003{i}.png"
        cv2.imwrite(str(event_dir / "frames" / fname), np.full((540, 960, 3), 30 + i * 10, np.uint8))
        # An annotated twin exists but must never be picked up.
        cv2.imwrite(str(event_dir / "frames_annotated" / fname.replace(".png", "_boxes.png")),
                    np.full((540, 960, 3), 200, np.uint8))
        manifest_frames.append({
            "label": label,
            "frame_index": 30 + i,
            "timestamp": f"2026072316054{i}",
            # Recorded relative to the WAFT repo root, as WAFT actually writes it.
            "event_path": f"outputs/camera_events/events/{event_dir.name}/frames/{fname}",
            "annotated_path": f"outputs/camera_events/events/{event_dir.name}/frames_annotated/{fname}",
        })

    manifest = {
        "event_id": 1,
        "event_name": event_dir.name,
        "trigger_mode": "farneback",
        "frames": manifest_frames,
    }
    if with_boxes:
        manifest["boxes"] = {
            "enabled": True,
            "num_boxes": 1,
            "boxes_xyxy": [[938, 381, 960, 440]],
            "processing_size": [960, 540],
        }
    (event_dir / "event.json").write_text(json.dumps(manifest))
    return event_dir


def test_load_event_reads_boxes_and_frames(tmp_path):
    event = load_event(_make_event(tmp_path))
    assert event.boxes == [Box(938, 381, 960, 440)]
    assert event.processing_size == (960, 540)
    assert [f.label for f in event.frames] == ["pre_01", "trigger", "post_01"]
    assert event.trigger_mode == "farneback"


def test_load_event_resolves_paths_after_relocation(tmp_path):
    """event.json paths are WAFT-root-relative and break on copy; we resolve by basename."""
    event_dir = _make_event(tmp_path)
    event = load_event(event_dir)
    assert len(event.frames) == 3
    for f in event.frames:
        assert f.path.exists()
        assert f.path.parent.name == "frames"  # never frames_annotated


def test_load_event_never_returns_annotated_frames(tmp_path):
    event = load_event(_make_event(tmp_path))
    assert all("annotated" not in str(f.path) for f in event.frames)


def test_load_event_sorts_burst_order_not_alphabetically(tmp_path):
    """Alphabetically 'post' < 'pre' < 'trigger', which is the wrong order."""
    event = load_event(_make_event(tmp_path, frames=("post_01", "pre_01", "trigger")))
    assert [f.label for f in event.frames] == ["pre_01", "trigger", "post_01"]


def test_pair_for_transition_spans_the_burst(tmp_path):
    event = load_event(_make_event(tmp_path, frames=("pre_01", "pre_02", "trigger", "post_01", "post_02")))
    before, after = event.pair_for_transition()
    assert before.label == "pre_01"
    assert after.label == "post_02"


def test_pair_for_transition_falls_back_without_pre_post(tmp_path):
    event = load_event(_make_event(tmp_path, frames=("trigger", "trigger")))
    pair = event.pair_for_transition()
    assert pair is not None and pair[0] is event.frames[0] and pair[1] is event.frames[-1]


def test_pair_for_transition_none_on_single_frame(tmp_path):
    event = load_event(_make_event(tmp_path, frames=("trigger",)))
    assert event.pair_for_transition() is None


def test_load_event_falls_back_to_boxes_trigger_sidecar(tmp_path):
    """Grid-triggered events have no boxes block in event.json."""
    event_dir = _make_event(tmp_path, with_boxes=False)
    (event_dir / "boxes_trigger.json").write_text(json.dumps([{"x1": 1, "y1": 2, "x2": 30, "y2": 40}]))
    event = load_event(event_dir)
    assert event.boxes == [Box(1, 2, 30, 40)]


def test_load_event_without_any_boxes(tmp_path):
    event = load_event(_make_event(tmp_path, with_boxes=False))
    assert event.boxes == []


def test_load_event_requires_manifest(tmp_path):
    (tmp_path / "not_an_event").mkdir()
    with pytest.raises(FileNotFoundError):
        load_event(tmp_path / "not_an_event")


def test_find_events_skips_non_event_dirs(tmp_path):
    _make_event(tmp_path)
    (tmp_path / "archive").mkdir()
    found = find_events(tmp_path)
    assert len(found) == 1 and found[0].name.startswith("event_")


def test_event_frame_phase():
    from vit_classifier.waft_events import EventFrame

    assert EventFrame("pre_01", Path("x.png")).phase == "pre"
    assert EventFrame("trigger", Path("x.png")).phase == "trigger"
    assert EventFrame("post_02", Path("x.png")).phase == "post"
