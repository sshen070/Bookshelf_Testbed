"""Event parsing and geometry. No torch, no ultralytics, no weights."""
import json

import pytest

from yolo_world.events import find_events, iou, load_event, scale_box


def write_event(tmp_path, frames=None, boxes=None, size=(960, 540), status="ready_for_waft"):
    event_dir = tmp_path / "event_test_000001"
    (event_dir / "frames").mkdir(parents=True)
    (event_dir / "frames_annotated").mkdir()
    frames = frames if frames is not None else [
        ("pre_01", 25), ("pre_02", 26), ("trigger", 29), ("post_01", 30),
    ]
    entries = []
    for label, idx in frames:
        fname = f"{label}_ts_frame_{idx:08d}.png"
        (event_dir / "frames" / fname).write_bytes(b"stub")
        # Recorded paths are relative to the WAFT repo root and are expected
        # to be stale; resolution must fall back to basename.
        entries.append({
            "label": label, "frame_index": idx,
            "event_path": f"outputs/somewhere/else/frames/{fname}",
        })
    manifest = {
        "event_name": "event_test_000001", "status": status, "frames": entries,
        "boxes": {"boxes_xyxy": boxes if boxes is not None else [[0, 0, 10, 10]],
                  "processing_size": list(size)},
    }
    (event_dir / "event.json").write_text(json.dumps(manifest))
    return event_dir


def test_loads_frames_by_basename_when_recorded_path_is_stale(tmp_path):
    event = load_event(write_event(tmp_path))
    assert len(event.frames) == 4
    assert all(f.path.is_file() for f in event.frames)


def test_frames_sorted_pre_trigger_post(tmp_path):
    event = load_event(write_event(
        tmp_path, frames=[("post_01", 30), ("trigger", 29), ("pre_01", 25)]))
    assert [f.label for f in event.frames] == ["pre_01", "trigger", "post_01"]


def test_frame_lookup_by_exact_label_then_phase(tmp_path):
    event = load_event(write_event(tmp_path))
    assert event.frame("post_01").label == "post_01"
    assert event.frame("pre").label == "pre_01"   # first of the phase
    assert event.frame("nope") is None


def test_ready_reflects_status(tmp_path):
    assert load_event(write_event(tmp_path)).ready
    assert not load_event(write_event(tmp_path / "b", status="pending_post_frames")).ready


def test_annotated_frames_are_never_loaded(tmp_path):
    event_dir = write_event(tmp_path)
    (event_dir / "frames_annotated" / "trigger_ts_frame_00000029_boxes.png").write_bytes(b"x")
    event = load_event(event_dir)
    assert all("frames_annotated" not in str(f.path) for f in event.frames)


def test_missing_manifest_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        load_event(tmp_path / "empty")


def test_find_events_skips_dirs_without_manifest(tmp_path):
    write_event(tmp_path)
    (tmp_path / "not_an_event").mkdir()
    assert [p.name for p in find_events(tmp_path)] == ["event_test_000001"]


def test_find_events_on_missing_dir_is_empty(tmp_path):
    assert find_events(tmp_path / "nope") == []


def test_iou_identical_boxes():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)


def test_iou_disjoint_boxes():
    assert iou([0, 0, 10, 10], [50, 50, 60, 60]) == 0.0


def test_iou_half_overlap():
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)


def test_scale_box_matches_the_resolution_change():
    # The scaling trap: a 480x270 box addressed against a 960x540 frame.
    assert scale_box([10, 10, 20, 20], (480, 270), (960, 540)) == [20, 20, 40, 40]


def test_scale_box_identity_when_sizes_match():
    assert scale_box([1, 2, 3, 4], (960, 540), (960, 540)) == [1, 2, 3, 4]
