"""Config resolution and the comparison report. No torch, no ultralytics."""
from pathlib import Path

from yolo_world.compare import compare_regions, load_vit_result, vit_regions
from yolo_world.config import PACKAGE_DIR, detector_kwargs, events_dir, load_config, resolve
from yolo_world.events import WaftEvent


class FakeDetection:
    def __init__(self, label, score, box):
        self.label, self.score, self.box_xyxy = label, score, box

    def to_dict(self):
        return {"label": self.label, "score": self.score, "box_xyxy": self.box_xyxy}


# ---- config -------------------------------------------------------------


def test_shipped_config_loads():
    cfg = load_config()
    assert cfg["model"]["name"].endswith(".pt")
    assert cfg["vocabulary"]["active"] in cfg["vocabulary"]["sets"]


def test_relative_paths_resolve_against_the_package_not_cwd():
    assert resolve("exemplars").parent == PACKAGE_DIR
    assert resolve("/absolute/path") == Path("/absolute/path")


def test_detector_kwargs_flattens_and_types():
    kw = detector_kwargs({"model": {"name": "x.pt", "imgsz": "320", "conf": "0.1"}})
    assert kw["imgsz"] == 320 and isinstance(kw["imgsz"], int)
    assert kw["conf"] == 0.1 and isinstance(kw["conf"], float)


def test_detector_kwargs_defaults_when_config_is_bare():
    kw = detector_kwargs({})
    assert kw["name"] == "yolov8s-worldv2.pt" and kw["imgsz"] == 640


def test_events_dir_points_at_wafts_tree():
    assert events_dir(load_config()).name == "events"


# ---- compare ------------------------------------------------------------


def _event(boxes, size=(960, 540)):
    return WaftEvent(event_dir=Path("/tmp/e"), event_name="e", status="ready_for_waft",
                     frames=[], boxes_xyxy=boxes, processing_size=size)


def test_matching_region_is_reported_with_both_verdicts():
    event = _event([[0, 0, 100, 100]])
    vit = {"regions": [{"box_xyxy": [0, 0, 100, 100], "event": "uncertain",
                        "after": {"label": "books"}}]}
    report = compare_regions(event, [FakeDetection("book", 0.5, [0, 0, 100, 100])],
                             (960, 540), vit)
    row = report["rows"][0]
    assert row["world_label"] == "book"
    assert row["vit_verdict"] == "uncertain"
    assert row["vit_after"] == "books"
    assert report["n_matched"] == 1


def test_detection_in_a_region_motion_never_flagged_is_surfaced():
    # The whole point of the single-stage argument: static objects.
    event = _event([[0, 0, 50, 50]])
    dets = [FakeDetection("pillar", 0.4, [500, 400, 600, 500])]
    report = compare_regions(event, dets, (960, 540), None)
    assert report["n_matched"] == 0
    assert report["unmatched_world"][0]["label"] == "pillar"


def test_boxes_are_rescaled_before_comparison():
    # Motion boxes recorded at 480x270 against a 960x540 frame must be scaled,
    # or every IoU is computed against the wrong quadrant.
    event = _event([[0, 0, 50, 50]], size=(480, 270))
    report = compare_regions(event, [FakeDetection("book", 0.5, [0, 0, 100, 100])],
                             (960, 540), None)
    assert report["rows"][0]["motion_box"] == [0, 0, 100, 100]
    assert report["rows"][0]["iou"] == 1.0


def test_below_threshold_iou_is_not_a_match():
    event = _event([[0, 0, 100, 100]])
    dets = [FakeDetection("book", 0.5, [90, 90, 200, 200])]
    report = compare_regions(event, dets, (960, 540), None, iou_match=0.3)
    assert report["rows"][0]["world_label"] is None
    assert report["n_matched"] == 0


def test_missing_vit_result_is_not_fatal(tmp_path):
    assert load_vit_result(tmp_path) is None
    assert vit_regions(None) == []


def test_half_written_vit_result_reads_as_absent(tmp_path):
    # Both WAFT processes rewrite JSON in place; a poll can land mid-write.
    (tmp_path / "vit_result.json").write_text('{"regions": [')
    assert load_vit_result(tmp_path) is None
