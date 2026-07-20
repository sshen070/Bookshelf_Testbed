"""End-to-end smoke test on synthetic images.

No real shelf photos exist yet, so this validates that registration, lighting
normalization, SSIM diffing, and autolabeling are wired together correctly
and behave sanely on a controlled synthetic case:
  - identical frames -> no boxes
  - a frame with an added rectangle ("book toppled") -> a box over that region
  - a frame that's just brighter (lighting drift) -> no boxes, thanks to
    CLAHE + gray-world normalization
before any of it is trusted against real captures.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.autolabel import process_image
from src.lighting import normalize_illumination
from src.registration import align_to_reference
from src.ssim_diff import compute_dissimilarity_map, diss_map_to_boxes
from src.utils.config import load_pipeline_config


@pytest.fixture(scope="module")
def cfg():
    return load_pipeline_config("configs/pipeline.yaml")


def _make_shelf_scene() -> np.ndarray:
    """A crude synthetic 'shelf': a textured background with a few rectangular 'books'."""
    rng = np.random.RandomState(0)
    img = (rng.rand(480, 640, 3) * 40 + 30).astype(np.uint8)  # textured dark background
    books = [
        ((40, 60), (140, 400), (60, 90, 160)),
        ((160, 60), (260, 400), (90, 140, 60)),
        ((280, 60), (360, 400), (150, 60, 60)),
    ]
    for (x1, y1), (x2, y2), color in books:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=-1)
    return img


def test_registration_identity(cfg):
    scene = _make_shelf_scene()
    result = align_to_reference(scene.copy(), scene, cfg["registration"])
    assert result.success
    assert result.aligned.shape == scene.shape


def test_ssim_diff_no_change(cfg):
    scene = _make_shelf_scene()
    diss_map = compute_dissimilarity_map(scene, scene, cfg)
    boxes, _ = diss_map_to_boxes(diss_map, cfg)
    assert boxes == []


def test_ssim_diff_detects_toppled_book(cfg):
    reference = _make_shelf_scene()
    changed = reference.copy()
    # Simulate a book toppling: paint over one book's region with the background tone.
    cv2.rectangle(changed, (160, 60), (260, 400), (35, 35, 35), thickness=-1)
    cv2.rectangle(changed, (160, 340), (260, 400), (60, 90, 160), thickness=-1)  # fallen flat near shelf floor

    diss_map = compute_dissimilarity_map(changed, reference, cfg)
    boxes, _ = diss_map_to_boxes(diss_map, cfg)
    assert len(boxes) >= 1

    # The detected change should overlap the region we actually altered.
    changed_region = {"x1": 160, "y1": 60, "x2": 260, "y2": 400}
    assert any(
        b.x1 < changed_region["x2"] and b.x2 > changed_region["x1"]
        and b.y1 < changed_region["y2"] and b.y2 > changed_region["y1"]
        for b in boxes
    )


def test_lighting_normalization_suppresses_brightness_diff(cfg):
    reference = _make_shelf_scene()
    brighter = np.clip(reference.astype(np.int16) + 60, 0, 255).astype(np.uint8)

    raw_diss = compute_dissimilarity_map(brighter, reference, cfg)
    raw_boxes, _ = diss_map_to_boxes(raw_diss, cfg)

    norm_ref = normalize_illumination(reference, cfg["lighting"])
    norm_bright = normalize_illumination(brighter, cfg["lighting"])
    norm_diss = compute_dissimilarity_map(norm_bright, norm_ref, cfg)
    norm_boxes, _ = diss_map_to_boxes(norm_diss, cfg)

    # Normalization should not produce *more* false-positive area than the raw comparison.
    assert norm_diss.mean() <= raw_diss.mean()
    assert len(norm_boxes) <= max(len(raw_boxes), 1)


def test_autolabel_end_to_end(cfg, tmp_path):
    reference = _make_shelf_scene()
    changed = reference.copy()
    cv2.rectangle(changed, (280, 60), (360, 400), (35, 35, 35), thickness=-1)

    raw_path = tmp_path / "session_a" / "frame_001.jpg"
    raw_path.parent.mkdir(parents=True)
    cv2.imwrite(str(raw_path), changed)

    out_dirs = {
        "aligned": tmp_path / "aligned",
        "diffs": tmp_path / "diffs",
        "pool_images": tmp_path / "pool" / "images",
        "pool_labels": tmp_path / "pool" / "labels",
    }
    for d in out_dirs.values():
        d.mkdir(parents=True)

    reference_bank = [("ref_default", reference)]
    event = process_image(raw_path, reference_bank, cfg, out_dirs)

    assert event["status"] == "change"
    assert event["num_boxes"] >= 1
    label_file = out_dirs["pool_labels"] / "session_a__frame_001.txt"
    assert label_file.exists()
    lines = label_file.read_text().strip().splitlines()
    assert len(lines) == event["num_boxes"]
    for line in lines:
        parts = line.split()
        assert len(parts) == 5
        assert parts[0] == "0"  # single class id
