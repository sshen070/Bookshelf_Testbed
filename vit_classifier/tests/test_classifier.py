"""Transition semantics, cache invalidation, and the real ViT path.

The transition and fingerprint tests need no model. The model-backed tests
download ~85MB of pretrained weights on first run, so they're opt-in via
VIT_TEST_MODEL=1 rather than slowing every run and breaking offline ones.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vit_classifier.boxes import Box
from vit_classifier.classifier import UNKNOWN, RoiPrediction, describe_transition, exemplar_fingerprint
from vit_classifier.config import classifier_kwargs, load_config

RUN_MODEL = os.environ.get("VIT_TEST_MODEL") == "1"
_skip_model = pytest.mark.skipif(not RUN_MODEL, reason="Set VIT_TEST_MODEL=1 to run (downloads weights).")


def _pred(label: str, score: float = 0.9) -> RoiPrediction:
    return RoiPrediction(label, score)


# ---- transition semantics ----------------------------------------------


@pytest.mark.parametrize(
    "before,after,expected",
    [
        ("mug", "background", "removed"),
        ("background", "bottle", "placed"),
        ("mug", "bottle", "substituted"),
        ("mug", "mug", "moved_or_same_class"),
        ("background", "background", "moved_or_same_class"),
        (UNKNOWN, "bottle", "uncertain"),
        ("mug", UNKNOWN, "uncertain"),
        (UNKNOWN, UNKNOWN, "uncertain"),
    ],
)
def test_describe_transition(before, after, expected):
    assert describe_transition(_pred(before), _pred(after), "background") == expected


def test_describe_transition_honors_configured_background_class():
    assert describe_transition(_pred("mug"), _pred("bare_desk"), "bare_desk") == "removed"
    # Under a different background name the same pair is just a substitution.
    assert describe_transition(_pred("mug"), _pred("bare_desk"), "background") == "substituted"


def test_unknown_beats_background_check():
    """An unknown must never be reported as a removal, even against background."""
    assert describe_transition(_pred(UNKNOWN), _pred("background"), "background") == "uncertain"


# ---- config -------------------------------------------------------------


def test_default_config_flattens_for_the_classifier():
    kwargs = classifier_kwargs(load_config())
    assert kwargs["name"].startswith("vit_")
    assert kwargs["background_class"] == "background"
    assert 0.0 <= kwargs["min_similarity"] <= 1.0
    assert kwargs["knn_k"] >= 1


def test_classifier_kwargs_fills_defaults_for_empty_config():
    kwargs = classifier_kwargs({})
    assert kwargs["name"] == "vit_small_patch14_dinov2.lvd142m"
    assert kwargs["knn_k"] == 5


# ---- bank fingerprint ---------------------------------------------------


def test_fingerprint_changes_when_an_exemplar_is_added(tmp_path):
    d = tmp_path / "mug"
    d.mkdir()
    cv2.imwrite(str(d / "a.png"), np.full((32, 32, 3), 100, np.uint8))
    before = exemplar_fingerprint(tmp_path)

    cv2.imwrite(str(d / "b.png"), np.full((32, 32, 3), 150, np.uint8))
    assert exemplar_fingerprint(tmp_path) != before


def test_fingerprint_ignores_non_images(tmp_path):
    d = tmp_path / "mug"
    d.mkdir()
    cv2.imwrite(str(d / "a.png"), np.full((32, 32, 3), 100, np.uint8))
    before = exemplar_fingerprint(tmp_path)

    (tmp_path / "NOTES.txt").write_text("how this bank was collected")
    assert exemplar_fingerprint(tmp_path) == before


# ---- model-backed -------------------------------------------------------


def _patch(color: tuple[int, int, int], noise: int = 12, size: int = 96) -> np.ndarray:
    rng = np.random.RandomState(abs(hash(color)) % (2**31))
    img = np.full((size, size, 3), color, np.int16)
    img += rng.randint(-noise, noise + 1, (size, size, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def _bank(tmp_path: Path, classes: dict[str, tuple[int, int, int]], n: int = 5) -> Path:
    exemplar_dir = tmp_path / "exemplars"
    for name, color in classes.items():
        d = exemplar_dir / name
        d.mkdir(parents=True)
        for i in range(n):
            cv2.imwrite(str(d / f"{i}.png"), _patch(color))
    return exemplar_dir


@_skip_model
def test_bank_roundtrip_and_knn(tmp_path):
    pytest.importorskip("timm")
    from vit_classifier.classifier import RoiClassifier

    classes = {"mug": (40, 40, 190), "bottle": (40, 170, 60), "background": (35, 35, 35)}
    exemplar_dir = _bank(tmp_path, classes)

    clf = RoiClassifier({"img_size": 224, "device": "cpu", "knn_k": 3, "min_similarity": 0.0})
    cache = tmp_path / "bank.npz"
    clf.build_bank(exemplar_dir, cache)

    assert clf.embeddings.shape[0] == 15
    assert clf.classes == sorted(classes)
    assert np.allclose(np.linalg.norm(clf.embeddings, axis=1), 1.0, atol=1e-4)

    preds = clf.classify_crops([_patch(c, noise=20) for c in classes.values()])
    assert [p.label for p in preds] == list(classes)

    reloaded = RoiClassifier({"img_size": 224, "device": "cpu", "knn_k": 3, "min_similarity": 0.0})
    assert reloaded.load_bank(cache, exemplar_dir) is True
    assert reloaded.labels == clf.labels

    cv2.imwrite(str(exemplar_dir / "mug" / "extra.png"), _patch((40, 40, 190)))
    assert reloaded.load_bank(cache, exemplar_dir) is False


@_skip_model
def test_unseen_object_is_unknown_not_nearest_class(tmp_path):
    pytest.importorskip("timm")
    from vit_classifier.classifier import RoiClassifier

    exemplar_dir = _bank(tmp_path, {"mug": (40, 40, 190)}, n=3)
    clf = RoiClassifier({"img_size": 224, "device": "cpu", "knn_k": 3, "min_similarity": 0.999})
    clf.build_bank(exemplar_dir, None)

    rng = np.random.RandomState(7)
    stranger = rng.randint(0, 256, (96, 96, 3), dtype=np.uint8)
    assert clf.classify_crops([stranger])[0].label == UNKNOWN


@_skip_model
def test_classify_pair_detects_removal(tmp_path):
    """The two-frame trick: same box, before and after, one removal."""
    pytest.importorskip("timm")
    from vit_classifier.classifier import RoiClassifier

    exemplar_dir = _bank(tmp_path, {"mug": (40, 40, 190), "background": (35, 35, 35)}, n=4)
    clf = RoiClassifier(
        {"img_size": 224, "device": "cpu", "knn_k": 3, "min_similarity": 0.0, "background_class": "background"}
    )
    clf.build_bank(exemplar_dir, None)

    box = Box(100, 100, 200, 200)
    before = np.full((300, 300, 3), 35, np.uint8)
    before[100:200, 100:200] = _patch((40, 40, 190), size=100)   # object present
    after = np.full((300, 300, 3), 35, np.uint8)                  # object gone

    (tr,) = clf.classify_pair(before, after, [box])
    assert tr.before.label == "mug"
    assert tr.after.label == "background"
    assert tr.event == "removed"


@_skip_model
def test_classify_pair_detects_placement(tmp_path):
    pytest.importorskip("timm")
    from vit_classifier.classifier import RoiClassifier

    exemplar_dir = _bank(tmp_path, {"mug": (40, 40, 190), "background": (35, 35, 35)}, n=4)
    clf = RoiClassifier(
        {"img_size": 224, "device": "cpu", "knn_k": 3, "min_similarity": 0.0, "background_class": "background"}
    )
    clf.build_bank(exemplar_dir, None)

    box = Box(100, 100, 200, 200)
    before = np.full((300, 300, 3), 35, np.uint8)
    after = np.full((300, 300, 3), 35, np.uint8)
    after[100:200, 100:200] = _patch((40, 40, 190), size=100)

    (tr,) = clf.classify_pair(before, after, [box])
    assert tr.event == "placed"


@_skip_model
def test_classify_pair_rejects_mismatched_frame_sizes(tmp_path):
    pytest.importorskip("timm")
    from vit_classifier.classifier import RoiClassifier

    exemplar_dir = _bank(tmp_path, {"mug": (40, 40, 190)}, n=3)
    clf = RoiClassifier({"img_size": 224, "device": "cpu", "knn_k": 3})
    clf.build_bank(exemplar_dir, None)

    with pytest.raises(ValueError, match="differ in size"):
        clf.classify_pair(np.zeros((100, 100, 3), np.uint8), np.zeros((200, 200, 3), np.uint8), [Box(0, 0, 50, 50)])


@_skip_model
def test_classify_keeps_alignment_when_a_box_is_uncroppable(tmp_path):
    """Output must stay 1:1 with input boxes even when one can't be cropped."""
    pytest.importorskip("timm")
    from vit_classifier.classifier import RoiClassifier

    exemplar_dir = _bank(tmp_path, {"mug": (40, 40, 190)}, n=3)
    clf = RoiClassifier({"img_size": 224, "device": "cpu", "knn_k": 3, "min_similarity": 0.0})
    clf.build_bank(exemplar_dir, None)

    img = np.full((300, 300, 3), 35, np.uint8)
    img[100:200, 100:200] = _patch((40, 40, 190), size=100)
    boxes = [Box(100, 100, 200, 200), Box(500, 500, 560, 560)]  # second is off-image

    preds = clf.classify(img, boxes)
    assert len(preds) == 2
    assert preds[0].label == "mug"
    assert preds[1].label == UNKNOWN
