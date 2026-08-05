"""Leave-one-out evaluation and the class similarity matrix.

No model needed: the eval operates on already-embedded vectors, so these
tests hand it small hand-built unit vectors whose geometry is obvious and
assert the arithmetic, rather than embedding real crops.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vit_classifier.classifier import UNKNOWN, knn_vote
from vit_classifier.eval_bank import class_similarity, format_report, leave_one_out


def _unit(*components: float) -> np.ndarray:
    v = np.array(components, dtype=np.float32)
    return v / np.linalg.norm(v)


def _orthogonal_bank() -> tuple[np.ndarray, list[str], list[str]]:
    """Two classes on orthogonal axes -- trivially separable, tightly clustered."""
    embeddings = np.stack([
        _unit(1.0, 0.0, 0.0),
        _unit(0.98, 0.2, 0.0),
        _unit(0.96, 0.28, 0.0),
        _unit(0.0, 0.0, 1.0),
        _unit(0.0, 0.2, 0.98),
        _unit(0.0, 0.28, 0.96),
    ]).astype(np.float32)
    labels = ["a", "a", "a", "b", "b", "b"]
    paths = [f"{c}/{i}.png" for c, i in zip(labels, [1, 2, 3, 1, 2, 3])]
    return embeddings, labels, paths


# ---- knn_vote ------------------------------------------------------------


def test_knn_vote_picks_the_majority_class():
    sims = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    pred = knn_vote(sims, ["a", "b", "b"], knn_k=3, min_similarity=0.0)
    assert pred.label == "b"
    assert pred.votes == {"a": 1, "b": 2}


def test_knn_vote_breaks_a_tie_on_the_closer_neighbour():
    """2-2 split: the class holding the single closest neighbour wins."""
    sims = np.array([0.9, 0.5, 0.85, 0.55], dtype=np.float32)
    pred = knn_vote(sims, ["a", "a", "b", "b"], knn_k=4, min_similarity=0.0)
    assert pred.label == "a"
    assert pred.votes == {"a": 2, "b": 2}


def test_knn_vote_rejects_below_min_similarity():
    sims = np.array([0.3, 0.2, 0.1], dtype=np.float32)
    pred = knn_vote(sims, ["a", "a", "a"], knn_k=3, min_similarity=0.55)
    assert pred.label == UNKNOWN
    # The vote still happened and is reported -- only the label was withheld.
    assert pred.votes == {"a": 3}
    assert pred.score == pytest.approx(0.3, abs=1e-6)


def test_knn_vote_clamps_k_to_the_bank_size():
    sims = np.array([0.9, 0.8], dtype=np.float32)
    pred = knn_vote(sims, ["a", "b"], knn_k=10, min_similarity=0.0)
    assert sum(pred.votes.values()) == 2


# ---- leave_one_out -------------------------------------------------------


def test_leave_one_out_scores_every_exemplar():
    embeddings, labels, paths = _orthogonal_bank()
    cases = leave_one_out(embeddings, labels, paths, knn_k=3, min_similarity=0.0)
    assert len(cases) == len(labels)
    assert all(c.correct for c in cases)
    assert [c.path for c in cases] == paths


def test_leave_one_out_actually_holds_the_exemplar_out():
    """A duplicated exemplar must not vote for itself.

    Two 'a's identical to each other and one 'b': with k=1 and the exemplar
    left in, every case would score a trivial 1.0. Held out, the lone 'b' has
    no same-class neighbour and must be misclassified -- which is the signal
    that the hold-out is real.
    """
    embeddings = np.stack([
        _unit(1.0, 0.0),
        _unit(1.0, 0.0),
        _unit(0.9, 0.44),
    ]).astype(np.float32)
    cases = leave_one_out(embeddings, ["a", "a", "b"], ["a/1.png", "a/2.png", "b/1.png"],
                          knn_k=1, min_similarity=0.0)
    assert cases[0].correct and cases[1].correct
    assert not cases[2].correct
    assert cases[2].predicted == "a"
    assert cases[2].score < 1.0


def test_leave_one_out_rejects_a_bank_too_small_to_hold_out():
    with pytest.raises(ValueError, match="at least 2"):
        leave_one_out(np.array([[1.0, 0.0]], dtype=np.float32), ["a"], ["a/1.png"],
                      knn_k=1, min_similarity=0.0)


def test_leave_one_out_honors_min_similarity():
    """A correct vote below threshold is still recorded as a failure."""
    embeddings, labels, paths = _orthogonal_bank()
    cases = leave_one_out(embeddings, labels, paths, knn_k=3, min_similarity=0.999)
    assert all(c.predicted == UNKNOWN for c in cases)
    assert not any(c.correct for c in cases)


# ---- class_similarity ----------------------------------------------------


def test_class_similarity_separates_orthogonal_classes():
    embeddings, labels, _ = _orthogonal_bank()
    classes, matrix = class_similarity(embeddings, labels)
    assert classes == ["a", "b"]
    # Each diagonal must dominate its own row.
    assert matrix[0, 0] > matrix[0, 1]
    assert matrix[1, 1] > matrix[1, 0]
    assert matrix[0, 1] == pytest.approx(matrix[1, 0], abs=1e-5)


def test_class_similarity_excludes_self_similarity_from_the_diagonal():
    """Self-similarity is 1.0 by construction and would inflate the diagonal."""
    embeddings = np.stack([_unit(1.0, 0.0), _unit(0.6, 0.8)]).astype(np.float32)
    _, matrix = class_similarity(embeddings, ["a", "a"])
    assert matrix[0, 0] == pytest.approx(0.6, abs=1e-5)


def test_class_similarity_reports_nan_for_a_single_exemplar_class():
    embeddings = np.stack([_unit(1.0, 0.0), _unit(0.0, 1.0), _unit(0.1, 0.99)]).astype(np.float32)
    classes, matrix = class_similarity(embeddings, ["a", "b", "b"])
    assert classes == ["a", "b"]
    assert np.isnan(matrix[0, 0])       # 'a' has no within-class pair
    assert not np.isnan(matrix[1, 1])


# ---- report --------------------------------------------------------------


def test_format_report_names_the_failing_exemplar_and_its_reason():
    embeddings, labels, paths = _orthogonal_bank()
    cases = leave_one_out(embeddings, labels, paths, knn_k=3, min_similarity=0.999)
    classes, matrix = class_similarity(embeddings, labels)
    report = format_report(cases, classes, matrix, knn_k=3, min_similarity=0.999)

    assert "0/6" in report or "0.0%" in report
    assert "scored below min_similarity" in report
    assert paths[0] in report


def test_format_report_renders_a_nan_diagonal_as_not_applicable():
    embeddings = np.stack([_unit(1.0, 0.0), _unit(0.0, 1.0), _unit(0.1, 0.99)]).astype(np.float32)
    labels, paths = ["a", "b", "b"], ["a/1.png", "b/1.png", "b/2.png"]
    cases = leave_one_out(embeddings, labels, paths, knn_k=1, min_similarity=0.0)
    classes, matrix = class_similarity(embeddings, labels)
    report = format_report(cases, classes, matrix, knn_k=1, min_similarity=0.0)
    assert "n/a" in report
