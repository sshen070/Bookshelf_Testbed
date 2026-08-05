"""Leave-one-out evaluation of an embedded exemplar bank.

Answers the only question that matters after `build-bank`: are these classes
actually separable, or does the bank just look full? Run it after every
rebuild -- adding exemplars can *hurt*, and a k-NN gives you no training curve
to notice that from.

Operates on an already-embedded bank (the arrays `build_bank` produces), so
this module imports without torch or timm. The vote itself is
`classifier.knn_vote`, the same function inference calls, so what is measured
here is what will happen at inference rather than a reimplementation that
drifts.

Two numbers to read:

- **Leave-one-out accuracy.** Each exemplar is classified against a bank with
  itself removed. Removal is the whole point: left in, every exemplar is its
  own nearest neighbour at similarity 1.0 and the score is a meaningless 100%.
- **The class similarity matrix.** Mean cosine similarity within each class
  (diagonal) against between each pair (off-diagonal). A class is separable
  when its diagonal clearly exceeds its row. This localizes a failure in a way
  accuracy cannot: two classes bleeding into each other shows up as a hot
  off-diagonal cell long before it costs you enough accuracy to notice.

Both are optimistic on a small bank of similar photos, and neither measures
the thing most likely to break in the field -- exemplars shot on a phone
versus crops the detector actually produced. Treat a high score here as
"nothing is fundamentally broken", not as field accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .classifier import knn_vote


@dataclass
class LooCase:
    """One held-out exemplar, scored against the rest of the bank."""

    path: str
    true_label: str
    predicted: str
    score: float
    votes: dict[str, int]

    @property
    def correct(self) -> bool:
        return self.predicted == self.true_label


def leave_one_out(
    embeddings: np.ndarray,
    labels: list[str],
    paths: list[str],
    knn_k: int,
    min_similarity: float,
) -> list[LooCase]:
    """Classify each exemplar against a bank with that exemplar removed."""
    n = len(labels)
    if n < 2:
        raise ValueError(f"Need at least 2 exemplars to leave one out, got {n}.")

    cases = []
    for i in range(n):
        keep = np.arange(n) != i
        sims = embeddings[i] @ embeddings[keep].T
        rest_labels = [label for j, label in enumerate(labels) if j != i]
        pred = knn_vote(sims, rest_labels, knn_k, min_similarity)
        cases.append(LooCase(paths[i], labels[i], pred.label, pred.score, pred.votes))
    return cases


def class_similarity(
    embeddings: np.ndarray, labels: list[str]
) -> tuple[list[str], np.ndarray]:
    """Mean cosine similarity within and between every pair of classes.

    The within-class diagonal excludes each exemplar's similarity to itself,
    which is 1.0 by construction and would inflate every diagonal toward the
    class looking tighter than it is. A class holding a single exemplar has no
    within-class pair at all and comes back NaN rather than a fabricated 1.0.
    """
    classes = sorted(set(labels))
    by_class = {c: [i for i, label in enumerate(labels) if label == c] for c in classes}

    matrix = np.full((len(classes), len(classes)), np.nan, dtype=np.float32)
    for a, ca in enumerate(classes):
        for b, cb in enumerate(classes):
            block = embeddings[by_class[ca]] @ embeddings[by_class[cb]].T
            if ca == cb:
                if len(by_class[ca]) < 2:
                    continue
                values = block[~np.eye(len(by_class[ca]), dtype=bool)]
            else:
                values = block.ravel()
            if values.size:
                matrix[a, b] = float(values.mean())
    return classes, matrix


def format_report(
    cases: list[LooCase],
    classes: list[str],
    matrix: np.ndarray,
    knn_k: int,
    min_similarity: float,
) -> str:
    lines = [f"\nLeave-one-out over {len(cases)} exemplar(s)  "
             f"(knn_k={knn_k}, min_similarity={min_similarity})\n"]

    for case in sorted(cases, key=lambda c: (c.true_label, c.path)):
        mark = "ok  " if case.correct else "FAIL"
        votes = ", ".join(f"{k}:{v}" for k, v in sorted(case.votes.items()))
        lines.append(
            f"  {mark} {case.path:38s} {case.true_label} -> {case.predicted}"
            f"  sim={case.score:.3f}  [{votes}]"
        )

    correct = sum(c.correct for c in cases)
    rejected = sum(c.predicted == "unknown" for c in cases)
    lines.append(f"\n  accuracy: {correct}/{len(cases)} = {correct / len(cases):.1%}"
                 f"   (rejected as unknown: {rejected})")

    lines.append("\n  per class:")
    for c in classes:
        of_class = [x for x in cases if x.true_label == c]
        hits = sum(x.correct for x in of_class)
        lines.append(f"    {c:20s} {hits}/{len(of_class)}")

    # A misclassification that keeps landing on the same wrong class is a
    # merge candidate; one that scatters is usually a bad exemplar.
    failures = [c for c in cases if not c.correct]
    if failures:
        lines.append("\n  failures:")
        for f in failures:
            why = ("scored below min_similarity" if f.predicted == "unknown"
                   else f"confused with {f.predicted}")
            lines.append(f"    {f.path:38s} {why} (sim={f.score:.3f})")

    lines.append("\n  mean cosine similarity between classes")
    lines.append("  " + " " * 20 + "".join(f"{c[:11]:>13s}" for c in classes))
    for a, ca in enumerate(classes):
        row = f"  {ca[:20]:20s}"
        for b in range(len(classes)):
            value = matrix[a, b]
            row += f"{'n/a':>13s}" if np.isnan(value) else f"{value:13.3f}"
        lines.append(row)
    lines.append("\n  (diagonal = within class, off-diagonal = between. A class is")
    lines.append("   separable when its diagonal clearly exceeds its own row.)")

    return "\n".join(lines)
