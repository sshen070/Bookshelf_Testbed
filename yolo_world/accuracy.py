"""Score detections against hand-labelled regions.

The two systems answer different questions — vit_classifier is *given* a region
and names it; YOLO-World finds regions and names them — so scoring them
against each other directly is not meaningful. What is meaningful is a common
task: **for this hand-labelled region, what label does each produce?**

YOLO-World is assigned a region's label by taking its highest-scoring
detection that overlaps the region (IoU >= match_iou). A region no detection
covers is a MISS, which is a real failure mode and is counted as one — the
classifier cannot miss, because it is handed the box.

Ground truth is a JSON list of [label, [x1,y1,x2,y2]] against a single frame.
It has to be written by looking at the image; there is none in the tree.
"""
from __future__ import annotations

import json
from pathlib import Path

from .events import iou

MISS = "(none)"


def load_truth(path: str | Path) -> list[tuple[str, list[int]]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    truth = []
    for i, row in enumerate(raw):
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError(
                f"Ground-truth row {i} should be [label, [x1,y1,x2,y2]], got {row!r}"
            )
        label, box = row
        if len(box) != 4:
            raise ValueError(f"Ground-truth row {i} box needs 4 numbers, got {box!r}")
        truth.append((str(label), [int(v) for v in box]))
    return truth


def assign(truth, detections, match_iou: float = 0.3) -> list[dict]:
    """Give each ground-truth region the best-overlapping detection's label."""
    rows = []
    for label, box in truth:
        best, best_iou = None, 0.0
        for d in detections:
            v = iou(box, d.box_xyxy)
            if v >= match_iou and v > best_iou and (best is None or d.score > best.score):
                best, best_iou = d, v
        rows.append({
            "truth": label,
            "box": box,
            "predicted": best.label if best else MISS,
            "score": round(best.score, 3) if best else 0.0,
            "prompt": best.prompt if best else "",
            "iou": round(best_iou, 3),
            "correct": bool(best and best.label == label),
        })
    return rows


def score(rows: list[dict]) -> dict:
    total = len(rows)
    correct = sum(r["correct"] for r in rows)
    missed = sum(r["predicted"] == MISS for r in rows)
    per_class: dict[str, dict] = {}
    for r in rows:
        c = per_class.setdefault(r["truth"], {"n": 0, "correct": 0, "missed": 0})
        c["n"] += 1
        c["correct"] += r["correct"]
        c["missed"] += r["predicted"] == MISS
    return {
        "total": total,
        "correct": correct,
        "missed": missed,
        "accuracy": correct / total if total else 0.0,
        "per_class": per_class,
    }


def merge_external(rows: list[dict], external: dict | None) -> list[dict]:
    """Fold in another system's predictions, keyed by ground-truth box."""
    if not external:
        return rows
    by_box = {tuple(map(int, k.split(","))): v for k, v in external.items()}
    for r in rows:
        hit = by_box.get(tuple(r["box"]))
        if hit is not None:
            r["other"] = hit.get("label", "?")
            r["other_score"] = round(float(hit.get("score", 0.0)), 3)
            r["other_correct"] = r["other"] == r["truth"]
    return rows


def format_report(rows: list[dict], stats: dict, other_name: str | None = None) -> str:
    has_other = other_name and any("other" in r for r in rows)
    w_t, w_p = 14, 22
    head = f"  {'truth':<{w_t}}{'YOLO-World':<{w_p}}{'':>7}"
    if has_other:
        head += f"  {other_name:<{w_p}}"
    lines = ["", head, "  " + "-" * (len(head) - 2)]

    for r in rows:
        mark = "OK " if r["correct"] else "   "
        pred = r["predicted"]
        if r["predicted"] != MISS:
            pred = f"{pred} ({r['score']:.2f})"
        line = f"  {r['truth']:<{w_t}}{pred:<{w_p}}{mark:>7}"
        if has_other and "other" in r:
            om = "OK" if r.get("other_correct") else "  "
            line += f"  {r['other'] + ' (' + format(r['other_score'], '.2f') + ')':<{w_p}}{om}"
        lines.append(line)

    lines.append("")
    lines.append(f"  YOLO-World   {stats['correct']}/{stats['total']} correct "
                 f"({stats['accuracy'] * 100:.0f}%), {stats['missed']} region(s) "
                 f"no detection covered")
    if has_other:
        oc = sum(1 for r in rows if r.get("other_correct"))
        lines.append(f"  {other_name:<12} {oc}/{len(rows)} correct "
                     f"({oc / len(rows) * 100:.0f}%), 0 missed "
                     f"(it is handed the box, so it cannot miss)")

    lines.append("\n  per class:")
    for cls, c in sorted(stats["per_class"].items()):
        lines.append(f"    {cls:<14}{c['correct']}/{c['n']}"
                     + (f"   ({c['missed']} missed)" if c["missed"] else ""))
    return "\n".join(lines)
