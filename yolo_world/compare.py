"""Side-by-side against vit_classifier on the same event.

The question this answers is not "which one scores higher" -- the two systems
do not emit comparable scores, and there is no ground truth in the tree to
score against. It is the more useful structural question: **do they even
agree about which regions are interesting?**

That matters because the two find regions in fundamentally different ways.
vit_classifier is handed motion boxes, which bracket whatever MOVED -- in
practice the operator's hand. YOLO-World finds whatever matches a prompt,
moving or not. A low region overlap is therefore the expected result, not a
bug, and it is the single clearest illustration of what switching
architectures would change.

Reads vit_result.json if present; never writes into the event directory.
"""
from __future__ import annotations

import json
from pathlib import Path

from .events import WaftEvent, iou, scale_box


def load_vit_result(event_dir: Path, name: str = "vit_result.json") -> dict | None:
    path = Path(event_dir) / name
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        # Both WAFT processes rewrite JSON in place, so a read can land
        # mid-write. A half-written file means "not available", not a crash.
        return None


def vit_regions(result: dict | None) -> list[dict]:
    return list(result.get("regions", [])) if result else []


def compare_regions(
    event: WaftEvent,
    detections: list,
    frame_size: tuple[int, int],
    vit_result: dict | None,
    iou_match: float = 0.3,
) -> dict:
    """Match YOLO-World detections against the motion boxes the ViT was given."""
    motion = event.boxes_xyxy
    if event.processing_size and tuple(event.processing_size) != frame_size:
        motion = [scale_box(b, tuple(event.processing_size), frame_size)
                  for b in motion]

    vit = vit_regions(vit_result)
    vit_by_box = {}
    for r in vit:
        box = r.get("box_xyxy")
        if box:
            vit_by_box[tuple(box)] = r

    rows = []
    matched_det = set()
    for i, mbox in enumerate(motion):
        best_j, best_iou = None, 0.0
        for j, det in enumerate(detections):
            v = iou(mbox, det.box_xyxy)
            if v > best_iou:
                best_j, best_iou = j, v

        vit_entry = None
        for box, r in vit_by_box.items():
            if iou(mbox, list(box)) > 0.9:
                vit_entry = r
                break

        row = {
            "motion_box": mbox,
            "vit_verdict": (vit_entry or {}).get("event"),
            "vit_after": ((vit_entry or {}).get("after") or {}).get("label"),
            "world_label": None,
            "world_score": None,
            "iou": round(best_iou, 3),
        }
        if best_j is not None and best_iou >= iou_match:
            row["world_label"] = detections[best_j].label
            row["world_score"] = round(detections[best_j].score, 3)
            matched_det.add(best_j)
        rows.append(row)

    unmatched = [d.to_dict() for j, d in enumerate(detections) if j not in matched_det]
    return {
        "rows": rows,
        "unmatched_world": unmatched,
        "n_motion_boxes": len(motion),
        "n_world_detections": len(detections),
        "n_matched": len(matched_det),
    }


def _fit(text: str, width: int) -> str:
    """Truncate to a fixed column so one long verdict cannot shift the table."""
    return text if len(text) <= width else text[: width - 1] + "…"


def format_comparison(event: WaftEvent, report: dict) -> str:
    box_w, vit_w, world_w = 24, 30, 20
    lines = [
        f"\n{event.event_name}",
        f"  {report['n_motion_boxes']} motion box(es) -> vit_classifier",
        f"  {report['n_world_detections']} detection(s) -> YOLO-World",
        f"  {report['n_matched']} region(s) both systems agree on\n",
        f"  {'motion box':<{box_w}}{'ViT verdict':<{vit_w}}"
        f"{'YOLO-World':<{world_w}}{'IoU':>6}",
        "  " + "-" * (box_w + vit_w + world_w + 6),
    ]
    for r in report["rows"]:
        vit = r["vit_verdict"] or "-"
        if r["vit_after"]:
            vit = f"{vit} ({r['vit_after']})"
        world = "-"
        if r["world_label"]:
            world = f"{r['world_label']} ({r['world_score']:.2f})"
        lines.append(
            f"  {_fit(str(r['motion_box']), box_w - 1):<{box_w}}"
            f"{_fit(vit, vit_w - 1):<{vit_w}}"
            f"{_fit(world, world_w - 1):<{world_w}}{r['iou']:>6.2f}"
        )

    if report["unmatched_world"]:
        lines.append("\n  Found by YOLO-World in regions motion never flagged:")
        for d in report["unmatched_world"][:12]:
            lines.append(f"    {d['label']:<18}{d['score']:.2f}  {d['box_xyxy']}")
        lines.append(
            "\n  ^ these are what a single-stage pipeline would catch and the\n"
            "    two-stage one cannot: objects that are simply THERE, with no\n"
            "    motion to trigger on."
        )
    return "\n".join(lines)
