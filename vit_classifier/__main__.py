"""CLI: python -m vit_classifier <command>

    extract-crops   dump crops from WAFT events, to hand-sort into class folders
    build-bank      embed the sorted exemplars and cache the bank
    eval-bank       leave-one-out over the bank -- are the classes separable?
    classify-event  classify a WAFT event burst -- before/after per box
    classify-image  classify boxes in a single image
    watch           live: classify each event as the detector finishes writing it

Bootstrap order matters: extract-crops -> hand-sort -> build-bank -> classify.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from .boxes import Box, crop, imread, load_boxes_json, scale_boxes
from .config import bank_paths, classifier_kwargs, load_config
from .waft_events import find_events, load_event
from .watch import (
    DEFAULT_INTERVAL,
    DEFAULT_WAFT_TIMEOUT,
    GATES,
    initial_handled,
    result_path,
    watch_events,
)


def _build_classifier(cfg: dict, ensure: bool = True):
    from .classifier import RoiClassifier

    clf = RoiClassifier(classifier_kwargs(cfg))
    if ensure:
        exemplar_dir, cache = bank_paths(cfg)
        clf.ensure_bank(exemplar_dir, cache)
    return clf


def _boxes_for_image(boxes: list[Box], processing_size, img) -> list[Box]:
    """Scale detector boxes into the coordinate space of the image being cropped."""
    h, w = img.shape[:2]
    if processing_size and tuple(processing_size) != (w, h):
        print(f"  scaling boxes from {tuple(processing_size)} to {(w, h)}")
        return scale_boxes(boxes, tuple(processing_size), (w, h))
    return boxes


# ---- extract-crops ------------------------------------------------------


def extract_crops(event_dirs: list[Path], out_dir: Path, cfg: dict) -> int:
    """Dump every box's crop from every frame of each burst, for hand-sorting.

    Crops from all frames, not just the transition pair: the pre-frames are
    where most `background` exemplars come from, and the spread across the
    burst gives the bank a few slightly different views of the same object,
    which is exactly what makes a k-NN robust.
    """
    crops_cfg = cfg.get("crops", {})
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for event_dir in event_dirs:
        event = load_event(event_dir)
        if not event.boxes:
            print(f"{event_dir.name}: no boxes recorded (grid-triggered event?), skipping.")
            continue
        for frame in event.frames:
            img = frame.load()
            if img is None:
                continue
            boxes = _boxes_for_image(event.boxes, event.processing_size, img)
            for i, box in enumerate(boxes):
                c = crop(
                    img,
                    box,
                    crops_cfg.get("pad_frac", 0.15),
                    crops_cfg.get("square", True),
                    crops_cfg.get("min_size_px", 8),
                )
                if c is None:
                    continue
                name = f"{event_dir.name}_box{i:02d}_{frame.label}.png"
                cv2.imwrite(str(out_dir / name), c)
                written += 1

    background = cfg.get("transitions", {}).get("background_class", "background")
    print(f"\nWrote {written} crop(s) to {out_dir}")
    print(
        "Now sort them into one folder per object, e.g.:\n"
        f"  {out_dir.parent / 'mug'}/\n"
        f"  {out_dir.parent / background}/   <- empty-region crops, mostly the pre_* ones\n"
        "then run: python -m vit_classifier build-bank"
    )
    return written


# ---- eval-bank ----------------------------------------------------------


def eval_bank(cfg: dict) -> None:
    """Leave-one-out over the cached bank, plus the class similarity matrix.

    Routes through `_build_classifier` rather than reading the .npz directly so
    a stale bank is rebuilt first -- evaluating embeddings that no longer match
    the exemplars on disk is worse than not evaluating at all.
    """
    from .eval_bank import class_similarity, format_report, leave_one_out

    clf = _build_classifier(cfg)
    if len(clf.labels) < 2:
        print(f"Bank holds {len(clf.labels)} exemplar(s) -- need at least 2 to leave one out.")
        return

    cases = leave_one_out(
        clf.embeddings, clf.labels, clf.exemplar_paths, clf.knn_k, clf.min_similarity
    )
    classes, matrix = class_similarity(clf.embeddings, clf.labels)
    print(format_report(cases, classes, matrix, clf.knn_k, clf.min_similarity))


# ---- classify -----------------------------------------------------------


def classify_event(event_dir: Path, cfg: dict, out_path: str | Path | None = None, clf=None) -> dict | None:
    """Classify one burst. Pass `clf` to reuse a loaded model across events.

    Building the classifier costs a model load plus a bank embed -- seconds, and
    the same seconds every time. Batch and live callers build one and hand it
    in; a single-event call leaves it None so an event with no boxes never pays
    for a model it isn't going to use.
    """
    event = load_event(event_dir)
    if not event.boxes:
        print(f"{event_dir.name}: no boxes recorded -- nothing to classify.")
        return None

    pair = event.pair_for_transition()
    if pair is None:
        print(f"{event_dir.name}: only {len(event.frames)} frame(s), cannot form a before/after pair.")
        return None
    before_frame, after_frame = pair

    before_img, after_img = before_frame.load(), after_frame.load()
    if before_img is None or after_img is None:
        print(f"{event_dir.name}: could not read the {before_frame.label}/{after_frame.label} frames.")
        return None

    clf = clf or _build_classifier(cfg)
    boxes = _boxes_for_image(event.boxes, event.processing_size, after_img)
    transitions = clf.classify_pair(before_img, after_img, boxes)

    print(f"\n{event.event_name or event_dir.name}  ({before_frame.label} -> {after_frame.label})")
    regions = []
    for box, tr in zip(boxes, transitions):
        regions.append({"box_xyxy": box.to_list(), **tr.to_dict()})
        print(
            f"  {box.to_list()}  {tr.before.label} ({tr.before.score:.2f})"
            f" -> {tr.after.label} ({tr.after.score:.2f})  = {tr.event}"
        )

    result = {
        "event_name": event.event_name or event_dir.name,
        "event_dir": str(event_dir),
        "before_frame": before_frame.label,
        "after_frame": after_frame.label,
        "regions": regions,
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Written to {out_path}")
    return result


def classify_image(image: str, boxes_json: str, cfg: dict, out_path: str | None = None) -> dict | None:
    img = imread(image)
    if img is None:
        raise FileNotFoundError(f"Could not read {image}")

    boxes, processing_size = load_boxes_json(boxes_json)
    if not boxes:
        print(f"{boxes_json} contains no boxes.")
        return None

    clf = _build_classifier(cfg)
    boxes = _boxes_for_image(boxes, processing_size, img)
    preds = clf.classify(img, boxes)

    print(f"\n{image}")
    regions = []
    for box, pred in zip(boxes, preds):
        regions.append({"box_xyxy": box.to_list(), **pred.to_dict()})
        print(f"  {box.to_list()}  {pred.label} ({pred.score:.2f})")

    result = {"image": str(image), "boxes_json": str(boxes_json), "regions": regions}
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Written to {out_path}")
    return result


# ---- watch --------------------------------------------------------------


def _write_skip_marker(out_path: Path, event_dir: Path, reason: str) -> None:
    """Record that a burst was looked at and had nothing classifiable in it.

    Without this, an event with no boxes (grid-triggered) or a single frame
    comes back on every `--backfill` restart, reprints its message, and is
    skipped again. It writes the same file a successful classification does, so
    "already handled" stays one existence check.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "event_name": event_dir.name,
                "event_dir": str(event_dir),
                "skipped": reason,
                "regions": [],
            },
            f,
            indent=2,
        )


def watch(
    events_dir: str | Path,
    cfg: dict,
    interval: float = DEFAULT_INTERVAL,
    wait_for_waft: str = "auto",
    waft_timeout: float = DEFAULT_WAFT_TIMEOUT,
    backfill: bool = False,
    reclassify: bool = False,
    results_dir: str | Path | None = None,
    max_events: int | None = None,
) -> int:
    """Classify WAFT events as they land, until interrupted."""
    events_dir = Path(events_dir)
    handled = initial_handled(events_dir, backfill, reclassify, results_dir)

    # A watcher is normally redirected to a log or piped through tee, where
    # Python block-buffers stdout and an operator sees nothing for minutes.
    # Line buffering makes every print below -- and every one in classify_event
    # -- land as it happens. Guarded: a wrapped stdout may not support it.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    print(f"\nWatching {events_dir} every {interval}s (waft gate: {wait_for_waft}).")
    if not events_dir.is_dir():
        print("  (doesn't exist yet -- waiting for the detector to create it)")
    if handled:
        note = "" if backfill else " -- pass --backfill to classify them too"
        print(f"Skipping {len(handled)} event(s) already on disk{note}.")

    # Load the model before the first event rather than on it. An event lands
    # about a second after its trigger; paying the model load and bank embed at
    # that point would put the first classification ten seconds behind the burst
    # it describes, and every later event would pay it again.
    clf = _build_classifier(cfg)
    print("Ready -- Ctrl-C to stop.")

    classified = 0

    def handle(event_dir: Path) -> None:
        nonlocal classified
        out_path = result_path(event_dir, results_dir)
        try:
            result = classify_event(event_dir, cfg, out_path, clf=clf)
        except Exception as exc:
            # One bad burst must not kill a watcher meant to run for a whole
            # session. Deliberately no skip marker: a failure here (a torn PNG,
            # an OOM behind a WAFT review) is usually transient, and marking it
            # would suppress the retry a later --backfill restart would do.
            print(f"{event_dir.name}: classification failed -- {exc!r}")
            return
        if result is None:
            _write_skip_marker(out_path, event_dir, "no boxes or no before/after pair")
            return
        classified += 1

    try:
        watch_events(
            events_dir,
            handle,
            interval=interval,
            handled=handled,
            wait_for_waft=wait_for_waft,
            waft_timeout=waft_timeout,
            max_events=max_events,
        )
    except KeyboardInterrupt:
        print()
    print(f"Stopped after classifying {classified} event(s).")
    return classified


# ---- entry point --------------------------------------------------------


def _resolve_event_dirs(args, parser) -> list[Path]:
    if args.event_dir:
        return [Path(args.event_dir)]
    if args.events_dir:
        found = find_events(args.events_dir)
        if not found:
            parser.error(f"No event directories with an event.json found under {args.events_dir}")
        return found
    parser.error("pass --event-dir for one event, or --events-dir for every event under a root")


def main() -> None:
    parser = argparse.ArgumentParser(prog="vit_classifier", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="Path to config.yaml (default: the one in this package).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract-crops", help="Dump WAFT event crops for hand-sorting.")
    p_extract.add_argument("--event-dir", help="A single event directory.")
    p_extract.add_argument("--events-dir", help="A root containing many event directories.")
    p_extract.add_argument("--out-dir", default=None, help="Default: <exemplar_dir>/_unsorted")

    sub.add_parser("build-bank", help="Embed exemplars/<class>/ and cache the bank.")

    sub.add_parser("eval-bank", help="Leave-one-out over the bank; are the classes separable?")

    p_event = sub.add_parser("classify-event", help="Classify a WAFT event burst (before/after per box).")
    p_event.add_argument("--event-dir")
    p_event.add_argument("--events-dir")
    p_event.add_argument("--out", help="Optional JSON output path (single event only).")

    p_image = sub.add_parser("classify-image", help="Classify boxes in a single image.")
    p_image.add_argument("--image", required=True)
    p_image.add_argument("--boxes-json", required=True, help="boxes.json or boxes_trigger.json.")
    p_image.add_argument("--out", help="Optional JSON output path.")

    p_watch = sub.add_parser(
        "watch",
        help="Live: classify each event as the detector finishes writing it.",
        description="Poll a live outputs/camera_events/events/ root and classify every "
                    "burst that reaches status=ready_for_waft. Results go to "
                    "<event_dir>/vit_result.json unless --results-dir is given.",
    )
    p_watch.add_argument("--events-dir", required=True,
                         help="WAFT's outputs/camera_events/events/ (may not exist yet).")
    p_watch.add_argument("--interval", type=float, default=DEFAULT_INTERVAL,
                         help=f"Seconds between polls (default: {DEFAULT_INTERVAL}).")
    p_watch.add_argument("--wait-for-waft", choices=GATES, default="auto",
                         help="auto: hold off only while a WAFT review is running (default). "
                              "always: require waft.status=complete. never: classify as soon "
                              "as the burst is written.")
    p_watch.add_argument("--waft-timeout", type=float, default=DEFAULT_WAFT_TIMEOUT,
                         help="Stop waiting on a 'running' review after this many seconds, so a "
                              f"dead subprocess can't strand the event (default: {DEFAULT_WAFT_TIMEOUT}; "
                              "0 waits indefinitely).")
    p_watch.add_argument("--backfill", action="store_true",
                         help="Also classify events already on disk at startup (skipping any that "
                              "already have a result -- this is how a restart catches up).")
    p_watch.add_argument("--reclassify", action="store_true",
                         help="With --backfill, redo events that already have a result file.")
    p_watch.add_argument("--results-dir",
                         help="Write <event_name>.json here instead of <event_dir>/vit_result.json.")
    p_watch.add_argument("--max-events", type=int, help="Exit after classifying this many events.")

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.command == "extract-crops":
        exemplar_dir, _ = bank_paths(cfg)
        out_dir = Path(args.out_dir) if args.out_dir else exemplar_dir / "_unsorted"
        extract_crops(_resolve_event_dirs(args, parser), out_dir, cfg)

    elif args.command == "build-bank":
        exemplar_dir, cache = bank_paths(cfg)
        clf = _build_classifier(cfg, ensure=False)
        clf.build_bank(exemplar_dir, cache)

    elif args.command == "eval-bank":
        eval_bank(cfg)

    elif args.command == "classify-event":
        event_dirs = _resolve_event_dirs(args, parser)
        if len(event_dirs) > 1 and args.out:
            parser.error("--out writes one file; use --event-dir for a single event.")
        # One model load for the whole batch. A single event keeps the lazy path
        # so `--event-dir` on a boxless event still costs nothing.
        clf = _build_classifier(cfg) if len(event_dirs) > 1 else None
        for event_dir in event_dirs:
            classify_event(event_dir, cfg, args.out, clf=clf)

    elif args.command == "classify-image":
        classify_image(args.image, args.boxes_json, cfg, args.out)

    elif args.command == "watch":
        watch(
            args.events_dir,
            cfg,
            interval=args.interval,
            wait_for_waft=args.wait_for_waft,
            waft_timeout=args.waft_timeout,
            backfill=args.backfill,
            reclassify=args.reclassify,
            results_dir=args.results_dir,
            max_events=args.max_events,
        )


if __name__ == "__main__":
    main()
