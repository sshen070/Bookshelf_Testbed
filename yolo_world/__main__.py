"""CLI: python -m yolo_world <command>

    vocab       list the prompt sets, show the active one
    detect      run on a single image
    detect-event  run on a WAFT event burst
    compare     side-by-side against vit_classifier on the same event
    bench       latency + energy, against the pipeline this would replace
    bake        save a checkpoint with the vocabulary embedded
    visual      detect from EXAMPLE CROPS instead of text prompts

Nothing here writes into WAFT's or vit_classifier's trees; results go where
--out says or to stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .accuracy import assign, format_report, load_truth, merge_external, score
from .compare import compare_regions, format_comparison, load_vit_result
from .config import detector_kwargs, events_dir, load_config
from .visual import VisualVocabError, load_spec
from .events import find_events, load_event
from .vocab import VocabError, active_name, available, expand, get


def _detector(cfg, vocab_name):
    from .detector import WorldDetector

    name, prompts, labels = get(cfg, vocab_name)
    det = WorldDetector(detector_kwargs(cfg))
    encode_s = det.set_vocabulary(prompts, labels)
    classes = sorted(set(labels))
    if classes == sorted(set(prompts)):
        print(f"vocabulary {name!r}: {', '.join(prompts)}")
    else:
        print(f"vocabulary {name!r}: {len(prompts)} prompt(s) -> "
              f"{len(classes)} class(es): {', '.join(classes)}")
    print(f"model {det.name} on {det.device} "
          f"(load {det.load_seconds:.1f}s, prompt encode {encode_s * 1e3:.0f} ms)")
    return det


def _imread(path):
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read {path}")
    return img


# ---- commands -----------------------------------------------------------


def cmd_vocab(args, cfg) -> int:
    act = active_name(cfg)
    sets = cfg.get("vocabulary", {}).get("sets") or {}
    for name in available(cfg):
        marker = "*" if name == act else " "
        prompts, labels = expand(sets[name], name)
        classes = sorted(set(labels))
        if classes == sorted(set(prompts)):
            print(f" {marker} {name:<14}{', '.join(prompts)}")
        else:
            print(f" {marker} {name:<14}{len(prompts)} prompts -> "
                  f"{', '.join(classes)}")
    print("\n* = active (vocabulary.active in config.yaml). Override with --vocab.")
    return 0


def cmd_detect(args, cfg) -> int:
    det = _detector(cfg, args.vocab)
    img = _imread(args.image)
    dets = det.detect(img)
    det.sync()

    print(f"\n{Path(args.image).name}  ({img.shape[1]}x{img.shape[0]})")
    if not dets:
        print("  nothing detected -- try lowering model.conf or rewording prompts")
    for d in dets:
        print(f"  {d.label:<18}{d.score:.3f}  {d.box_xyxy}")

    if args.out:
        payload = {"image": str(args.image), "vocabulary": det.prompts,
                   "detections": [d.to_dict() for d in dets]}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWritten to {args.out}")
    return 0


def _event_dirs(args, cfg) -> list[Path]:
    if args.event_dir:
        return [Path(args.event_dir)]
    found = find_events(args.events_dir or events_dir(cfg))
    if not found:
        print(f"No events under {args.events_dir or events_dir(cfg)}")
    return found


def cmd_detect_event(args, cfg) -> int:
    dirs = _event_dirs(args, cfg)
    if not dirs:
        return 1
    det = _detector(cfg, args.vocab)
    label = args.frame or cfg.get("events", {}).get("frame_label", "trigger")

    for event_dir in dirs:
        event = load_event(event_dir)
        frame = event.frame(label)
        if frame is None:
            print(f"\n{event.event_name}: no {label!r} frame")
            continue
        dets = det.detect(_imread(frame.path))
        det.sync()
        print(f"\n{event.event_name}  ({frame.label})")
        for d in dets[:args.top]:
            print(f"  {d.label:<18}{d.score:.3f}  {d.box_xyxy}")
        if not dets:
            print("  nothing detected")
    return 0


def cmd_compare(args, cfg) -> int:
    dirs = _event_dirs(args, cfg)
    if not dirs:
        return 1
    det = _detector(cfg, args.vocab)
    label = args.frame or cfg.get("events", {}).get("frame_label", "trigger")
    iou_match = float(cfg.get("compare", {}).get("iou_match", 0.3))
    vit_name = cfg.get("compare", {}).get("vit_result_name", "vit_result.json")

    for event_dir in dirs:
        event = load_event(event_dir)
        frame = event.frame(label)
        if frame is None:
            continue
        img = _imread(frame.path)
        dets = det.detect(img)
        det.sync()
        report = compare_regions(
            event, dets, (img.shape[1], img.shape[0]),
            load_vit_result(event_dir, vit_name), iou_match,
        )
        print(format_comparison(event, report))
    return 0


def cmd_bench(args, cfg) -> int:
    import time

    from .bench import format_table, run_workload

    dirs = _event_dirs(args, cfg)
    if not dirs:
        return 1
    event = load_event(dirs[0])
    frames = [_imread(f.path) for f in event.frames]
    print(f"{len(frames)} frame(s) from {event.event_name}, "
          f"{args.duration:.0f}s per workload\n")

    det = _detector(cfg, args.vocab)
    state = {"i": 0}

    def step():
        det.detect(frames[state["i"] % len(frames)])
        state["i"] += 1
        det.sync()

    rows = [run_workload("idle (no workload)", lambda: time.sleep(0.05), 6.0)]
    idle_mw = rows[0]["mw"]
    rows.append(run_workload(f"YOLO-World {det.imgsz}px", step, args.duration))
    print(format_table(rows, idle_mw))
    print("\nFor the pipeline this replaces, run WAFT's detector loop under the\n"
          "same sampler -- measured on this board it was 4.0 fps / 1287 mJ per\n"
          "frame, and it still needs a classification stage after it.")
    return 0


def cmd_accuracy(args, cfg) -> int:
    truth = load_truth(args.truth)
    det = _detector(cfg, args.vocab)
    img = _imread(args.frame)
    dets = det.detect(img)
    det.sync()

    rows = assign(truth, dets, args.match_iou)
    external = None
    other_name = None
    if args.other:
        with open(args.other, "r", encoding="utf-8") as f:
            payload = json.load(f)
        external = payload.get("predictions", payload)
        other_name = payload.get("name", "other") if isinstance(payload, dict) else "other"
        rows = merge_external(rows, external)

    stats = score(rows)
    print(f"\n{Path(args.frame).name}  {len(truth)} labelled region(s), "
          f"{len(dets)} detection(s)")
    print(format_report(rows, stats, other_name))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"frame": str(args.frame), "rows": rows, "stats": stats}, f, indent=2)
        print(f"\nWritten to {args.out}")
    return 0


def cmd_visual(args, cfg) -> int:
    from .visual import VisualDetector

    refs = load_spec(args.spec)
    kwargs = detector_kwargs(cfg)
    kwargs["visual_model"] = cfg.get("model", {}).get("visual_name", "yoloe-11s-seg.pt")
    det = VisualDetector(kwargs)
    report = det.build(refs, _imread)
    counts = ", ".join(f"{c}x{n}" for c, n in report["counts"].items())
    print(f"visual vocabulary: {counts}  "
          f"({report['images']} reference image(s), dim {report['dim']})")
    print(f"model {det.name} on {det.device}")

    img = _imread(args.frame)
    dets = det.detect(img)
    det.sync()

    if args.truth:
        rows = assign(load_truth(args.truth), dets, args.match_iou)
        stats = score(rows)
        print(f"\n{Path(args.frame).name}  {len(rows)} labelled region(s), "
              f"{len(dets)} detection(s)")
        print(format_report(rows, stats))
    else:
        print(f"\n{Path(args.frame).name}  {len(dets)} detection(s)")
        for d in dets[:args.top]:
            print(f"  {d.label:<18}{d.score:.3f}  {d.box_xyxy}")
    return 0


def cmd_bake(args, cfg) -> int:
    det = _detector(cfg, args.vocab)
    out = det.bake(args.out)
    print(f"\nBaked {len(det.prompts)} prompt(s) into {out}")
    print("Load it with ultralytics YOLO() -- no clip package, no text encoder,\n"
          "and no way to silently run a different vocabulary than this one.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m yolo_world",
                                     description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="override config.yaml")
    parser.add_argument("--vocab", default=None, help="prompt set name")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("vocab", help="list prompt sets")

    p = sub.add_parser("detect", help="run on one image")
    p.add_argument("--image", required=True)
    p.add_argument("--out", default=None)

    for name, helptext in (("detect-event", "run on a WAFT event burst"),
                           ("compare", "side-by-side against vit_classifier")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--event-dir", default=None)
        p.add_argument("--events-dir", default=None)
        p.add_argument("--frame", default=None, help="frame label, e.g. trigger/post_02")
        if name == "detect-event":
            p.add_argument("--top", default=10, type=int)

    p = sub.add_parser("bench", help="latency + energy")
    p.add_argument("--event-dir", default=None)
    p.add_argument("--events-dir", default=None)
    p.add_argument("--duration", default=20.0, type=float)

    p = sub.add_parser("accuracy", help="score against hand-labelled regions")
    p.add_argument("--frame", required=True)
    p.add_argument("--truth", required=True, help="JSON [[label,[x1,y1,x2,y2]],...]")
    p.add_argument("--other", default=None,
                   help="another system's predictions, to score side by side")
    p.add_argument("--match-iou", default=0.3, type=float)
    p.add_argument("--out", default=None)

    p = sub.add_parser("visual", help="detect from example crops, not text")
    p.add_argument("--spec", required=True,
                   help='JSON {"class": [[image, [x1,y1,x2,y2]], ...]}')
    p.add_argument("--frame", required=True)
    p.add_argument("--truth", default=None, help="score against labelled regions")
    p.add_argument("--match-iou", default=0.3, type=float)
    p.add_argument("--top", default=12, type=int)

    p = sub.add_parser("bake", help="embed the vocabulary into a checkpoint")
    p.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    handlers = {
        "vocab": cmd_vocab, "detect": cmd_detect, "detect-event": cmd_detect_event,
        "compare": cmd_compare, "bench": cmd_bench, "bake": cmd_bake,
        "accuracy": cmd_accuracy, "visual": cmd_visual,
    }
    try:
        return handlers[args.command](args, cfg)
    except (VocabError, VisualVocabError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
