"""Publish WAFT/ViT detection events to the server, so it can rank video feeds.

Runs **on the Jetson**, beside `status_client.py`. That one answers "how is this
node?"; this one answers "what did it just see?", and the server needs both to
decide which cameras are worth the link.

    python3 event_client.py --dry-run --events-dir <waft-events-dir>
    python3 event_client.py --host <broker> --site-id dam-01 --camera-id cam-north-03
    python3 event_client.py --http http://server:8000/events

Publishes to `devices/<node-id>/events`. `event_metadata.py` builds the
document; everything here is about *when* to send it.

## Two sends per event, and why

An event becomes complete (`status: ready_for_waft`) within a second of the
trigger. Its classification lands seconds later -- the ViT is a separate
process, holds the GPU, and may not be running at all.

Waiting for the label before telling the server anything would be the wrong
trade: the decision being made is whether to pull a *live* feed, and a feed
requested twenty seconds late has missed the thing it was requested for. So:

    revision 1   on burst completion    motion magnitudes, boxes, thumbnail
    revision 2   when the ViT lands     the same event plus labels

Both carry the same `event_id` (see `event_metadata.stable_id`), so the second
updates the first rather than reading as a second detection. A server acting on
revision 1 loses nothing by the update arriving later -- and if the classifier
never runs, revision 1 was always the whole story.

`--classify-wait` bounds the hope: after it, an unclassified event stops being
watched. Without that bound a node with no classifier would re-check every
event it had ever seen, forever.

## Startup sends nothing by default

Only events appearing *after* start are published, matching
`vit_classifier/watch.py`. A publisher restarted after an outage must not
replay a day of stale detections into the server's ranking as though a camera
had just lit up. `--backfill` is there when catching up is what you want.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

from event_metadata import (
    DEFAULT_THUMBNAIL_QUALITY,
    DEFAULT_THUMBNAIL_WIDTH,
    build_document,
)
from status_client import MqttSender, collect_status, send_http

# WAFT sets this once every post-frame is on disk. Publishing earlier means a
# burst whose frames the manifest names are not all written yet, so the
# thumbnail could be missing and the boxes could still change.
READY_STATUS = "ready_for_waft"

DEFAULT_POLL_INTERVAL = 1.0
# Long enough for a ViT review to finish on this rig (~20s, more when several
# queue up), short enough that a node with no classifier is not checking
# thousands of stale directories.
DEFAULT_CLASSIFY_WAIT = 60.0
# A full heartbeat reads a few dozen files under /proc and /sys. Events can
# arrive in bursts, and re-reading all of it per event would be the most
# expensive part of publishing one.
DEFAULT_HEALTH_INTERVAL = 30.0

# Ceiling on remembered event names. Each is ~60 bytes, so this is trivial
# memory, but a publisher running for months would otherwise grow without
# bound. Oldest are dropped first; they are long gone from disk by then.
MAX_TRACKED = 5000


def find_events(events_dir: Path) -> list[Path]:
    """Every event directory, oldest first.

    A missing root is not an error -- this is routinely started before the
    detector has created it. Ordered by mtime so a backlog is published in the
    order the events actually happened rather than alphabetically.
    """
    if not events_dir.is_dir():
        return []
    found = []
    for path in events_dir.iterdir():
        manifest = path / "event.json"
        if not path.is_dir() or not manifest.exists():
            continue
        try:
            found.append((manifest.stat().st_mtime, path))
        except OSError:
            continue  # deleted between listing and stat
    return [path for _, path in sorted(found, key=lambda item: item[0])]


def is_ready(event_dir: Path) -> bool:
    """Has WAFT finished writing this burst?

    Forgiving by design: the detector rewrites `event.json` in place with a
    truncating write, so a poll can land mid-write. Unreadable means "not yet",
    never "broken" -- the next poll is a second away.
    """
    try:
        with open(event_dir / "event.json", "r", encoding="utf-8") as f:
            event = json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    return isinstance(event, dict) and event.get("status") == READY_STATUS


def is_classified(event_dir: Path) -> bool:
    return (event_dir / "vit_result.json").exists()


class HealthCache:
    """The device health digest, re-read at most every `interval` seconds."""

    def __init__(self, device_id: str, interval: float = DEFAULT_HEALTH_INTERVAL):
        self.device_id = device_id
        self.interval = interval
        self._status: dict | None = None
        self._read_at = 0.0
        self._have_read = False
        self._warned = False

    def get(self) -> dict | None:
        now = time.monotonic()
        if self._have_read and now - self._read_at < self.interval:
            return self._status

        # Stamped before the attempt, not after a success. A box where health
        # cannot be read at all -- a non-Linux dev machine, a container with no
        # /proc -- would otherwise re-attempt on every single event, since the
        # result stays None and never looks cached.
        self._have_read = True
        self._read_at = now
        try:
            self._status = collect_status(self.device_id)
        except Exception as e:  # a health read must never cost an event
            self._status = None
            if not self._warned:
                # Once. The condition persists for the life of the process, and
                # repeating it per event would bury the detections in the log.
                print(f"health: cannot read device status ({e}). Events will be sent "
                      f"without device_health; nothing else is affected.", file=sys.stderr)
                self._warned = True
        return self._status


class EventPublisher:
    """Tracks which events have been sent, and at which revision."""

    def __init__(self, args, send):
        self.args = args
        self.send = send
        self.events_dir = Path(args.events_dir)
        self.health = HealthCache(args.node_id, args.health_interval)
        # name -> {"revision": int, "first_seen": monotonic}
        self.tracked: dict[str, dict] = {}
        self.sent_count = 0
        self.failed_count = 0

    def suppress_existing(self) -> int:
        """Mark what is already on disk as handled, so a start is not a flood."""
        existing = find_events(self.events_dir)
        now = time.monotonic()
        for path in existing:
            self.tracked[path.name] = {"revision": 2, "first_seen": now}
        return len(existing)

    def _prune(self) -> None:
        if len(self.tracked) <= MAX_TRACKED:
            return
        # dicts preserve insertion order, so the oldest entries are first.
        for name in list(self.tracked)[: len(self.tracked) - MAX_TRACKED]:
            del self.tracked[name]

    def poll(self) -> None:
        now = time.monotonic()
        for event_dir in find_events(self.events_dir):
            name = event_dir.name
            state = self.tracked.get(name)

            if state is not None and state["revision"] >= 2:
                continue  # fully published
            if not is_ready(event_dir):
                continue

            classified = is_classified(event_dir)
            want = 2 if classified else 1

            if state is None:
                state = self.tracked[name] = {"revision": 0, "first_seen": now}
                self._prune()

            if state["revision"] >= want:
                # Sent at this revision already. Stop watching once the wait for
                # a classifier that may never come has expired.
                if now - state["first_seen"] > self.args.classify_wait:
                    state["revision"] = 2
                continue

            if self.publish(event_dir, want):
                state["revision"] = want
            else:
                # Leave the revision alone so the next poll retries. A broker
                # that was briefly unreachable should not cost the event.
                self.failed_count += 1

    def publish(self, event_dir: Path, revision: int) -> bool:
        document = build_document(
            event_dir,
            node_id=self.args.node_id,
            site_id=self.args.site_id,
            camera_id=self.args.camera_id,
            device_status=self.health.get(),
            model_version=self.args.model_version,
            requested_seconds=self.args.clip_seconds,
            preroll_seconds=self.args.preroll_seconds,
            thumbnail_width=self.args.thumbnail_width,
            thumbnail_quality=self.args.thumbnail_quality,
            include_thumbnail=not self.args.no_thumbnail,
        )
        if document is None:
            print(f"{event_dir.name}: no readable event.json, skipping", file=sys.stderr)
            return True  # not retryable; don't spin on it

        detections = document["detections"]
        summary = ", ".join(f"{d['class']}@{d['confidence']:.2f}" for d in detections) or "no boxes"
        flow = document["motion"]["flow"]["dominant_score"]
        colour = document["motion"]["colour"]["dominant_score"]
        print(
            f"{event_dir.name} r{revision}: {document['event_type']} "
            f"[{summary}] flow={flow} colour={colour}",
            flush=True,
        )

        if self.send is None:
            printable = dict(document)
            if printable.get("thumbnail_b64"):
                printable["thumbnail_b64"] = f"<{len(printable['thumbnail_b64'])} b64 chars>"
            print(json.dumps(printable, indent=2))
            self.sent_count += 1
            return True

        if self.send(document):
            self.sent_count += 1
            return True
        return False


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Publish WAFT/ViT detection events to the status server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = p.add_argument_group("what to watch")
    source.add_argument(
        "--events-dir",
        default=os.environ.get("WAFT_EVENTS_DIR", "../WAFT/WAFT/outputs/camera_events/events"),
        help="WAFT's events directory (camera_event_detector.py writes here)",
    )
    source.add_argument("--interval", type=float, default=DEFAULT_POLL_INTERVAL, help="seconds between polls")
    source.add_argument("--backfill", action="store_true",
                        help="also publish events already on disk at startup")
    source.add_argument("--once", action="store_true",
                        help="publish what is ready now and exit; implies --backfill")
    source.add_argument("--classify-wait", type=float, default=DEFAULT_CLASSIFY_WAIT,
                        help="seconds to keep watching an event for its ViT result")

    identity = p.add_argument_group("identity")
    identity.add_argument("--node-id", default=os.environ.get("STATUS_DEVICE_ID") or socket.gethostname(),
                          help="this device; matches status_client.py's device_id")
    identity.add_argument("--site-id", default=os.environ.get("STATUS_SITE_ID", "site-01"),
                          help="the installation this node belongs to")
    identity.add_argument("--camera-id", default=None,
                          help="camera as video_link knows it; defaults to --node-id. "
                               "Must match sender.py's --camera-id, since the server acts on "
                               "the event by publishing to video/<camera-id>/trigger.")
    identity.add_argument("--model-version", default=os.environ.get("VIT_MODEL_VERSION"),
                          help="classifier build that produced the labels, recorded per event")
    identity.add_argument("--health-interval", type=float, default=DEFAULT_HEALTH_INTERVAL,
                          help="seconds between device-health reads; events in between reuse "
                               "the last one, since a full read touches dozens of /proc files")

    clip = p.add_argument_group("clip request")
    clip.add_argument("--clip-seconds", type=float, default=30.0, help="streaming window to request")
    clip.add_argument("--preroll-seconds", type=float, default=5.0, help="pre-trigger footage to request")

    thumb = p.add_argument_group("thumbnail")
    thumb.add_argument("--thumbnail-width", type=int, default=DEFAULT_THUMBNAIL_WIDTH)
    thumb.add_argument("--thumbnail-quality", type=int, default=DEFAULT_THUMBNAIL_QUALITY)
    thumb.add_argument("--no-thumbnail", action="store_true",
                       help="send numbers only; roughly 10x smaller per event")

    transport = p.add_argument_group("transport")
    transport.add_argument("--host", default=os.environ.get("STATUS_MQTT_HOST"), help="MQTT broker")
    transport.add_argument("--port", type=int, default=int(os.environ.get("STATUS_MQTT_PORT", 1883)))
    transport.add_argument("--topic", default=None, help="default: devices/<node-id>/events")
    transport.add_argument("--username", default=os.environ.get("STATUS_MQTT_USERNAME"))
    transport.add_argument("--password", default=os.environ.get("STATUS_MQTT_PASSWORD"),
                           help="prefer the env var; argv is visible to every process via ps")
    transport.add_argument("--http", metavar="URL", help="POST JSON here instead of using MQTT")
    transport.add_argument("--timeout", type=float, default=10.0, help="seconds to wait for an ack")
    transport.add_argument("--dry-run", action="store_true", help="print documents, contact nothing")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    topic = args.topic or f"devices/{args.node_id}/events"

    if not args.dry_run and not args.http and not args.host:
        print("Need one of --host (MQTT), --http URL, or --dry-run.", file=sys.stderr)
        return 2

    events_dir = Path(args.events_dir)
    if not events_dir.is_dir():
        # Not fatal: the detector may simply not have started yet, and this
        # process is meant to outlive individual detector runs.
        print(f"note: {events_dir} does not exist yet -- waiting for it.", file=sys.stderr)

    sender = None
    send = None
    if not args.dry_run:
        if args.http:
            send = lambda doc: send_http(args.http, doc, args.timeout)  # noqa: E731
        else:
            try:
                sender = MqttSender(args.host, args.port, topic, args.node_id,
                                    args.username, args.password, args.timeout,
                                    client_prefix="events")
            except ImportError:
                print("paho-mqtt is not installed. pip install -r requirements.txt (or use --http).",
                      file=sys.stderr)
                return 1
            except OSError as e:
                print(f"Could not connect to {args.host}:{args.port}: {e}", file=sys.stderr)
                if isinstance(e, ConnectionRefusedError):
                    print(
                        "Nothing is listening there. This publishes to an MQTT broker\n"
                        "(mosquitto) -- status_server.py is another client of the same\n"
                        "broker, not a broker itself. Start one (see mosquitto.conf.example)\n"
                        "or skip it with --http http://<server>:8000/events",
                        file=sys.stderr,
                    )
                return 1
            # retain=False: an event is an occurrence, not state. A retained one
            # would be replayed to every later subscriber as a fresh detection.
            send = lambda doc: sender.send(doc, retain=False, label="event")  # noqa: E731

    publisher = EventPublisher(args, send)
    if not (args.backfill or args.once):
        suppressed = publisher.suppress_existing()
        if suppressed:
            print(f"Ignoring {suppressed} event(s) already on disk (--backfill to publish them).")

    print(f"Watching {events_dir} -> {args.http or topic}", flush=True)

    try:
        while True:
            publisher.poll()
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if sender:
            sender.close()

    print(f"Published {publisher.sent_count} document(s), {publisher.failed_count} send failure(s).")
    return 1 if publisher.failed_count else 0


if __name__ == "__main__":
    sys.exit(main())