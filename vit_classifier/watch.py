"""Live mode: watch WAFT's events directory and classify each burst as it lands.

`classify-event` walks a directory of finished events. This walks the same
directory *while camera_event_detector.py is still writing into it*, which needs
a rule for when an event is safe to read -- an event directory exists on disk
long before it is complete.

WAFT's event lifecycle, both fields in `event.json`:

    status       pending_post_frames -> ready_for_waft
    waft.status  not_started -> running -> complete | failed

`status == "ready_for_waft"` is the completion signal that matters here. The
detector sets it only after the last post-frame has been copied in, so by the
time it appears every frame the manifest names is already on disk. Classify
anything earlier and the burst has no post-frame yet: `pair_for_transition()`
would compare pre against trigger and report on a placement that is still
mid-air.

The second field is about *contention*, not data. Nothing in this package reads
`waft_results/` -- the frames and boxes both come from the detector, and are
final before WAFT review starts. But with `--auto-waft` on, that review holds
the GPU for ~20s immediately after the event lands, and on the Jetson's shared
8GB that is the wrong moment to be running a ViT as well. Hence the default
`auto` gate: wait out a review that is actually running, don't wait for one that
was never started.

Reads are deliberately forgiving. Both producers rewrite `event.json` in place
with a plain truncating write, so a poll can land mid-write; an unreadable or
half-written manifest means "not yet", never "broken", and the next poll is a
second away.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

# The event.json values this module keys off. They are WAFT's, not ours --
# camera_event_detector.py writes the first, waft_event_review.py the rest.
READY_STATUS = "ready_for_waft"
WAFT_RUNNING = "running"
WAFT_COMPLETE = "complete"

# How to treat an event whose WAFT review has not finished:
#   auto    -- classify unless a review is currently running (default)
#   always  -- classify only after waft.status == "complete"
#   never   -- classify as soon as the burst is complete, review or not
GATES = ("auto", "always", "never")

DEFAULT_INTERVAL = 1.0
# Long enough to sit out a normal review (~20s on this rig, more when several
# queue up behind --waft-max-active), short enough that a review whose process
# died without updating event.json doesn't strand the event forever.
DEFAULT_WAFT_TIMEOUT = 180.0

DEFAULT_RESULT_NAME = "vit_result.json"


@dataclass
class EventState:
    """The two lifecycle fields of an `event.json`, or empty if unreadable."""

    status: str = ""
    waft_status: str = ""
    waft_started_at: str | None = None
    readable: bool = False


def read_event_state(event_dir: str | Path) -> EventState:
    """Read the lifecycle fields out of `<event_dir>/event.json`.

    Never raises: a missing, half-written, or malformed manifest comes back as
    an empty (unreadable) state, which no gate treats as ready.
    """
    path = Path(event_dir) / "event.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            event = json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return EventState()
    if not isinstance(event, dict):
        return EventState()

    waft = event.get("waft")
    waft = waft if isinstance(waft, dict) else {}
    return EventState(
        status=event.get("status", ""),
        waft_status=waft.get("status", ""),
        waft_started_at=waft.get("started_at"),
        readable=True,
    )


def _seconds_since(timestamp: str | None, now: datetime | None = None) -> float:
    """Age of an event.json ISO timestamp in seconds, or inf if unreadable.

    inf rather than 0 so an unparseable start time *expires* the wait instead of
    extending it. A review that can never time out is a burst that never gets
    classified, which is worse than one classified while the GPU is busy.
    """
    if not timestamp:
        return float("inf")
    try:
        started = datetime.fromisoformat(timestamp)
        return ((now or datetime.now()) - started).total_seconds()
    except (ValueError, TypeError):
        return float("inf")


def is_ready(
    state: EventState,
    wait_for_waft: str = "auto",
    waft_timeout: float = DEFAULT_WAFT_TIMEOUT,
    now: datetime | None = None,
) -> bool:
    """Is this event complete enough to classify under the given gate?"""
    if wait_for_waft not in GATES:
        raise ValueError(f"wait_for_waft must be one of {GATES}, got {wait_for_waft!r}")

    if state.status != READY_STATUS:
        # Still collecting post-frames (or the manifest was unreadable).
        return False
    if wait_for_waft == "never":
        return True
    if state.waft_status == WAFT_COMPLETE:
        return True
    if wait_for_waft == "always":
        return False
    if state.waft_status != WAFT_RUNNING:
        # not_started (auto-waft off, or the queue was full) or failed --
        # nothing is coming, so there is nothing to wait for.
        return True
    if waft_timeout <= 0:
        return False
    return _seconds_since(state.waft_started_at, now) > waft_timeout


def event_dirs(events_dir: str | Path) -> list[Path]:
    """Every directory under `events_dir` that has an event.json.

    Unlike `waft_events.find_events`, a missing root is not an error: the
    watcher is often started before the detector has created it.
    """
    events_dir = Path(events_dir)
    if not events_dir.is_dir():
        return []
    return sorted(p for p in events_dir.iterdir() if p.is_dir() and (p / "event.json").exists())


def result_path(event_dir: str | Path, results_dir: str | Path | None = None) -> Path:
    """Where this event's classification lands.

    Beside the event by default, so the result travels with the burst it
    describes and doubles as the "already done" marker across restarts.
    """
    event_dir = Path(event_dir)
    if results_dir is None:
        return event_dir / DEFAULT_RESULT_NAME
    return Path(results_dir) / f"{event_dir.name}.json"


def initial_handled(
    events_dir: str | Path,
    backfill: bool = False,
    reclassify: bool = False,
    results_dir: str | Path | None = None,
) -> set[str]:
    """Event names to suppress at startup.

    Default is everything already on disk: a live watcher is for what happens
    next, and an operator who wants the history has `classify-event`. With
    `backfill`, only events that already carry a result are suppressed, which is
    what makes a restart pick up exactly what it missed while it was down.
    """
    handled = set()
    for path in event_dirs(events_dir):
        if not backfill:
            handled.add(path.name)
        elif not reclassify and result_path(path, results_dir).exists():
            handled.add(path.name)
    return handled


def find_ready_events(
    events_dir: str | Path,
    handled: Iterable[str] = (),
    wait_for_waft: str = "auto",
    waft_timeout: float = DEFAULT_WAFT_TIMEOUT,
    now: datetime | None = None,
) -> list[Path]:
    """Complete, not-yet-handled events, oldest trigger first.

    Ordered by event.json mtime so a backlog is worked through in the order the
    bursts actually happened, not alphabetically by name.
    """
    handled = set(handled)
    ready = []
    for path in event_dirs(events_dir):
        if path.name in handled:
            continue
        if not is_ready(read_event_state(path), wait_for_waft, waft_timeout, now):
            continue
        try:
            mtime = (path / "event.json").stat().st_mtime
        except OSError:
            continue  # deleted between listing and stat
        ready.append((mtime, path))
    return [path for _, path in sorted(ready, key=lambda item: item[0])]


def watch_events(
    events_dir: str | Path,
    on_event: Callable[[Path], None],
    interval: float = DEFAULT_INTERVAL,
    handled: Iterable[str] | None = None,
    wait_for_waft: str = "auto",
    waft_timeout: float = DEFAULT_WAFT_TIMEOUT,
    max_events: int | None = None,
    max_polls: int | None = None,
) -> int:
    """Poll `events_dir` forever, calling `on_event` once per completed event.

    Returns the number of events handled. Polling rather than inotify: it costs
    one `scandir` plus a small read per unfinished event per second, it needs no
    extra dependency on the Jetson, and it behaves the same over a network mount
    where inotify silently reports nothing.

    An event is marked handled before `on_event` runs, so no burst can be
    classified twice. Handler exceptions propagate -- the CLI wraps its own
    handler so one bad burst doesn't end the session, and a caller that wants
    the loop to die on error gets that by not wrapping. `max_polls` exists so
    tests can bound the loop; production callers leave it None.
    """
    seen = set(handled or ())
    count = 0
    polls = 0
    while True:
        for event_dir in find_ready_events(events_dir, seen, wait_for_waft, waft_timeout):
            seen.add(event_dir.name)
            on_event(event_dir)
            count += 1
            if max_events is not None and count >= max_events:
                return count
        polls += 1
        if max_polls is not None and polls >= max_polls:
            return count
        time.sleep(interval)
