"""Live watching: when is an event complete, and what does the poll loop pick up?

No torch, no timm, no camera -- these drive the same event.json lifecycle
camera_event_detector.py and waft_event_review.py write, by hand.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vit_classifier.watch import (
    DEFAULT_RESULT_NAME,
    EventState,
    event_dirs,
    find_ready_events,
    initial_handled,
    is_ready,
    read_event_state,
    result_path,
    watch_events,
)


def _write_event(
    events_dir: Path,
    name: str = "event_20260728_120934_133445_000001_frame_00000029",
    status: str = "ready_for_waft",
    waft_status: str = "not_started",
    started_at: str | None = None,
) -> Path:
    """One event directory in whatever lifecycle state the test needs."""
    event_dir = events_dir / name
    event_dir.mkdir(parents=True, exist_ok=True)
    waft: dict = {"status": waft_status, "result_path": ""}
    if started_at is not None:
        waft["started_at"] = started_at
    (event_dir / "event.json").write_text(
        json.dumps({"event_id": 1, "event_name": name, "status": status, "waft": waft})
    )
    return event_dir


# ---- reading the manifest -----------------------------------------------


def test_read_event_state(tmp_path):
    state = read_event_state(_write_event(tmp_path, waft_status="running", started_at="2026-07-28T12:09:35"))
    assert state.readable
    assert state.status == "ready_for_waft"
    assert state.waft_status == "running"
    assert state.waft_started_at == "2026-07-28T12:09:35"


def test_read_event_state_survives_a_torn_write(tmp_path):
    """Both producers rewrite event.json in place, so a poll can land mid-write."""
    event_dir = _write_event(tmp_path)
    full = (event_dir / "event.json").read_text()
    (event_dir / "event.json").write_text(full[: len(full) // 2])

    state = read_event_state(event_dir)
    assert not state.readable
    assert not is_ready(state)


def test_read_event_state_on_missing_manifest(tmp_path):
    assert read_event_state(tmp_path / "nope").readable is False


# ---- the completion gate ------------------------------------------------


def test_burst_still_collecting_post_frames_is_not_ready(tmp_path):
    """The whole point: pre-vs-trigger would report on a placement still mid-air."""
    state = read_event_state(_write_event(tmp_path, status="pending_post_frames"))
    assert not is_ready(state, "auto")
    assert not is_ready(state, "never")
    assert not is_ready(state, "always")


def test_auto_gate_classifies_when_no_review_was_started(tmp_path):
    """--auto-waft off: waft.status never leaves not_started, so nothing is coming."""
    state = read_event_state(_write_event(tmp_path, waft_status="not_started"))
    assert is_ready(state, "auto")


def test_auto_gate_waits_out_a_running_review(tmp_path):
    now = datetime(2026, 7, 28, 12, 9, 40)
    state = read_event_state(
        _write_event(tmp_path, waft_status="running", started_at="2026-07-28T12:09:35")
    )
    assert not is_ready(state, "auto", waft_timeout=180.0, now=now)
    assert is_ready(state, "never", now=now)


def test_auto_gate_gives_up_on_a_review_that_never_finished(tmp_path):
    """A review process that died without updating event.json must not strand the event."""
    started = "2026-07-28T12:09:35"
    state = read_event_state(_write_event(tmp_path, waft_status="running", started_at=started))
    later = datetime.fromisoformat(started) + timedelta(seconds=181)
    assert is_ready(state, "auto", waft_timeout=180.0, now=later)


def test_zero_timeout_waits_indefinitely(tmp_path):
    state = read_event_state(_write_event(tmp_path, waft_status="running", started_at="2026-07-28T12:09:35"))
    later = datetime.fromisoformat("2026-07-28T12:09:35") + timedelta(days=1)
    assert not is_ready(state, "auto", waft_timeout=0.0, now=later)


def test_unparseable_start_time_expires_rather_than_blocks(tmp_path):
    """Failing open: an untimeable review would otherwise never be classified."""
    state = read_event_state(_write_event(tmp_path, waft_status="running", started_at="garbage"))
    assert is_ready(state, "auto")


def test_running_review_with_no_start_time_expires(tmp_path):
    state = read_event_state(_write_event(tmp_path, waft_status="running"))
    assert is_ready(state, "auto")


def test_auto_gate_does_not_wait_on_a_failed_review(tmp_path):
    state = read_event_state(_write_event(tmp_path, waft_status="failed"))
    assert is_ready(state, "auto")
    assert not is_ready(state, "always")


def test_always_gate_needs_a_complete_review(tmp_path):
    assert is_ready(read_event_state(_write_event(tmp_path, waft_status="complete")), "always")
    assert not is_ready(read_event_state(_write_event(tmp_path, waft_status="running")), "always")


def test_unknown_gate_is_rejected():
    with pytest.raises(ValueError):
        is_ready(EventState(status="ready_for_waft", readable=True), "sometimes")


# ---- scanning the events root -------------------------------------------


def test_event_dirs_tolerates_a_missing_root(tmp_path):
    """The watcher is often started before the detector has made the directory."""
    assert event_dirs(tmp_path / "not_yet") == []


def test_event_dirs_ignores_a_directory_without_a_manifest(tmp_path):
    (tmp_path / "event_half_made" / "frames").mkdir(parents=True)
    _write_event(tmp_path, name="event_real")
    assert [p.name for p in event_dirs(tmp_path)] == ["event_real"]


def test_find_ready_events_skips_incomplete_and_handled(tmp_path):
    _write_event(tmp_path, name="event_a")
    _write_event(tmp_path, name="event_b", status="pending_post_frames")
    _write_event(tmp_path, name="event_c")
    found = find_ready_events(tmp_path, handled={"event_c"})
    assert [p.name for p in found] == ["event_a"]


def test_find_ready_events_orders_by_trigger_time(tmp_path):
    """Oldest burst first, so a backlog replays in the order things happened."""
    first = _write_event(tmp_path, name="event_zzz_older")
    second = _write_event(tmp_path, name="event_aaa_newer")
    os.utime(first / "event.json", (1_000_000, 1_000_000))
    os.utime(second / "event.json", (2_000_000, 2_000_000))
    assert [p.name for p in find_ready_events(tmp_path)] == ["event_zzz_older", "event_aaa_newer"]


# ---- result paths and startup state -------------------------------------


def test_result_path_defaults_beside_the_event(tmp_path):
    event_dir = _write_event(tmp_path)
    assert result_path(event_dir) == event_dir / DEFAULT_RESULT_NAME
    assert result_path(event_dir, tmp_path / "out") == tmp_path / "out" / f"{event_dir.name}.json"


def test_initial_handled_suppresses_history_by_default(tmp_path):
    _write_event(tmp_path, name="event_a")
    _write_event(tmp_path, name="event_b")
    assert initial_handled(tmp_path) == {"event_a", "event_b"}


def test_backfill_only_suppresses_already_classified_events(tmp_path):
    done = _write_event(tmp_path, name="event_done")
    _write_event(tmp_path, name="event_todo")
    (done / DEFAULT_RESULT_NAME).write_text("{}")

    assert initial_handled(tmp_path, backfill=True) == {"event_done"}
    assert initial_handled(tmp_path, backfill=True, reclassify=True) == set()


def test_backfill_honours_a_results_dir(tmp_path):
    events, results = tmp_path / "events", tmp_path / "results"
    results.mkdir()
    _write_event(events, name="event_done")
    _write_event(events, name="event_todo")
    (results / "event_done.json").write_text("{}")
    assert initial_handled(events, backfill=True, results_dir=results) == {"event_done"}


# ---- the poll loop ------------------------------------------------------


def test_watch_events_picks_up_an_event_that_completes_mid_run(tmp_path):
    """The live case: the directory exists while the detector is still writing it."""
    _write_event(tmp_path, name="event_a", status="pending_post_frames")
    seen: list[str] = []

    def on_event(event_dir: Path) -> None:
        seen.append(event_dir.name)

    # Nothing is ready yet, so the first two polls find nothing.
    assert watch_events(tmp_path, on_event, interval=0.0, max_polls=2) == 0
    assert seen == []

    _write_event(tmp_path, name="event_a", status="ready_for_waft")
    assert watch_events(tmp_path, on_event, interval=0.0, max_events=1) == 1
    assert seen == ["event_a"]


def test_watch_events_respects_the_handled_set(tmp_path):
    _write_event(tmp_path, name="event_old")
    _write_event(tmp_path, name="event_new")
    seen: list[str] = []
    watch_events(tmp_path, seen.append, interval=0.0, handled={"event_old"}, max_events=1)
    assert [Path(p).name for p in seen] == ["event_new"]


def test_watch_events_does_not_repeat_an_event(tmp_path):
    _write_event(tmp_path, name="event_a")
    seen: list[Path] = []
    assert watch_events(tmp_path, seen.append, interval=0.0, max_polls=3) == 1
    assert len(seen) == 1


def test_watch_events_lets_a_handler_error_propagate(tmp_path):
    """The loop doesn't swallow errors; the CLI handler is what keeps a session alive."""
    _write_event(tmp_path, name="event_a")
    calls = []

    def explode(event_dir: Path) -> None:
        calls.append(event_dir.name)
        raise RuntimeError("bank is empty")

    with pytest.raises(RuntimeError):
        watch_events(tmp_path, explode, interval=0.0, max_polls=3)
    assert calls == ["event_a"]


def test_watch_events_survives_a_handler_that_catches_its_own_errors(tmp_path):
    """What the CLI actually does: log the bad burst, keep watching, don't retry it."""
    _write_event(tmp_path, name="event_a")
    _write_event(tmp_path, name="event_b")
    calls = []

    def handle(event_dir: Path) -> None:
        calls.append(event_dir.name)
        try:
            raise RuntimeError("torn png")
        except RuntimeError:
            pass

    assert watch_events(tmp_path, handle, interval=0.0, max_polls=3) == 2
    assert sorted(calls) == ["event_a", "event_b"]


def test_skip_marker_stops_a_boxless_event_coming_back(tmp_path):
    """A grid-triggered burst has nothing to classify; the marker records that once."""
    from vit_classifier.__main__ import _write_skip_marker

    event_dir = _write_event(tmp_path, name="event_a")
    out = result_path(event_dir)
    _write_skip_marker(out, event_dir, "no boxes or no before/after pair")

    assert json.loads(out.read_text())["skipped"] == "no boxes or no before/after pair"
    assert initial_handled(tmp_path, backfill=True) == {"event_a"}
