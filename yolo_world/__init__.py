"""Open-vocabulary detection for the bookshelf testbed.

WAFT answers *where something moved*. vit_classifier answers *what is there*,
given a region. This answers **both at once, from text prompts** -- and is
here to test whether that is a better trade than the two-stage pipeline.

Standalone: imports nothing from `WAFT/`, `vit_classifier/`, `video_link/`,
`device_status/`, or the deprecated `src/`. It reads WAFT's on-disk events and
vit_classifier's result files, and writes into neither.

`config`, `vocab`, `events`, `compare`, and `bench` import cleanly WITHOUT
torch or ultralytics -- only `detector` needs the model stack, so prompt
management, event parsing, and comparison logic all work on a bare machine
(and the tests run there).
"""
from .config import detector_kwargs, events_dir, load_config, resolve
from .events import EventFrame, WaftEvent, find_events, iou, load_event, scale_box
from .vocab import VocabError, available, expand, get, validate

__all__ = [
    "load_config", "detector_kwargs", "events_dir", "resolve",
    "load_event", "find_events", "WaftEvent", "EventFrame", "iou", "scale_box",
    "get", "available", "expand", "validate", "VocabError",
]
