"""ViT-based ROI classification for WAFT motion detections.

WAFT says *where* something moved; this says *what* is there. Standalone by
design -- it imports nothing from the deprecated Bookshelf_Testbed pipeline
and is not wired into it.

Only `boxes`, `waft_events`, `config`, and `watch` are imported eagerly;
`classifier` pulls in torch/timm and is left to the caller, so box parsing, crop
extraction, and event watching work on a machine with no model stack.

    from vit_classifier import load_event, scale_boxes
    from vit_classifier.classifier import RoiClassifier   # needs torch + timm
"""
from .boxes import Box, crop, imread, load_boxes_json, parse_boxes_payload, scale_box, scale_boxes
from .config import bank_paths, classifier_kwargs, load_config
from .waft_events import EventFrame, WaftEvent, find_events, load_event, load_flow_pair
from .watch import EventState, find_ready_events, is_ready, read_event_state, watch_events

__all__ = [
    "Box",
    "EventFrame",
    "EventState",
    "WaftEvent",
    "bank_paths",
    "classifier_kwargs",
    "crop",
    "find_events",
    "find_ready_events",
    "imread",
    "is_ready",
    "load_boxes_json",
    "load_config",
    "load_event",
    "load_flow_pair",
    "parse_boxes_payload",
    "read_event_state",
    "scale_box",
    "scale_boxes",
    "watch_events",
]
