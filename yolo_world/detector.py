"""YOLO-World wrapper: open-vocabulary detection from text prompts.

ultralytics is imported lazily so that `config`, `vocab`, and `events` stay
importable on a machine with no model stack -- the same split vit_classifier
uses, and what lets the tests run without torch.

The prompt-then-detect contract in one paragraph: `set_classes(prompts)` runs
the prompts through a CLIP text encoder once and installs the resulting
embeddings as the detection head's class vectors. Inference afterwards costs
exactly what a closed-set YOLO costs -- the vocabulary lives in the head and
is not re-encoded per frame. Measured on the Orin Nano: the first
`set_classes` costs ~6.7 s (loading CLIP), later ones 0.26-0.41 s, and
detection itself is 34 ms/frame regardless. Changing what the system looks for
is seconds, not a training run.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Detection:
    label: str            # the class as reported (may be a vit_classifier name)
    score: float
    box_xyxy: list[int]
    prompt: str = ""      # the phrase that actually matched, when they differ

    def to_dict(self) -> dict:
        d = {"label": self.label, "score": round(self.score, 4),
             "box_xyxy": self.box_xyxy}
        if self.prompt and self.prompt != self.label:
            d["prompt"] = self.prompt
        return d


class WorldDetector:
    """Frozen YOLO-World + a text-prompt vocabulary. Never trained here."""

    def __init__(self, cfg: dict | None = None) -> None:
        cfg = cfg or {}
        try:
            import torch
            from ultralytics import YOLOWorld
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise RuntimeError(
                "YOLO-World needs torch + ultralytics.\n"
                "  pip install -r yolo_world/requirements.txt\n"
                "On a Jetson, install NVIDIA's CUDA torch wheel FIRST (see "
                "WAFT/WAFT/README_EVENT_DETECTION.md), then ultralytics with "
                "--no-deps, or pip will pull a CPU-only x86 torch over it."
            ) from exc

        self._torch = torch
        self.name = cfg.get("name", "yolov8s-worldv2.pt")
        self.imgsz = int(cfg.get("imgsz", 640))
        self.conf = float(cfg.get("conf", 0.05))
        self.iou = float(cfg.get("iou", 0.7))

        device = cfg.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        t0 = time.perf_counter()
        self.model = YOLOWorld(self.name)
        self.load_seconds = time.perf_counter() - t0
        self.prompts: list[str] = []

    def set_vocabulary(self, prompts: list[str],
                       labels: list[str] | None = None) -> float:
        """Install a vocabulary. Returns seconds spent encoding the prompts.

        The model is parked on the CPU for the duration. ultralytics builds the
        CLIP token tensor on the CPU unconditionally, so once the backbone has
        moved to CUDA -- which the first predict() does -- a second
        `set_classes` raises "Expected all tensors to be on the same device".
        That would make vocabulary switching a one-shot operation, which is
        exactly the capability this package exists to exercise. predict()
        places the model back on its device afterwards.
        """
        t0 = time.perf_counter()
        try:
            self.model.to("cpu")
        except Exception:  # noqa: BLE001 - placement is best-effort
            pass
        self.model.set_classes(list(prompts))
        self.prompts = list(prompts)
        self.labels = list(labels) if labels else list(prompts)
        if len(self.labels) != len(self.prompts):
            raise ValueError(
                f"{len(self.prompts)} prompt(s) but {len(self.labels)} label(s); "
                "they are matched positionally and must be the same length."
            )
        return time.perf_counter() - t0

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect on one BGR frame. Pass an ARRAY, not a path.

        Handing ultralytics a path makes it re-decode the file every call --
        measured at ~32 ms per 960x540 PNG on the Orin Nano, which is as much
        as the inference itself and turns a benchmark into a disk benchmark.
        """
        if not self.prompts:
            raise RuntimeError(
                "No vocabulary set -- call set_vocabulary() first, or the "
                "model will detect against whatever classes it shipped with."
            )
        result = self.model.predict(
            image, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
            device=self.device, verbose=False,
        )[0]
        out = []
        for cls, score, box in zip(result.boxes.cls, result.boxes.conf,
                                   result.boxes.xyxy):
            idx = int(cls)
            in_range = idx < len(self.prompts)
            out.append(Detection(
                label=self.labels[idx] if in_range else f"class_{idx}",
                score=float(score),
                box_xyxy=[int(v) for v in box.tolist()],
                prompt=self.prompts[idx] if in_range else "",
            ))
        out.sort(key=lambda d: -d.score)
        return out

    def sync(self) -> None:
        """Block until queued GPU work finishes -- required before timing."""
        if self.device == "cuda" and self._torch.cuda.is_available():
            self._torch.cuda.synchronize()

    def bake(self, out_path: str | Path) -> Path:
        """Save a checkpoint with the current vocabulary embedded.

        Removes the CLIP text encoder from the runtime path entirely, which
        matters on a remote node: the baked model needs no tokenizer, no
        `clip` package, and cannot silently re-encode a different vocabulary
        than the one an operator thinks is loaded. The trade is that changing
        vocabulary now means re-baking rather than editing config.
        """
        if not self.prompts:
            raise RuntimeError("Nothing to bake -- set a vocabulary first.")
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(out_path))
        return out_path
