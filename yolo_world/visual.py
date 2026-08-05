"""Visual prompts: teach the detector from example crops instead of words.

Text prompting has a hard floor — if CLIP has no concept for the object, no
wording reaches it. Measured on this testbed, the `plushie` was invisible to
`green stuffed animal toy`, `stuffed animal`, `teddy bear`, `green plush toy`
and `moss ball` alike: zero detections every time. With a visual prompt it
comes back at **0.89**.

The workflow is vit_classifier's, with localization added: point at an example
of the object, give it a name, and the detector finds more of them. One VPE
(visual prompt embedding) is extracted per reference region, averaged per
class, and installed as that class's detection vector — the direct analogue of
averaging an exemplar bank, except the result can also say *where*.

**Reference regions must come from the deployment domain.** This is not a
style note, it is the whole result. Measured on one shelf frame, 13 labelled
regions:

    references from scene frames    ->  3/3 held-out regions correct
    references from exemplar photos ->  1/13

The phone close-ups in `vit_classifier/exemplars/` transfer almost nothing:
YOLOE labelled the orange magazine files `books` at 0.91. Same domain gap that
costs the ViT its accuracy, reached by an independent route. Point this at
crops the detector itself produced.

Requires the YOLOE checkpoint, which is a different model from YOLO-World.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class VisualVocabError(ValueError):
    """Raised for a reference spec that would silently misbehave."""


@dataclass
class Reference:
    """One example of an object: where it is, and in which image."""

    label: str
    image: Path
    box: list[int]


def load_spec(path: str | Path) -> list[Reference]:
    """Read a reference spec.

        {
          "containers": [["frames/pre_01.png", [150, 155, 290, 300]], ...],
          "plushie":    [["frames/pre_01.png", [465, 405, 615, 490]]]
        }

    Image paths resolve relative to the spec file, so a spec can sit beside
    the frames it points at and stay portable.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or not raw:
        raise VisualVocabError(
            f"{path} should map class name -> list of [image, [x1,y1,x2,y2]]."
        )

    refs: list[Reference] = []
    for label, entries in raw.items():
        if " " in label:
            # ultralytics asserts on spaces; catching it here names the class.
            raise VisualVocabError(
                f"Class name {label!r} contains a space, which the model rejects. "
                "Use an underscore, e.g. 'green_box'."
            )
        if not isinstance(entries, list) or not entries:
            raise VisualVocabError(f"Class {label!r} has no reference regions.")
        for i, entry in enumerate(entries):
            if not isinstance(entry, list) or len(entry) != 2:
                raise VisualVocabError(
                    f"{label!r} reference {i} should be [image, [x1,y1,x2,y2]], "
                    f"got {entry!r}"
                )
            img, box = entry
            if len(box) != 4:
                raise VisualVocabError(
                    f"{label!r} reference {i} box needs 4 numbers, got {box!r}"
                )
            x1, y1, x2, y2 = (int(v) for v in box)
            if x2 <= x1 or y2 <= y1:
                raise VisualVocabError(
                    f"{label!r} reference {i} box {box!r} is empty or inverted."
                )
            resolved = Path(img)
            if not resolved.is_absolute():
                resolved = (path.parent / resolved).resolve()
            refs.append(Reference(str(label), resolved, [x1, y1, x2, y2]))
    return refs


def classes_of(refs: list[Reference]) -> list[str]:
    """Sorted class list. Order fixes the class indices the model reports."""
    return sorted({r.label for r in refs})


class VisualDetector:
    """YOLOE driven by visual prompts rather than text."""

    def __init__(self, cfg: dict | None = None) -> None:
        cfg = cfg or {}
        try:
            import torch
            from ultralytics import YOLOE
            from ultralytics.models.yolo.yoloe import YOLOEVPDetectPredictor
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise RuntimeError(
                "Visual prompts need torch + ultralytics with YOLOE.\n"
                "  pip install --no-deps ultralytics ultralytics-thop"
            ) from exc

        self._torch = torch
        self._VP = YOLOEVPDetectPredictor
        self.name = cfg.get("visual_model", "yoloe-11s-seg.pt")
        self.imgsz = int(cfg.get("imgsz", 640))
        self.conf = float(cfg.get("conf", 0.05))
        self.iou = float(cfg.get("iou", 0.7))
        device = cfg.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = YOLOE(self.name)
        self.classes: list[str] = []

    def _vpe(self, image: np.ndarray, box: list[int]):
        """Embed one reference region.

        A fresh predictor per reference is deliberate: the predictor caches the
        prompt tensor and its source shape, and reusing one across images of
        different sizes silently embeds the previous image's geometry.
        """
        predictor = self._VP(overrides={
            "task": "segment", "mode": "predict",
            "imgsz": self.imgsz, "verbose": False,
        })
        predictor.setup_model(self.model.model)
        predictor.set_prompts({
            "bboxes": np.array([box], dtype=np.float32),
            "cls": np.array([0]),
        })
        vpe = predictor.get_vpe(image)[0]
        return self._torch.nn.functional.normalize(vpe, dim=-1, p=2)

    def build(self, refs: list[Reference], loader) -> dict:
        """Install a vocabulary from reference regions. Returns a small report.

        `loader(path) -> BGR array` is injected so this module never depends on
        how images are read.
        """
        if not refs:
            raise VisualVocabError("No reference regions -- nothing to learn from.")
        torch = self._torch
        self.classes = classes_of(refs)

        per_class: dict[str, list] = {c: [] for c in self.classes}
        cache: dict[Path, np.ndarray] = {}
        for ref in refs:
            if ref.image not in cache:
                img = loader(ref.image)
                if img is None:
                    raise FileNotFoundError(f"Could not read reference image {ref.image}")
                cache[ref.image] = img
            per_class[ref.label].append(self._vpe(cache[ref.image], ref.box))

        vectors = []
        for c in self.classes:
            stacked = torch.stack(per_class[c]).mean(0)
            vectors.append(torch.nn.functional.normalize(stacked, dim=-1, p=2))

        embeddings = torch.cat(vectors, dim=0).unsqueeze(0)
        self.model.set_classes(self.classes, embeddings)
        # set_classes leaves a predictor bound to the old vocabulary behind.
        self.model.predictor = None
        return {
            "classes": self.classes,
            "counts": {c: len(per_class[c]) for c in self.classes},
            "images": len(cache),
            "dim": int(embeddings.shape[-1]),
        }

    def detect(self, image: np.ndarray):
        """Detect using the installed visual vocabulary."""
        from .detector import Detection

        if not self.classes:
            raise RuntimeError("No visual vocabulary installed -- call build() first.")
        result = self.model.predict(
            image, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
            device=self.device, verbose=False,
        )[0]
        out = []
        for cls, score, box in zip(result.boxes.cls, result.boxes.conf,
                                   result.boxes.xyxy):
            idx = int(cls)
            out.append(Detection(
                label=self.classes[idx] if idx < len(self.classes) else f"class_{idx}",
                score=float(score),
                box_xyxy=[int(v) for v in box.tolist()],
                prompt="(visual)",
            ))
        out.sort(key=lambda d: -d.score)
        return out

    def sync(self) -> None:
        if self.device == "cuda" and self._torch.cuda.is_available():
            self._torch.cuda.synchronize()
