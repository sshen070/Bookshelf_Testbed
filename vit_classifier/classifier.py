"""Frozen ViT + k-NN over a bank of exemplar crops.

WAFT answers *where* something moved. This answers *what* is there. The two
are deliberately separate processes: WAFT's detector is the always-on cheap
path, and nothing here should ever be in its per-frame hot loop.

Why a frozen backbone plus k-NN rather than a fine-tuned classification head:
the object vocabulary is small, known, and changes whenever someone puts a new
object in the scene. A fine-tuned head means a training run per vocabulary
change and a labeled dataset that doesn't exist. Here, adding a class means
dropping 5-10 crops into a new folder and re-running `build-bank`, which takes
seconds and needs no labels beyond the folder name. The trade is a few points
of accuracy against being maintainable by whoever is running the rig.

Two ways to use it:

  classify(img, boxes)                  -- what is in each box, one frame
  classify_pair(before, after, boxes)   -- what was there vs. what is there now

`classify_pair` is the useful one. A motion box brackets the region that
*changed between two frames*, so classifying only the later frame answers
"what is in the region now", which for an object that left the scene is
"empty background" -- true and useless. Cropping the SAME box from both frames
and diffing the two predictions recovers the actual event:

    mug -> background   = removed
    background -> hand  = placed
    mug -> bottle       = substituted

For a WAFT flow pair those two frames are frame1 and frame2. For a saved
camera event they are the pre and post frames of the burst (see waft_events.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .boxes import IMAGE_EXTS, Box, crop, imread

UNKNOWN = "unknown"


@dataclass
class RoiPrediction:
    """One region's classification. `score` is cosine similarity in [-1, 1]."""

    label: str
    score: float
    votes: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"label": self.label, "score": round(float(self.score), 4), "votes": self.votes}


@dataclass
class Transition:
    """The same box classified in two frames, plus the event word for the pair."""

    before: RoiPrediction
    after: RoiPrediction
    event: str

    def to_dict(self) -> dict:
        return {"before": self.before.to_dict(), "after": self.after.to_dict(), "event": self.event}


def describe_transition(before: RoiPrediction, after: RoiPrediction, background_class: str) -> str:
    """Reduce a (before, after) prediction pair to one event word.

    Module-level and dependency-free so the semantics are testable without
    instantiating a ViT.
    """
    if before.label == UNKNOWN or after.label == UNKNOWN:
        # Deliberately not guessing. An unknown on either side usually means a
        # genuinely new object -- a gap in the exemplar bank worth surfacing,
        # not something to paper over as "removed".
        return "uncertain"
    if before.label == after.label:
        # Same class both sides, yet the detector flagged motion here: the same
        # kind of object, moved -- or swapped for another of its kind. Class
        # labels alone cannot separate those two.
        return "moved_or_same_class"
    if before.label == background_class:
        return "placed"
    if after.label == background_class:
        return "removed"
    return "substituted"


def knn_vote(
    sims: np.ndarray, labels: list[str], knn_k: int, min_similarity: float
) -> RoiPrediction:
    """One k-NN vote over a row of similarities against the bank.

    Module-level and torch-free so `eval_bank` can score a held-out exemplar
    through exactly the code path inference uses. An evaluation that
    reimplements the vote is an evaluation of the wrong thing -- it would keep
    reporting the old behaviour after this one changed.
    """
    k = min(knn_k, sims.shape[0])
    top_idx = np.argpartition(-sims, k - 1)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    votes: dict[str, int] = {}
    for i in top_idx:
        votes[labels[i]] = votes.get(labels[i], 0) + 1

    # Rank by vote count, then by best similarity within the class, so a
    # 2-2 tie is broken by which neighbor was actually closer.
    def class_key(name: str) -> tuple[int, float]:
        best = max(sims[i] for i in top_idx if labels[i] == name)
        return votes[name], float(best)

    winner = max(votes, key=class_key)
    score = float(max(sims[i] for i in top_idx if labels[i] == winner))

    # Nothing in the bank is close enough. Saying UNKNOWN is the point:
    # an object the bank has never seen must not be forced into the
    # nearest known class just because it is nearest.
    label = winner if score >= min_similarity else UNKNOWN
    return RoiPrediction(label, score, votes)


def exemplar_fingerprint(exemplar_dir: Path) -> str:
    """Cheap signature of the exemplar folder, so a stale cache is detected.

    Path + size + mtime rather than hashing pixels: rebuilding the bank takes
    seconds, and a spurious rebuild costs far less than silently serving
    embeddings for exemplars that have since been edited or deleted.
    """
    exemplar_dir = Path(exemplar_dir)
    parts = []
    for p in sorted(exemplar_dir.rglob("*")):
        if p.suffix.lower() in IMAGE_EXTS:
            st = p.stat()
            parts.append(f"{p.relative_to(exemplar_dir).as_posix()}:{st.st_size}:{int(st.st_mtime)}")
    return "|".join(parts)


class RoiClassifier:
    """Frozen ViT embedder + k-NN over the exemplar bank.

    torch/timm are imported lazily in __init__ so the rest of this package
    (box parsing, cropping, WAFT event reading) stays importable without a
    model stack installed.
    """

    def __init__(self, cfg: dict | None = None) -> None:
        cfg = cfg or {}
        try:
            import timm
            import torch
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise RuntimeError(
                "The ViT classifier needs torch + timm.\n"
                "  pip install -r vit_classifier/requirements.txt\n"
                "On a Jetson, install NVIDIA's CUDA torch wheel first -- see WAFT's "
                "README_EVENT_DETECTION.md, then `pip install timm` on top."
            ) from exc

        self._torch = torch
        self.cfg = cfg
        self.model_name = cfg.get("name", "vit_small_patch14_dinov2.lvd142m")
        self.knn_k = int(cfg.get("knn_k", 5))
        self.min_similarity = float(cfg.get("min_similarity", 0.55))
        self.background_class = cfg.get("background_class", "background")
        self.crop_pad_frac = float(cfg.get("crop_pad_frac", 0.15))
        self.crop_square = bool(cfg.get("crop_square", True))
        self.crop_min_size_px = int(cfg.get("crop_min_size_px", 8))

        device = cfg.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        create_kwargs: dict = {"pretrained": True, "num_classes": 0}
        img_size = cfg.get("img_size")
        if img_size:
            # DINOv2 checkpoints are natively 518px, slow on CPU. timm can
            # interpolate the position embeddings down, but only with
            # dynamic_img_size on the ViTs that support it.
            create_kwargs["img_size"] = int(img_size)
            create_kwargs["dynamic_img_size"] = True

        self.model = timm.create_model(self.model_name, **create_kwargs)
        self.model.eval().to(self.device)

        data_config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**data_config, is_training=False)
        self.input_size = tuple(data_config["input_size"])

        self.labels: list[str] = []
        self.embeddings: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.exemplar_paths: list[str] = []

    @property
    def classes(self) -> list[str]:
        return sorted(set(self.labels))

    # ---- embedding -------------------------------------------------------

    def embed(self, crops_bgr: list[np.ndarray], batch_size: int = 16) -> np.ndarray:
        """Embed BGR crops into L2-normalized feature vectors, shape (N, D)."""
        import cv2
        from PIL import Image

        if not crops_bgr:
            return np.zeros((0, self.model.num_features), dtype=np.float32)

        tensors = [
            self.transform(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))
            for c in crops_bgr
        ]

        out = []
        with self._torch.no_grad():
            for i in range(0, len(tensors), batch_size):
                batch = self._torch.stack(tensors[i : i + batch_size]).to(self.device)
                feats = self.model(batch)
                feats = self._torch.nn.functional.normalize(feats, dim=1)
                out.append(feats.cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    # ---- exemplar bank ---------------------------------------------------

    def build_bank(self, exemplar_dir: str | Path, cache_path: str | Path | None = None) -> None:
        """Embed every image under exemplar_dir/<class_name>/ and optionally cache it."""
        exemplar_dir = Path(exemplar_dir)
        if not exemplar_dir.is_dir():
            raise FileNotFoundError(
                f"No exemplar directory at {exemplar_dir}. Create one folder per object class "
                "(e.g. exemplars/mug/, exemplars/background/) with a handful of cropped examples "
                "in each -- `python -m vit_classifier extract-crops` will dump candidates to sort."
            )

        crops, labels, paths = [], [], []
        for class_dir in sorted(p for p in exemplar_dir.iterdir() if p.is_dir()):
            if class_dir.name.startswith("_"):
                continue  # _unsorted/ and friends are staging areas, not classes
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                img = imread(img_path)
                if img is None:
                    print(f"WARNING: could not read exemplar {img_path}, skipping.")
                    continue
                crops.append(img)
                labels.append(class_dir.name)
                paths.append(img_path.relative_to(exemplar_dir).as_posix())

        if not crops:
            raise FileNotFoundError(
                f"{exemplar_dir} has no usable images in class subfolders. "
                "Expected layout: exemplars/<class_name>/<crop>.png"
            )

        self.embeddings = self.embed(crops)
        self.labels = labels
        self.exemplar_paths = paths

        print(f"Embedded {len(crops)} exemplars across {len(self.classes)} class(es) using {self.model_name}.")
        for name in self.classes:
            count = labels.count(name)
            flag = "  <- fewer than knn_k, this class can never win a full vote" if count < self.knn_k else ""
            print(f"  {name}: {count} exemplar(s){flag}")
        if self.background_class not in self.classes:
            print(
                f"NOTE: no '{self.background_class}' class in the bank, so removals cannot be "
                "detected -- every transition will read as 'substituted' or 'moved_or_same_class'. "
                "Add a folder of empty-region crops named after `transitions.background_class`."
            )

        if cache_path is not None:
            cache_path = Path(cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                cache_path,
                embeddings=self.embeddings,
                labels=np.array(self.labels),
                paths=np.array(self.exemplar_paths),
                model_name=self.model_name,
                input_size=np.array(self.input_size),
                fingerprint=exemplar_fingerprint(exemplar_dir),
            )
            print(f"Bank cached to {cache_path}")

    def load_bank(self, cache_path: str | Path, exemplar_dir: str | Path | None = None) -> bool:
        """Load a cached bank. Returns False if missing or stale (caller rebuilds)."""
        cache_path = Path(cache_path)
        if not cache_path.exists():
            return False
        data = np.load(cache_path, allow_pickle=False)

        if str(data["model_name"]) != self.model_name:
            print(f"Bank was built with {data['model_name']}, now using {self.model_name} -- rebuilding.")
            return False
        if tuple(data["input_size"]) != tuple(self.input_size):
            print("Bank was built at a different input size -- rebuilding.")
            return False
        if exemplar_dir is not None and str(data["fingerprint"]) != exemplar_fingerprint(Path(exemplar_dir)):
            print("Exemplar images changed since the bank was cached -- rebuilding.")
            return False

        self.embeddings = data["embeddings"]
        self.labels = [str(x) for x in data["labels"]]
        self.exemplar_paths = [str(x) for x in data["paths"]]
        return True

    def ensure_bank(self, exemplar_dir: str | Path, cache_path: str | Path) -> None:
        if not self.load_bank(cache_path, exemplar_dir):
            self.build_bank(exemplar_dir, cache_path)

    # ---- classification --------------------------------------------------

    def classify_crops(self, crops_bgr: list[np.ndarray]) -> list[RoiPrediction]:
        """k-NN vote over the exemplar bank, one prediction per crop."""
        if self.embeddings.shape[0] == 0:
            raise RuntimeError("Exemplar bank is empty -- call build_bank() or ensure_bank() first.")
        if not crops_bgr:
            return []

        feats = self.embed(crops_bgr)
        sims = feats @ self.embeddings.T  # cosine sim; both sides are L2-normalized
        return [knn_vote(row, self.labels, self.knn_k, self.min_similarity) for row in sims]

    def classify(self, img: np.ndarray, boxes: list[Box]) -> list[RoiPrediction]:
        """Classify each box's crop from one frame.

        Boxes too small or degenerate to crop get an UNKNOWN placeholder, so the
        returned list always lines up 1:1 with `boxes`.
        """
        crops, keep_idx = [], []
        for i, box in enumerate(boxes):
            c = crop(img, box, self.crop_pad_frac, self.crop_square, self.crop_min_size_px)
            if c is not None:
                crops.append(c)
                keep_idx.append(i)

        preds = self.classify_crops(crops)
        out = [RoiPrediction(UNKNOWN, 0.0) for _ in boxes]
        for slot, pred in zip(keep_idx, preds):
            out[slot] = pred
        return out

    def classify_pair(
        self, before_img: np.ndarray, after_img: np.ndarray, boxes: list[Box]
    ) -> list[Transition]:
        """Classify each box in BOTH frames and label the transition.

        The two frames must be the same size and the same viewpoint -- the same
        box coordinates are cropped from both, so a moved camera makes every
        transition meaningless. WAFT's frame pairs and event bursts satisfy this
        by construction (fixed camera); a handheld capture does not.
        """
        if before_img.shape[:2] != after_img.shape[:2]:
            raise ValueError(
                f"before/after frames differ in size ({before_img.shape[:2]} vs {after_img.shape[:2]}); "
                "the same box cannot address both."
            )
        before = self.classify(before_img, boxes)
        after = self.classify(after_img, boxes)
        return [
            Transition(b, a, describe_transition(b, a, self.background_class))
            for b, a in zip(before, after)
        ]
