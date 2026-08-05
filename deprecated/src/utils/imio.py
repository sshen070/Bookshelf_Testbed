"""Read any image format the rig might produce, including iPhone HEIC/HEIF,
returning a BGR uint8 array so the rest of the pipeline (all OpenCV-based)
doesn't need to know the source format.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    _HEIF_AVAILABLE = True
except ImportError:
    _HEIF_AVAILABLE = False

from PIL import Image

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".heic", ".heif")


def imread_any(path: str | Path) -> np.ndarray | None:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".heic", ".heif"):
        if not _HEIF_AVAILABLE:
            raise RuntimeError(
                f"{path} is HEIC/HEIF but pillow-heif isn't installed. "
                "Run: pip install pillow-heif"
            )
        try:
            with Image.open(path) as im:
                rgb = np.array(im.convert("RGB"))
        except Exception:
            return None
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return cv2.imread(str(path))
