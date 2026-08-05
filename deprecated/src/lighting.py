"""Lighting-drift robustness: reference-bank selection + local normalization.

Two complementary tactics:
1. Reference bank -- store several reference captures of the *unchanged*
   shelf under different lighting (morning/evening/overcast/lamp-on) and pick
   whichever one best matches the incoming frame's illumination before doing
   any pixel diffing. This avoids diffing a bright frame against a dark
   reference, which is the single biggest source of false positives.
2. Local normalization (CLAHE + gray-world white balance) applied to both
   frames post-alignment, so residual illumination differences within a
   frame (a shadow moving across the shelf) don't get read as structural
   change either.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.utils.imio import IMAGE_EXTS, imread_any


def gray_world_white_balance(img: np.ndarray) -> np.ndarray:
    result = img.astype(np.float32)
    for c in range(3):
        channel_mean = result[:, :, c].mean()
        if channel_mean > 1e-6:
            result[:, :, c] *= 128.0 / channel_mean
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_clahe(img: np.ndarray, clip_limit: float, tile_grid: tuple[int, int]) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tuple(tile_grid))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def normalize_illumination(img: np.ndarray, cfg: dict) -> np.ndarray:
    out = img
    if cfg.get("gray_world_white_balance", True):
        out = gray_world_white_balance(out)
    out = apply_clahe(out, cfg["clahe_clip_limit"], cfg["clahe_tile_grid"])
    return out


def _luminance_histogram(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
    return cv2.normalize(hist, hist).flatten()


def select_best_reference(
    frame: np.ndarray,
    reference_bank: list[tuple[str, np.ndarray]],
    cfg: dict,
    full_pipeline_cfg: dict | None = None,
    structural_shortlist_k: int | None = None,
) -> tuple[str, np.ndarray]:
    """Pick the reference image that best matches `frame`.

    `reference_bank` is a list of (name, image) pairs, e.g. every file in
    data/reference/. Falls back to the first entry if the bank has only one.

    Stage 1 (always): rank the whole bank by lighting-histogram similarity --
    cheap, and the main defense against diffing a bright frame against a dark
    reference.

    Stage 2 (only if `full_pipeline_cfg` is given): actually register+diff the
    top `structural_shortlist_k` lighting-shortlisted candidates against
    `frame` and pick whichever yields the LOWEST residual dissimilarity. This
    is what absorbs "object placed back a few mm off from its original spot"
    jitter -- if the bank already contains a confirmed-normal reference
    captured after a previous session's settling (see src/add_reference.py),
    it'll be preferred over the founding baseline for frames that resemble it,
    without the founding baseline ever being discarded. Skipped by default
    (full_pipeline_cfg=None) since it costs a real registration pass per
    shortlisted candidate -- pass it explicitly where that cost is acceptable.
    """
    if len(reference_bank) == 1:
        return reference_bank[0]

    metric = cfg.get("reference_bank_metric", "hist_correlation")
    frame_hist = _luminance_histogram(frame)
    frame_mean = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())

    scored = []
    for name, ref_img in reference_bank:
        if metric == "mean_luminance":
            ref_mean = float(cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY).mean())
            score = -abs(frame_mean - ref_mean)
        else:
            ref_hist = _luminance_histogram(ref_img)
            score = cv2.compareHist(frame_hist, ref_hist, cv2.HISTCMP_CORREL)
        scored.append((score, name, ref_img))
    scored.sort(key=lambda t: t[0], reverse=True)

    if full_pipeline_cfg is None:
        _, best_name, best_img = scored[0]
        return best_name, best_img

    from src.registration import align_to_reference
    from src.ssim_diff import compute_dissimilarity_map

    if structural_shortlist_k is None:
        structural_shortlist_k = cfg.get("structural_shortlist_k", 3)

    best_name, best_img, best_diss = None, None, np.inf
    for _, name, ref_img in scored[:structural_shortlist_k]:
        reg = align_to_reference(frame, ref_img, full_pipeline_cfg["registration"])
        if not reg.success:
            continue
        diss = float(compute_dissimilarity_map(reg.aligned, ref_img, full_pipeline_cfg).mean())
        if diss < best_diss:
            best_name, best_img, best_diss = name, ref_img, diss

    if best_name is None:  # every shortlisted candidate failed registration
        _, best_name, best_img = scored[0]
    return best_name, best_img


def load_reference_bank(reference_dir: str | Path) -> list[tuple[str, np.ndarray]]:
    reference_dir = Path(reference_dir)
    bank = []
    for path in sorted(reference_dir.glob("*")):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        img = imread_any(path)
        if img is not None:
            bank.append((path.stem, img))
    if not bank:
        raise FileNotFoundError(
            f"No reference images found in {reference_dir}. "
            "Capture at least one clean, unchanged shelf image per lighting "
            "condition and place it here before running the pipeline."
        )
    return bank
