"""Align an incoming frame to a fixed reference frame.

Fixed-reference (not frame-to-frame) alignment is the deliberate design choice
here: it's what lets a slow, multi-week drift (a binder creeping, a crack
widening) still show up as a single large displacement from the anchor,
instead of vanishing into a string of imperceptible frame-to-frame deltas.

Strategy: try ECC (sub-pixel, intensity-based, great when the camera hasn't
moved and only lighting changed) first; fall back to ORB+RANSAC homography
(feature-based, tolerant of larger shifts/rotation, e.g. a bumped tripod).
If neither converges within tolerance, the frame is flagged rather than
silently compared against a misaligned reference.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

_ECC_MOTION = {
    "translation": cv2.MOTION_TRANSLATION,
    "euclidean": cv2.MOTION_EUCLIDEAN,
    "affine": cv2.MOTION_AFFINE,
    "homography": cv2.MOTION_HOMOGRAPHY,
}


@dataclass
class RegistrationResult:
    aligned: np.ndarray | None
    method: str  # "ecc" | "orb" | "failed"
    reprojection_error_px: float
    success: bool


def _to_gray_f32(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return gray.astype(np.float32) / 255.0


def _register_ecc(moving: np.ndarray, reference: np.ndarray, cfg: dict) -> tuple[np.ndarray | None, float]:
    motion_type = _ECC_MOTION[cfg["ecc_motion"]]
    warp_mode = motion_type
    warp_matrix = (
        np.eye(3, 3, dtype=np.float32) if warp_mode == cv2.MOTION_HOMOGRAPHY else np.eye(2, 3, dtype=np.float32)
    )
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        cfg["ecc_max_iterations"],
        cfg["ecc_eps"],
    )
    ref_gray = _to_gray_f32(reference)
    mov_gray = _to_gray_f32(moving)
    try:
        _, warp_matrix = cv2.findTransformECC(ref_gray, mov_gray, warp_matrix, warp_mode, criteria, None, 5)
    except cv2.error:
        return None, float("inf")

    h, w = reference.shape[:2]
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        aligned = cv2.warpPerspective(moving, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        aligned = cv2.warpAffine(moving, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    # ECC's criteria isn't a pixel-space error, so we can't report a real
    # reprojection error here; convergence itself (no exception above) is
    # the success signal, and the ORB path below is what supplies a proper
    # px error when we fall back to it.
    return aligned, 0.0


def _register_orb(moving: np.ndarray, reference: np.ndarray, cfg: dict) -> tuple[np.ndarray | None, float]:
    orb = cv2.ORB_create(nfeatures=cfg["orb_features"])
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) if reference.ndim == 3 else reference
    mov_gray = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY) if moving.ndim == 3 else moving

    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(mov_gray, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None, float("inf")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des2, des1, k=2)
    good = [m for m, n in matches if m.distance < cfg["orb_good_match_ratio"] * n.distance]
    if len(good) < cfg["min_inliers_for_homography"]:
        return None, float("inf")

    src_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return None, float("inf")

    inlier_mask = inlier_mask.ravel().astype(bool)
    if inlier_mask.sum() < cfg["min_inliers_for_homography"]:
        return None, float("inf")

    projected = cv2.perspectiveTransform(src_pts[inlier_mask], H)
    reproj_err = float(np.mean(np.linalg.norm(projected - dst_pts[inlier_mask], axis=2)))

    h, w = reference.shape[:2]
    aligned = cv2.warpPerspective(moving, H, (w, h), flags=cv2.INTER_LINEAR)
    return aligned, reproj_err


def align_to_reference(moving: np.ndarray, reference: np.ndarray, cfg: dict) -> RegistrationResult:
    """Align `moving` onto `reference`'s coordinate frame.

    `cfg` is the `registration` section of configs/pipeline.yaml.
    """
    if moving.shape[:2] != reference.shape[:2]:
        moving = cv2.resize(moving, (reference.shape[1], reference.shape[0]))

    if cfg["method"] == "ecc":
        aligned, err = _register_ecc(moving, reference, cfg)
        if aligned is not None:
            return RegistrationResult(aligned, "ecc", err, True)
        # ECC failed to converge -- try the more robust feature-based fallback.
        aligned, err = _register_orb(moving, reference, cfg)
        if aligned is not None and err <= cfg["max_reprojection_error_px"]:
            return RegistrationResult(aligned, "orb", err, True)
        return RegistrationResult(None, "failed", err, False)

    aligned, err = _register_orb(moving, reference, cfg)
    if aligned is not None and err <= cfg["max_reprojection_error_px"]:
        return RegistrationResult(aligned, "orb", err, True)
    return RegistrationResult(None, "failed", err, False)
