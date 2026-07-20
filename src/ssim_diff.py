"""Multi-scale SSIM dissimilarity map -> candidate change bounding boxes.

Two window sizes are combined: a large one for coarse, high-contrast changes
(a book toppled) and a small one for localized, subtle shifts (a binder
nudged a few mm) -- the latter is the setting that has to generalize to
hairline, low-contrast cracks on real dam imagery.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def iou(self, other: "BBox") -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = self.area() + other.area() - inter
        return inter / union if union > 0 else 0.0

    def to_yolo(self, img_w: int, img_h: int) -> tuple[float, float, float, float]:
        cx = (self.x1 + self.x2) / 2 / img_w
        cy = (self.y1 + self.y2) / 2 / img_h
        w = (self.x2 - self.x1) / img_w
        h = (self.y2 - self.y1) / img_h
        return cx, cy, w, h


def compute_dissimilarity_map(img_a: np.ndarray, img_b: np.ndarray, cfg: dict) -> np.ndarray:
    """Combine multi-scale (1 - SSIM) maps into a single [0, 1] dissimilarity map."""
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY) if img_a.ndim == 3 else img_a
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY) if img_b.ndim == 3 else img_b

    win_sizes = cfg["ssim"]["win_sizes"]
    win_weights = cfg["ssim"]["win_weights"]
    assert abs(sum(win_weights) - 1.0) < 1e-6, "ssim.win_weights must sum to 1.0"

    combined = np.zeros(gray_a.shape, dtype=np.float32)
    for win_size, weight in zip(win_sizes, win_weights):
        _, ssim_map = ssim(gray_a, gray_b, win_size=win_size, full=True)
        combined += weight * (1.0 - ssim_map).astype(np.float32)
    return np.clip(combined, 0.0, 1.0)


def diss_map_to_boxes(diss_map: np.ndarray, cfg: dict) -> tuple[list[BBox], np.ndarray]:
    """Threshold + clean up the dissimilarity map, return candidate boxes and the binary mask."""
    thresh = (diss_map >= cfg["ssim"]["diss_threshold"]).astype(np.uint8) * 255

    open_k = cfg["morphology"]["open_kernel"]
    close_k = cfg["morphology"]["close_kernel"]
    if open_k > 1:
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
    if close_k > 1:
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))

    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    h, w = diss_map.shape[:2]
    pad_frac = cfg["bbox"]["pad_frac"]
    min_area = cfg["morphology"]["min_component_area_px"]
    min_solidity = cfg["morphology"].get("min_solidity", 0.0)

    boxes = []
    for label in range(1, num_labels):  # skip background label 0
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x, y, bw, bh = (int(v) for v in stats[label, cv2.CC_STAT_LEFT: cv2.CC_STAT_LEFT + 4])

        if min_solidity > 0:
            # A real toppled/shifted object leaves a compact blob (high solidity =
            # contour area / convex-hull area). A registration/parallax artifact on
            # loose hanging wires leaves a thin, sprawling, tangled shape (low
            # solidity) -- this is what actually distinguishes the two, not area.
            component_crop = (labels_im[y : y + bh, x : x + bw] == label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(component_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                hull_area = cv2.contourArea(cv2.convexHull(contour))
                solidity = (cv2.contourArea(contour) / hull_area) if hull_area > 0 else 1.0
                if solidity < min_solidity:
                    continue

        pad_x, pad_y = int(bw * pad_frac), int(bh * pad_frac)
        boxes.append(BBox(
            x1=max(0, x - pad_x),
            y1=max(0, y - pad_y),
            x2=min(w, x + bw + pad_x),
            y2=min(h, y + bh + pad_y),
        ))

    boxes = _merge_overlapping(boxes, cfg["bbox"]["merge_iou_threshold"])
    max_boxes = cfg["bbox"]["max_boxes_per_image"]
    if len(boxes) > max_boxes:
        boxes = sorted(boxes, key=lambda b: b.area(), reverse=True)[:max_boxes]
    return boxes, thresh


def _merge_overlapping(boxes: list[BBox], iou_threshold: float) -> list[BBox]:
    merged: list[BBox] = []
    used = [False] * len(boxes)
    for i, box in enumerate(boxes):
        if used[i]:
            continue
        group = [box]
        used[i] = True
        for j in range(i + 1, len(boxes)):
            if not used[j] and box.iou(boxes[j]) >= iou_threshold:
                group.append(boxes[j])
                used[j] = True
        merged.append(BBox(
            x1=min(b.x1 for b in group),
            y1=min(b.y1 for b in group),
            x2=max(b.x2 for b in group),
            y2=max(b.y2 for b in group),
        ))
    return merged
