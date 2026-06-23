"""Bead and hole detection.

Given a calibrated photo we need to find every bead and, inside each one, its
central hole.  We rely on a connected-component / contour-hierarchy approach:
the outer contour of a blob is the bead, and a contour nested inside it is the
hole.  Where the hole is too faint to segment we fall back to a Hough-circle
search within the bead ROI.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BeadCandidate:
    outer_contour: np.ndarray            # (N, 2) outer edge points.
    hole_contour: np.ndarray | None      # (M, 2) inner edge points, or None.
    hole_circle: tuple[float, float, float] | None  # (cx, cy, r) Hough fallback.
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    outer_area: float


def _rect_overlap(a, b) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def segment_beads(
    image_bgr: np.ndarray,
    *,
    pixels_per_mm: float,
    min_diameter_mm: float = 1.0,
    max_diameter_mm: float = 8.0,
    min_circularity: float = 0.6,
    exclude_roi: tuple[int, int, int, int] | None = None,
) -> list[BeadCandidate]:
    """Detect beads on a (preferably white) background.

    Parameters
    ----------
    pixels_per_mm:
        Used to translate the physical size gate into pixels so the same
        detector works at any magnification.
    min_diameter_mm / max_diameter_mm:
        Physical size band of acceptable beads; anything outside is discarded as
        dust or background structure.
    exclude_roi:
        Bounding box of the calibration reference, ignored during detection.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Beads are darker than the white background -> inverse Otsu gives foreground.
    _, binary = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]  # shape (N, 4): [next, prev, first_child, parent]

    min_r_px = (min_diameter_mm / 2.0) * pixels_per_mm
    max_r_px = (max_diameter_mm / 2.0) * pixels_per_mm
    min_area = np.pi * min_r_px ** 2
    max_area = np.pi * max_r_px ** 2

    beads: list[BeadCandidate] = []
    for idx, contour in enumerate(contours):
        # Only top-level contours (parent == -1) are bead outer boundaries.
        if hierarchy[idx][3] != -1:
            continue
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        perim = cv2.arcLength(contour, True)
        if perim == 0:
            continue
        circularity = 4 * np.pi * area / (perim * perim)
        if circularity < min_circularity:
            continue

        bbox = cv2.boundingRect(contour)
        if exclude_roi is not None and _rect_overlap(bbox, exclude_roi):
            continue

        m = cv2.moments(contour)
        if m["m00"] == 0:
            continue
        centroid = (m["m10"] / m["m00"], m["m01"] / m["m00"])

        hole_contour = _largest_child_hole(
            contours, hierarchy, idx, min_hole_area=0.01 * area
        )
        hole_circle = None
        if hole_contour is None:
            hole_circle = _hough_hole(gray, bbox)

        beads.append(
            BeadCandidate(
                outer_contour=contour.reshape(-1, 2).astype(np.float64),
                hole_contour=(
                    hole_contour.reshape(-1, 2).astype(np.float64)
                    if hole_contour is not None
                    else None
                ),
                hole_circle=hole_circle,
                bbox=bbox,
                centroid=centroid,
                outer_area=area,
            )
        )

    # Sort top-to-bottom, left-to-right for stable, human-friendly numbering.
    beads.sort(key=lambda b: (round(b.centroid[1] / 50), b.centroid[0]))
    return beads


def _largest_child_hole(contours, hierarchy, parent_idx, *, min_hole_area):
    """Return the largest contour nested directly inside ``parent_idx``."""
    child = hierarchy[parent_idx][2]
    best = None
    best_area = 0.0
    while child != -1:
        area = cv2.contourArea(contours[child])
        if area >= min_hole_area and area > best_area:
            best_area = area
            best = contours[child]
        child = hierarchy[child][0]  # next sibling
    return best


def _hough_hole(gray, bbox):
    """Fallback hole detection via Hough circles inside the bead bounding box."""
    x, y, w, h = bbox
    roi = gray[y : y + h, x : x + w]
    if roi.size == 0:
        return None
    roi_blur = cv2.medianBlur(roi, 3)
    min_dim = min(w, h)
    circles = cv2.HoughCircles(
        roi_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dim,
        param1=120,
        param2=20,
        minRadius=max(2, int(0.05 * min_dim)),
        maxRadius=int(0.45 * min_dim),
    )
    if circles is None:
        return None
    cx, cy, r = circles[0][0]
    return (float(cx + x), float(cy + y), float(r))
