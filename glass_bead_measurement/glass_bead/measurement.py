"""Per-bead dimensional measurement.

This ties the geometry together: take a detected bead, refine its edges to
sub-pixel precision, fit circles to the outer rim and the hole, convert to
millimetres with the calibration, and attach a confidence score.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .calibration import Calibration
from .circle_fit import CircleFit, fit_circle_ransac
from .confidence import score_confidence
from .detection import BeadCandidate
from .subpixel import refine_edge_points


@dataclass
class BeadMeasurement:
    index: int
    outer_diameter_mm: float
    hole_diameter_mm: float | None
    wall_thickness_mm: float | None
    roundness_pct: float
    confidence_pct: float
    reliable: bool
    # Pixel-space geometry for visualisation.
    outer_circle: CircleFit
    hole_circle: CircleFit | None
    notes: list[str]

    def as_dict(self) -> dict:
        return {
            "bead": self.index,
            "outer_diameter_mm": _round2(self.outer_diameter_mm),
            "hole_diameter_mm": _round2(self.hole_diameter_mm),
            "wall_thickness_mm": _round2(self.wall_thickness_mm),
            "roundness_pct": round(self.roundness_pct, 1),
            "confidence_pct": round(self.confidence_pct, 1),
            "reliable": self.reliable,
            "notes": "; ".join(self.notes),
        }


def _round2(value):
    return None if value is None else round(float(value), 2)


def measure_bead(
    gray: np.ndarray,
    candidate: BeadCandidate,
    calibration: Calibration,
    index: int,
    *,
    min_confidence: float = 90.0,
    edge_contrast: float = 0.0,
) -> BeadMeasurement:
    """Measure a single bead candidate.

    ``gray`` must be the float grayscale image used for sub-pixel refinement.
    """
    notes: list[str] = []

    # --- Outer boundary -----------------------------------------------------
    rough_outer = fit_circle_ransac(candidate.outer_contour)
    refined_outer_pts = refine_edge_points(
        gray, candidate.outer_contour, (rough_outer.cx, rough_outer.cy)
    )
    outer = fit_circle_ransac(refined_outer_pts)
    outer_diameter_mm = calibration.px_to_mm(outer.diameter)

    # --- Hole boundary ------------------------------------------------------
    hole: CircleFit | None = None
    hole_diameter_mm: float | None = None
    if candidate.hole_contour is not None and len(candidate.hole_contour) >= 5:
        rough_hole = fit_circle_ransac(candidate.hole_contour)
        refined_hole_pts = refine_edge_points(
            gray, candidate.hole_contour, (rough_hole.cx, rough_hole.cy),
            search=3.0,
        )
        hole = fit_circle_ransac(refined_hole_pts)
        hole_diameter_mm = calibration.px_to_mm(hole.diameter)
    elif candidate.hole_circle is not None:
        cx, cy, r = candidate.hole_circle
        hole = CircleFit(cx, cy, r, rms_residual=1.0, r_max=r, r_min=r, n_points=0)
        hole_diameter_mm = calibration.px_to_mm(hole.diameter)
        notes.append("Hole found via Hough fallback (lower precision).")
    else:
        notes.append("No central hole detected.")

    wall_thickness_mm = None
    if hole_diameter_mm is not None:
        wall_thickness_mm = (outer_diameter_mm - hole_diameter_mm) / 2.0
        if wall_thickness_mm <= 0:
            notes.append("Hole diameter >= outer diameter; check segmentation.")

    confidence_pct = score_confidence(
        outer=outer,
        hole=hole,
        calibration=calibration,
        edge_contrast=edge_contrast,
        pixels_per_mm=calibration.pixels_per_mm,
    )
    reliable = confidence_pct >= min_confidence
    if not reliable:
        notes.append(
            f"Confidence {confidence_pct:.0f}% below {min_confidence:.0f}% threshold."
        )

    return BeadMeasurement(
        index=index,
        outer_diameter_mm=outer_diameter_mm,
        hole_diameter_mm=hole_diameter_mm,
        wall_thickness_mm=wall_thickness_mm,
        roundness_pct=outer.roundness_pct,
        confidence_pct=confidence_pct,
        reliable=reliable,
        outer_circle=outer,
        hole_circle=hole,
        notes=notes,
    )


def local_edge_contrast(gray: np.ndarray, candidate: BeadCandidate) -> float:
    """Estimate edge sharpness around a bead as a 0..1 contrast figure.

    Used by the confidence model: a crisp dark/light transition at the rim is a
    strong indicator that the measured edge is real and well localised.
    """
    x, y, w, h = candidate.bbox
    pad = int(0.15 * max(w, h))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(gray.shape[1], x + w + pad)
    y1 = min(gray.shape[0], y + h + pad)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0
    gx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    # Normalise: 255*sqrt(2) is the theoretical max Sobel magnitude.
    return float(np.clip(np.percentile(mag, 99) / (255.0 * 1.414), 0.0, 1.0))
