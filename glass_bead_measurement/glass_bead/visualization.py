"""Annotated output rendering.

Draws the fitted geometry back onto the photo so an operator can sanity-check
every measurement at a glance: green outer boundary, red hole, labelled
dimensions, plus a scale bar derived from the calibration.
"""

from __future__ import annotations

import cv2
import numpy as np

from .calibration import Calibration
from .measurement import BeadMeasurement

GREEN = (0, 200, 0)
RED = (0, 0, 255)
YELLOW = (0, 215, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def _put_label(img, text, org, color, scale=0.6, thickness=2):
    """Draw text with a dark outline so it stays readable over any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK,
                thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


def draw_scale_bar(img, calibration: Calibration, length_mm: float = 1.0):
    h, w = img.shape[:2]
    bar_px = int(round(length_mm * calibration.pixels_per_mm))
    if bar_px <= 0 or bar_px > w * 0.8:
        return
    margin = 30
    x0 = margin
    y0 = h - margin
    cv2.line(img, (x0, y0), (x0 + bar_px, y0), WHITE, 6)
    cv2.line(img, (x0, y0), (x0 + bar_px, y0), BLACK, 2)
    _put_label(img, f"{length_mm:.0f} mm", (x0, y0 - 12), WHITE, 0.6, 2)


def annotate(
    image_bgr: np.ndarray,
    measurements: list[BeadMeasurement],
    calibration: Calibration,
) -> np.ndarray:
    """Return a copy of ``image_bgr`` annotated with all measurements."""
    out = image_bgr.copy()

    for m in measurements:
        oc = m.outer_circle
        center = (int(round(oc.cx)), int(round(oc.cy)))
        radius = int(round(oc.radius))
        color = GREEN if m.reliable else YELLOW
        cv2.circle(out, center, radius, color, 2, cv2.LINE_AA)
        cv2.drawMarker(out, center, color, cv2.MARKER_CROSS, 12, 2)

        if m.hole_circle is not None:
            hc = m.hole_circle
            cv2.circle(
                out,
                (int(round(hc.cx)), int(round(hc.cy))),
                max(1, int(round(hc.radius))),
                RED,
                2,
                cv2.LINE_AA,
            )

        # Stacked label block beside the bead.
        lx = center[0] + radius + 8
        ly = center[1] - radius
        lines = [
            f"#{m.index}",
            f"OD {m.outer_diameter_mm:.2f} mm",
        ]
        if m.hole_diameter_mm is not None:
            lines.append(f"ID {m.hole_diameter_mm:.2f} mm")
        if m.wall_thickness_mm is not None:
            lines.append(f"Wall {m.wall_thickness_mm:.2f} mm")
        lines.append(f"Conf {m.confidence_pct:.0f}%")
        for i, line in enumerate(lines):
            _put_label(out, line, (lx, ly + 22 * i + 20), color, 0.55, 1)

    draw_scale_bar(out, calibration, 1.0)
    _put_label(
        out,
        f"Scale: {calibration.pixels_per_mm:.1f} px/mm ({calibration.method})",
        (30, 40),
        WHITE,
        0.6,
        2,
    )
    return out
