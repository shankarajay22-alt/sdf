"""Synthetic test-image generation.

Real calibrated bead photos are not always at hand, so we can render a
physically-consistent scene: a white background, an ArUco calibration marker of
known size, and beads of known outer/inner diameter.  Because the ground truth
is known exactly, these images double as the basis for the accuracy tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SyntheticBead:
    cx_mm: float
    cy_mm: float
    outer_diameter_mm: float
    hole_diameter_mm: float


def render_scene(
    beads: list[SyntheticBead],
    *,
    pixels_per_mm: float = 150.0,
    canvas_mm: tuple[float, float] = (40.0, 30.0),
    marker_length_mm: float = 10.0,
    marker_origin_mm: tuple[float, float] = (2.0, 2.0),
    dictionary_id: int | None = None,
    add_marker: bool = True,
) -> tuple[np.ndarray, dict]:
    """Render a scene and return ``(image_bgr, ground_truth)``.

    ``ground_truth`` contains the exact ``pixels_per_mm`` and per-bead
    dimensions so tests can assert measured-vs-true error.
    """
    w_mm, h_mm = canvas_mm
    w = int(round(w_mm * pixels_per_mm))
    h = int(round(h_mm * pixels_per_mm))
    img = np.full((h, w, 3), 245, dtype=np.uint8)  # near-white background

    def mm2px(x_mm, y_mm):
        return int(round(x_mm * pixels_per_mm)), int(round(y_mm * pixels_per_mm))

    if add_marker:
        aruco = cv2.aruco
        dict_id = aruco.DICT_4X4_50 if dictionary_id is None else dictionary_id
        dictionary = aruco.getPredefinedDictionary(dict_id)
        side_px = int(round(marker_length_mm * pixels_per_mm))
        if hasattr(aruco, "generateImageMarker"):
            marker = aruco.generateImageMarker(dictionary, 0, side_px)
        else:  # legacy API
            marker = aruco.drawMarker(dictionary, 0, side_px)
        mx, my = mm2px(*marker_origin_mm)
        marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        img[my : my + side_px, mx : mx + side_px] = marker_bgr

    for b in beads:
        cx, cy = mm2px(b.cx_mm, b.cy_mm)
        r_out = int(round(b.outer_diameter_mm / 2.0 * pixels_per_mm))
        r_in = int(round(b.hole_diameter_mm / 2.0 * pixels_per_mm))
        # Glass bead: a darker ring on white, with a white hole in the centre.
        cv2.circle(img, (cx, cy), r_out, (70, 90, 120), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), r_out, (40, 50, 70), 2, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), r_in, (245, 245, 245), -1, cv2.LINE_AA)

    ground_truth = {
        "pixels_per_mm": pixels_per_mm,
        "marker_length_mm": marker_length_mm,
        "beads": [
            {
                "outer_diameter_mm": b.outer_diameter_mm,
                "hole_diameter_mm": b.hole_diameter_mm,
            }
            for b in beads
        ],
    }
    return img, ground_truth


def default_demo_scene() -> tuple[np.ndarray, dict]:
    """A representative 2-bead scene in the 1.7-4 mm working range."""
    beads = [
        SyntheticBead(cx_mm=18.0, cy_mm=12.0, outer_diameter_mm=2.50,
                      hole_diameter_mm=0.80),
        SyntheticBead(cx_mm=30.0, cy_mm=20.0, outer_diameter_mm=3.80,
                      hole_diameter_mm=1.20),
    ]
    return render_scene(beads)
