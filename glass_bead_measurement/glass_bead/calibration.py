"""Pixel-to-millimetre calibration.

No calibration, no measurement.  Every reported dimension is only as trustworthy
as the scale factor that converts pixels to millimetres, so this module is the
gatekeeper of the whole pipeline.

Three reference types are supported, in descending order of robustness:

* ``aruco``  -- an ArUco marker whose physical side length (mm) is known.  The
  four corners give four independent length measurements, so we also get a
  built-in self-consistency check.  This is the recommended reference.
* ``circle`` -- a printed calibration dot/ring of known diameter (mm).
* ``manual`` -- the operator supplies ``pixels_per_mm`` directly (e.g. from a
  ruler measured by hand).  Use only when nothing else is available.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .circle_fit import fit_circle_ransac


@dataclass
class Calibration:
    pixels_per_mm: float
    method: str
    confidence: float          # 0..1 self-consistency of the reference.
    # Bounding box (x, y, w, h) of the reference in the image, if known, so the
    # bead detector can ignore it.
    roi: tuple[int, int, int, int] | None = None
    detail: str = ""

    @property
    def mm_per_pixel(self) -> float:
        return 1.0 / self.pixels_per_mm if self.pixels_per_mm else float("inf")

    def px_to_mm(self, value_px: float) -> float:
        return value_px / self.pixels_per_mm

    def as_dict(self) -> dict:
        return {
            "pixels_per_mm": round(self.pixels_per_mm, 3),
            "method": self.method,
            "confidence": round(self.confidence, 3),
            "detail": self.detail,
        }


class CalibrationError(RuntimeError):
    """Raised when a required calibration reference cannot be established."""


def _aruco_detector(dictionary_id: int):
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(dictionary_id)
    # OpenCV moved the API around 4.7; support both new and old layouts.
    if hasattr(aruco, "ArucoDetector"):
        params = aruco.DetectorParameters()
        try:
            params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        except AttributeError:
            pass
        return aruco.ArucoDetector(dictionary, params)
    return None, dictionary  # legacy path handled in caller


def calibrate_from_aruco(
    image_bgr: np.ndarray,
    marker_length_mm: float,
    *,
    dictionary_id: int | None = None,
) -> Calibration:
    """Establish scale from an ArUco marker of known side length."""
    if marker_length_mm <= 0:
        raise CalibrationError("marker_length_mm must be positive.")

    aruco = cv2.aruco
    dictionary_id = (
        aruco.DICT_4X4_50 if dictionary_id is None else dictionary_id
    )
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    detector = _aruco_detector(dictionary_id)
    if isinstance(detector, tuple):  # legacy API
        _, dictionary = detector
        corners, ids, _ = aruco.detectMarkers(gray, dictionary)
    else:
        corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(corners) == 0:
        raise CalibrationError("No ArUco calibration marker found in the image.")

    # Use the first marker. ``corners[i]`` is (1, 4, 2): TL, TR, BR, BL.
    pts = corners[0].reshape(4, 2).astype(np.float64)
    side_lengths = [
        np.linalg.norm(pts[0] - pts[1]),
        np.linalg.norm(pts[1] - pts[2]),
        np.linalg.norm(pts[2] - pts[3]),
        np.linalg.norm(pts[3] - pts[0]),
    ]
    mean_side = float(np.mean(side_lengths))
    pixels_per_mm = mean_side / marker_length_mm

    # Confidence from how square the marker looks (sides equal == 1.0).
    spread = float(np.std(side_lengths) / mean_side) if mean_side else 1.0
    confidence = float(np.clip(1.0 - spread * 4.0, 0.0, 1.0))

    x, y, w, h = cv2.boundingRect(pts.astype(np.float32))
    # Pad the ROI so beads touching the marker are still excluded.
    pad = int(0.25 * max(w, h))
    roi = (max(0, x - pad), max(0, y - pad), w + 2 * pad, h + 2 * pad)

    return Calibration(
        pixels_per_mm=pixels_per_mm,
        method="aruco",
        confidence=confidence,
        roi=roi,
        detail=(
            f"ArUco id {int(ids[0][0])}, side {mean_side:.1f} px "
            f"= {marker_length_mm} mm"
        ),
    )


def calibrate_from_circle(
    image_bgr: np.ndarray,
    diameter_mm: float,
    *,
    min_radius_px: int = 15,
    max_radius_px: int | None = None,
) -> Calibration:
    """Establish scale from a printed calibration circle of known diameter.

    The most circular dark blob in the image is assumed to be the reference.
    For repeatable results prefer a solid black dot on a white card.
    """
    if diameter_mm <= 0:
        raise CalibrationError("diameter_mm must be positive.")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if max_radius_px is None:
        max_radius_px = min(gray.shape) // 2

    best = None
    best_circularity = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < np.pi * min_radius_px ** 2:
            continue
        perim = cv2.arcLength(c, True)
        if perim == 0:
            continue
        circularity = 4 * np.pi * area / (perim * perim)
        (_, _), r = cv2.minEnclosingCircle(c)
        if r < min_radius_px or r > max_radius_px:
            continue
        if circularity > best_circularity:
            best_circularity = circularity
            best = c

    if best is None or best_circularity < 0.8:
        raise CalibrationError(
            "No clear calibration circle found (need a solid round reference)."
        )

    fit = fit_circle_ransac(best.reshape(-1, 2).astype(np.float64))
    pixels_per_mm = fit.diameter / diameter_mm
    x, y, w, h = cv2.boundingRect(best)
    pad = int(0.3 * max(w, h))
    roi = (max(0, x - pad), max(0, y - pad), w + 2 * pad, h + 2 * pad)

    return Calibration(
        pixels_per_mm=pixels_per_mm,
        method="circle",
        confidence=float(np.clip(best_circularity, 0.0, 1.0)),
        roi=roi,
        detail=(
            f"Calibration circle {fit.diameter:.1f} px = {diameter_mm} mm "
            f"(circularity {best_circularity:.3f})"
        ),
    )


def calibrate_manual(pixels_per_mm: float) -> Calibration:
    """Use an operator-supplied scale factor."""
    if pixels_per_mm <= 0:
        raise CalibrationError("pixels_per_mm must be positive.")
    return Calibration(
        pixels_per_mm=pixels_per_mm,
        method="manual",
        confidence=0.85,
        roi=None,
        detail=f"Manual scale {pixels_per_mm:.2f} px/mm",
    )


def calibrate(
    image_bgr: np.ndarray,
    *,
    method: str,
    marker_length_mm: float | None = None,
    diameter_mm: float | None = None,
    pixels_per_mm: float | None = None,
    dictionary_id: int | None = None,
) -> Calibration:
    """Dispatch to the requested calibration method.

    Raises ``CalibrationError`` if the reference is missing or invalid -- the
    pipeline treats that as a hard stop, never a guess.
    """
    method = method.lower()
    if method == "aruco":
        if marker_length_mm is None:
            raise CalibrationError("aruco calibration needs marker_length_mm.")
        return calibrate_from_aruco(
            image_bgr, marker_length_mm, dictionary_id=dictionary_id
        )
    if method == "circle":
        if diameter_mm is None:
            raise CalibrationError("circle calibration needs diameter_mm.")
        return calibrate_from_circle(image_bgr, diameter_mm)
    if method == "manual":
        if pixels_per_mm is None:
            raise CalibrationError("manual calibration needs pixels_per_mm.")
        return calibrate_manual(pixels_per_mm)
    raise CalibrationError(f"Unknown calibration method: {method!r}")
