"""End-to-end measurement pipeline.

This is the single entry point the UI and CLI call.  It enforces the workflow
from the specification, in order:

1. Quality gate (blur / glare / resolution)         -> hard stop on failure
2. Calibration reference detection                  -> hard stop on failure
3. Bead detection & segmentation
4. Sub-pixel circle fitting (outer + hole)
5. OD / ID / wall thickness / roundness / confidence
6. Annotated image

The guiding rule from the brief is honoured throughout: *never estimate a
dimension without calibration, and refuse to report a measurement we don't
trust.*
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .calibration import Calibration, CalibrationError, calibrate
from .detection import segment_beads
from .measurement import BeadMeasurement, local_edge_contrast, measure_bead
from .quality import ImageQuality, assess_quality
from .visualization import annotate


@dataclass
class MeasurementResult:
    ok: bool
    message: str
    quality: ImageQuality
    calibration: Calibration | None
    beads: list[BeadMeasurement] = field(default_factory=list)
    annotated: np.ndarray | None = None

    def summary_text(self) -> str:
        """Human-readable, spec-formatted report string."""
        if not self.ok:
            return f"Measurement not reliable. {self.message}".strip()
        if not self.beads:
            return "No beads detected in the image."
        lines = []
        for m in self.beads:
            lines.append(f"Bead #{m.index}")
            lines.append(f"Outer Diameter: {m.outer_diameter_mm:.2f} mm")
            if m.hole_diameter_mm is not None:
                lines.append(f"Hole Diameter: {m.hole_diameter_mm:.2f} mm")
            if m.wall_thickness_mm is not None:
                lines.append(f"Wall Thickness: {m.wall_thickness_mm:.2f} mm")
            lines.append(f"Roundness: {m.roundness_pct:.1f}%")
            lines.append(f"Confidence: {m.confidence_pct:.0f}%")
            if not m.reliable:
                lines.append("  -> Measurement not reliable. Please retake photo.")
            lines.append("")
        return "\n".join(lines).strip()


def measure_image(
    image_bgr: np.ndarray,
    *,
    calibration_method: str = "aruco",
    marker_length_mm: float | None = None,
    diameter_mm: float | None = None,
    pixels_per_mm: float | None = None,
    dictionary_id: int | None = None,
    min_diameter_mm: float = 1.0,
    max_diameter_mm: float = 8.0,
    min_confidence: float = 90.0,
    enforce_quality: bool = True,
) -> MeasurementResult:
    """Run the full measurement pipeline on a single BGR image."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image supplied.")

    # 1. Quality gate -------------------------------------------------------
    quality = assess_quality(image_bgr)
    if enforce_quality and not quality.passed:
        return MeasurementResult(
            ok=False,
            message="Please retake photo. " + " ".join(quality.reasons),
            quality=quality,
            calibration=None,
        )

    # 2. Calibration (hard requirement) ------------------------------------
    try:
        calibration = calibrate(
            image_bgr,
            method=calibration_method,
            marker_length_mm=marker_length_mm,
            diameter_mm=diameter_mm,
            pixels_per_mm=pixels_per_mm,
            dictionary_id=dictionary_id,
        )
    except CalibrationError as exc:
        return MeasurementResult(
            ok=False,
            message=f"Calibration reference missing/invalid: {exc}",
            quality=quality,
            calibration=None,
        )

    # 3. Bead detection -----------------------------------------------------
    candidates = segment_beads(
        image_bgr,
        pixels_per_mm=calibration.pixels_per_mm,
        min_diameter_mm=min_diameter_mm,
        max_diameter_mm=max_diameter_mm,
        exclude_roi=calibration.roi,
    )
    if not candidates:
        return MeasurementResult(
            ok=True,
            message="Calibration succeeded but no beads were detected.",
            quality=quality,
            calibration=calibration,
            beads=[],
            annotated=annotate(image_bgr, [], calibration),
        )

    # 4-5. Measure each bead ------------------------------------------------
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    measurements: list[BeadMeasurement] = []
    for i, cand in enumerate(candidates, start=1):
        contrast = local_edge_contrast(gray, cand)
        measurements.append(
            measure_bead(
                gray,
                cand,
                calibration,
                index=i,
                min_confidence=min_confidence,
                edge_contrast=contrast,
            )
        )

    # 6. Annotated image ----------------------------------------------------
    annotated = annotate(image_bgr, measurements, calibration)

    reliable = sum(1 for m in measurements if m.reliable)
    return MeasurementResult(
        ok=True,
        message=f"{len(measurements)} bead(s) measured, {reliable} reliable.",
        quality=quality,
        calibration=calibration,
        beads=measurements,
        annotated=annotated,
    )
