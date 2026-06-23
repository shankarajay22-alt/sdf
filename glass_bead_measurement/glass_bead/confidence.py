"""Measurement confidence scoring.

A single number that answers "how much should I trust this measurement?".  It
is deliberately conservative: any one weak signal (poor calibration, ragged
edge, non-circular rim) pulls the score down, because in metrology an
over-confident wrong answer is worse than an honest "retake the photo".
"""

from __future__ import annotations

import numpy as np

from .calibration import Calibration
from .circle_fit import CircleFit


def _residual_factor(fit: CircleFit) -> float:
    """0..1 score from the fit's RMS radial residual (relative to radius)."""
    if fit.radius <= 0:
        return 0.0
    rel = fit.rms_residual / fit.radius
    # 0.5% relative residual -> ~0.5 score; tighten/loosen via the scale.
    return float(np.clip(1.0 - rel * 100.0, 0.0, 1.0))


def _roundness_factor(fit: CircleFit) -> float:
    return float(np.clip(fit.roundness_pct / 100.0, 0.0, 1.0))


def _resolution_factor(fit: CircleFit, pixels_per_mm: float) -> float:
    """Penalise beads that occupy too few pixels to measure precisely.

    The brief recommends >=300 px diameter; we score linearly up to that.
    """
    diameter_px = fit.diameter
    return float(np.clip(diameter_px / 300.0, 0.0, 1.0))


def score_confidence(
    *,
    outer: CircleFit,
    hole: CircleFit | None,
    calibration: Calibration,
    edge_contrast: float,
    pixels_per_mm: float,
) -> float:
    """Combine geometric, optical and calibration cues into a 0..100 score."""
    outer_fit = _residual_factor(outer)
    roundness = _roundness_factor(outer)
    resolution = _resolution_factor(outer, pixels_per_mm)
    contrast = float(np.clip(edge_contrast, 0.0, 1.0))
    calib = float(np.clip(calibration.confidence, 0.0, 1.0))

    hole_fit = _residual_factor(hole) if hole is not None else 0.7

    # Weighted geometric mean keeps the score honest: a near-zero factor in any
    # critical dimension drags the whole result down rather than averaging out.
    factors = np.array(
        [outer_fit, roundness, resolution, contrast, calib, hole_fit]
    )
    weights = np.array([0.25, 0.15, 0.15, 0.15, 0.20, 0.10])
    factors = np.clip(factors, 1e-3, 1.0)
    score = float(np.exp(np.sum(weights * np.log(factors))))
    return round(score * 100.0, 1)
