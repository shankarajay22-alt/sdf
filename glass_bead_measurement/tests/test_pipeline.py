"""Accuracy and behaviour tests on synthetic, ground-truth-known scenes."""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glass_bead import measure_image  # noqa: E402
from glass_bead.circle_fit import fit_circle_ransac  # noqa: E402
from glass_bead.synthetic import (  # noqa: E402
    SyntheticBead,
    default_demo_scene,
    render_scene,
)


def test_circle_fit_recovers_known_circle():
    angles = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    cx, cy, r = 123.4, 88.7, 55.2
    pts = np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])
    rng = np.random.default_rng(0)
    pts += rng.normal(0, 0.2, pts.shape)
    fit = fit_circle_ransac(pts)
    assert abs(fit.cx - cx) < 0.2
    assert abs(fit.cy - cy) < 0.2
    assert abs(fit.radius - r) < 0.2


def test_pipeline_measures_demo_scene_accurately():
    img, gt = default_demo_scene()
    result = measure_image(
        img,
        calibration_method="aruco",
        marker_length_mm=gt["marker_length_mm"],
    )
    assert result.ok, result.message
    assert len(result.beads) == len(gt["beads"])

    # Match measured beads to ground truth by nearest outer diameter.
    true_ods = sorted(b["outer_diameter_mm"] for b in gt["beads"])
    meas_ods = sorted(b.outer_diameter_mm for b in result.beads)
    for t, m in zip(true_ods, meas_ods):
        # Synthetic edges are clean; expect well under 0.1 mm error.
        assert abs(t - m) < 0.10, f"OD error too large: true {t}, meas {m}"


def test_hole_diameter_accuracy():
    img, gt = render_scene(
        [SyntheticBead(20, 15, outer_diameter_mm=3.0, hole_diameter_mm=1.0)],
        pixels_per_mm=180.0,
    )
    result = measure_image(img, calibration_method="aruco",
                           marker_length_mm=gt["marker_length_mm"])
    assert result.ok
    bead = result.beads[0]
    assert bead.hole_diameter_mm is not None
    assert abs(bead.hole_diameter_mm - 1.0) < 0.12


def test_missing_calibration_is_rejected():
    img, _ = render_scene(
        [SyntheticBead(20, 15, 3.0, 1.0)],
        add_marker=False,
    )
    result = measure_image(img, calibration_method="aruco", marker_length_mm=10.0)
    assert not result.ok
    assert "calibration" in result.message.lower()


def test_manual_calibration_path():
    img, gt = render_scene(
        [SyntheticBead(20, 15, 3.0, 1.0)],
        pixels_per_mm=160.0,
        add_marker=False,
    )
    result = measure_image(
        img,
        calibration_method="manual",
        pixels_per_mm=gt["pixels_per_mm"],
    )
    assert result.ok
    assert abs(result.beads[0].outer_diameter_mm - 3.0) < 0.12


def test_blurry_image_is_rejected():
    img, gt = default_demo_scene()
    blurred = cv2.GaussianBlur(img, (25, 25), 12)
    result = measure_image(
        blurred, calibration_method="manual",
        pixels_per_mm=gt["pixels_per_mm"],
    )
    assert not result.ok
    assert "retake" in result.message.lower()


def test_roundness_is_high_for_clean_beads():
    img, gt = default_demo_scene()
    result = measure_image(img, calibration_method="aruco",
                           marker_length_mm=gt["marker_length_mm"])
    for bead in result.beads:
        assert bead.roundness_pct > 90.0
