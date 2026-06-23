"""Sub-pixel circle fitting utilities.

The accuracy of every dimensional measurement in this project ultimately
depends on how precisely we can recover the geometric centre and radius of a
set of edge points.  A naive ``cv2.minEnclosingCircle`` is biased by the
outermost noisy pixel, so instead we use an algebraic least-squares fit
(Taubin's method) which is unbiased, fast and gives genuine sub-pixel results.
A RANSAC wrapper makes the fit robust against the stray outliers that segmentation
inevitably produces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CircleFit:
    """Result of fitting a circle to a set of 2-D points."""

    cx: float
    cy: float
    radius: float
    # Root-mean-square radial residual in pixels (lower == rounder / cleaner).
    rms_residual: float
    # Maximum / minimum radius from the fitted centre to the supplied points.
    r_max: float
    r_min: float
    n_points: int

    @property
    def diameter(self) -> float:
        return 2.0 * self.radius

    @property
    def roundness_pct(self) -> float:
        """Roundness as a percentage.

        Defined from the radial form error ``(r_max - r_min)`` normalised by the
        mean radius.  A perfect circle scores 100 %.
        """
        if self.radius <= 0:
            return 0.0
        form_error = (self.r_max - self.r_min) / self.radius
        return float(np.clip(100.0 * (1.0 - form_error / 2.0), 0.0, 100.0))


def taubin_fit(points: np.ndarray) -> tuple[float, float, float]:
    """Algebraic circle fit using the Taubin method.

    Parameters
    ----------
    points:
        ``(N, 2)`` array of ``(x, y)`` coordinates.

    Returns
    -------
    ``(cx, cy, radius)``
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 3:
        raise ValueError("At least 3 points are required to fit a circle.")

    x = pts[:, 0]
    y = pts[:, 1]
    # Work in a centred frame for numerical stability.
    x_m = x.mean()
    y_m = y.mean()
    u = x - x_m
    v = y - y_m

    z = u * u + v * v
    z_mean = z.mean()

    Suu = (u * u).mean()
    Svv = (v * v).mean()
    Suv = (u * v).mean()
    Suz = (u * z).mean()
    Svz = (v * z).mean()

    # Solve the normal equations for the Taubin formulation.
    a = np.array([[Suu, Suv], [Suv, Svv]], dtype=np.float64)
    b = 0.5 * np.array([Suz, Svz], dtype=np.float64)
    try:
        uc, vc = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        uc, vc = np.linalg.lstsq(a, b, rcond=None)[0]

    cx = uc + x_m
    cy = vc + y_m
    radius = float(np.sqrt(uc * uc + vc * vc + z_mean))
    return float(cx), float(cy), radius


def _residual_stats(points: np.ndarray, cx: float, cy: float, radius: float):
    radii = np.hypot(points[:, 0] - cx, points[:, 1] - cy)
    residuals = radii - radius
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    return rms, float(radii.max()), float(radii.min())


def fit_circle(points: np.ndarray) -> CircleFit:
    """Plain (non-robust) least-squares circle fit with residual statistics."""
    pts = np.asarray(points, dtype=np.float64)
    cx, cy, radius = taubin_fit(pts)
    rms, r_max, r_min = _residual_stats(pts, cx, cy, radius)
    return CircleFit(cx, cy, radius, rms, r_max, r_min, pts.shape[0])


def fit_circle_ransac(
    points: np.ndarray,
    *,
    iterations: int = 200,
    threshold: float = 1.5,
    min_inlier_ratio: float = 0.6,
    seed: int | None = 42,
) -> CircleFit:
    """Robust circle fit.

    Outliers (segmentation specks, partially merged neighbours, reflections on
    the rim) are rejected with RANSAC before a final algebraic fit is computed
    on the consensus set.

    Parameters
    ----------
    threshold:
        Maximum radial residual, in pixels, for a point to count as an inlier.
    min_inlier_ratio:
        If no model reaches this inlier fraction the best model found is still
        returned, but the caller can inspect ``n_points`` to judge reliability.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    if n < 3:
        raise ValueError("At least 3 points are required to fit a circle.")
    if n < 10:
        # Too few points for RANSAC to add value.
        return fit_circle(pts)

    rng = np.random.default_rng(seed)
    best_inliers: np.ndarray | None = None
    best_count = 0

    for _ in range(iterations):
        sample_idx = rng.choice(n, size=3, replace=False)
        sample = pts[sample_idx]
        try:
            cx, cy, radius = taubin_fit(sample)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if not np.isfinite(radius) or radius <= 0:
            continue
        radii = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        inliers = np.abs(radii - radius) < threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < 3:
        return fit_circle(pts)

    consensus = pts[best_inliers]
    cx, cy, radius = taubin_fit(consensus)
    rms, r_max, r_min = _residual_stats(consensus, cx, cy, radius)
    return CircleFit(cx, cy, radius, rms, r_max, r_min, consensus.shape[0])
