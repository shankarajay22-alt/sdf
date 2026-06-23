"""Sub-pixel edge localisation.

OpenCV contours are quantised to whole pixels.  At ~150 px/mm a one-pixel
error is already ~0.007 mm, so to approach the ±0.01 mm target we must locate
each edge to a fraction of a pixel.  We do this by walking a short intensity
profile across every contour point, along the local radial direction, and
finding the gradient peak with parabolic interpolation.
"""

from __future__ import annotations

import numpy as np


def _sample_profile(gray: np.ndarray, x: float, y: float) -> float:
    """Bilinearly sample ``gray`` at floating-point ``(x, y)``."""
    h, w = gray.shape
    if x < 0 or y < 0 or x > w - 1 or y > h - 1:
        return float("nan")
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    dx = x - x0
    dy = y - y0
    v00 = gray[y0, x0]
    v01 = gray[y0, x1]
    v10 = gray[y1, x0]
    v11 = gray[y1, x1]
    top = v00 * (1 - dx) + v01 * dx
    bot = v10 * (1 - dx) + v11 * dx
    return float(top * (1 - dy) + bot * dy)


def refine_edge_points(
    gray: np.ndarray,
    contour_xy: np.ndarray,
    center: tuple[float, float],
    *,
    search: float = 4.0,
    step: float = 0.25,
) -> np.ndarray:
    """Refine contour points to sub-pixel accuracy along the radial direction.

    For each point we sample the image intensity on a short line segment that
    passes through the point and points away from ``center``.  The true edge is
    the location of maximum intensity gradient, estimated to sub-pixel precision
    by fitting a parabola to the three samples around the discrete maximum.

    Points whose gradient is too weak (e.g. blur, low contrast) are left at
    their original location so the fit degrades gracefully rather than drifting.

    Parameters
    ----------
    gray:
        Single-channel ``float`` image.
    contour_xy:
        ``(N, 2)`` array of integer-ish contour coordinates.
    center:
        Approximate circle centre used to define the outward radial direction.
    search:
        Half-length of the search segment in pixels.
    step:
        Sampling step in pixels along the segment.
    """
    gray_f = gray.astype(np.float64)
    cx, cy = center
    offsets = np.arange(-search, search + 1e-9, step)
    refined = []

    for px, py in contour_xy:
        dirx = px - cx
        diry = py - cy
        norm = np.hypot(dirx, diry)
        if norm < 1e-6:
            refined.append((px, py))
            continue
        dirx /= norm
        diry /= norm

        samples = np.array(
            [_sample_profile(gray_f, px + o * dirx, py + o * diry) for o in offsets]
        )
        if np.isnan(samples).any():
            refined.append((px, py))
            continue

        # Gradient magnitude along the profile.
        grad = np.abs(np.gradient(samples))
        k = int(np.argmax(grad))
        if k <= 0 or k >= len(grad) - 1:
            refined.append((px, py))
            continue

        # Parabolic interpolation of the gradient peak for sub-pixel offset.
        g0, g1, g2 = grad[k - 1], grad[k], grad[k + 1]
        denom = g0 - 2 * g1 + g2
        delta = 0.0 if abs(denom) < 1e-9 else 0.5 * (g0 - g2) / denom
        delta = float(np.clip(delta, -1.0, 1.0))
        best_offset = offsets[k] + delta * step

        refined.append((px + best_offset * dirx, py + best_offset * diry))

    return np.asarray(refined, dtype=np.float64)
