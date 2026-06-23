"""Whole-image quality gates.

Before we measure anything we decide whether the photo is even worth measuring.
A blurry frame, a blown-out reflection or an under-resolved bead will silently
corrupt a sub-pixel pipeline, so we reject them up front with explicit reasons.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class ImageQuality:
    blur_score: float          # Variance of Laplacian; higher == sharper.
    reflection_ratio: float    # Fraction of (near-)saturated pixels.
    mean_brightness: float
    width: int
    height: int
    megapixels: float
    passed: bool
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "blur_score": round(self.blur_score, 1),
            "reflection_ratio": round(self.reflection_ratio, 4),
            "mean_brightness": round(self.mean_brightness, 1),
            "resolution": f"{self.width}x{self.height}",
            "megapixels": round(self.megapixels, 1),
            "passed": self.passed,
            "reasons": list(self.reasons),
        }


def edge_focus_score(gray: np.ndarray) -> float:
    """Sharpness measured on edge pixels only.

    A plain variance-of-Laplacian over the whole frame is dominated by the large
    flat (white) background in macro bead shots and reads as "blurry" even when
    the beads are tack-sharp.  Instead we restrict the Laplacian variance to the
    strongest-gradient pixels -- i.e. the actual edges -- which tracks true focus
    quality regardless of how much empty background surrounds the beads.
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    threshold = max(20.0, float(np.percentile(mag, 99.5)))
    mask = mag >= threshold
    if int(mask.sum()) < 50:
        # No real edges to judge; fall back to the global measure.
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap[mask].var())


def assess_quality(
    image_bgr: np.ndarray,
    *,
    min_blur: float = 150.0,
    max_reflection_ratio: float = 0.04,
    min_megapixels: float = 2.0,
    saturation_level: int = 250,
) -> ImageQuality:
    """Evaluate global image quality.

    Parameters
    ----------
    min_blur:
        Minimum acceptable edge-focus score (variance of Laplacian on edge
        pixels).  Sharp macro shots score in the thousands to tens of thousands;
        a few hundred or less usually means motion/defocus blur.
    max_reflection_ratio:
        Maximum tolerated fraction of saturated pixels.  Specular glints on
        glass beads destroy edge contrast, so too many of them is a reject.
    min_megapixels:
        Guards against down-scaled images that cannot support sub-pixel work.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    mp = (w * h) / 1e6

    blur_score = edge_focus_score(gray)
    reflection_ratio = float(np.mean(gray >= saturation_level))
    mean_brightness = float(gray.mean())

    reasons: list[str] = []
    if blur_score < min_blur:
        reasons.append(
            f"Image is too blurry (sharpness {blur_score:.0f} < {min_blur:.0f})."
        )
    if reflection_ratio > max_reflection_ratio:
        reasons.append(
            "Too much reflection/glare "
            f"({reflection_ratio * 100:.1f}% of pixels are blown out)."
        )
    if mp < min_megapixels:
        reasons.append(
            f"Resolution too low ({mp:.1f} MP < {min_megapixels:.1f} MP)."
        )

    return ImageQuality(
        blur_score=blur_score,
        reflection_ratio=reflection_ratio,
        mean_brightness=mean_brightness,
        width=w,
        height=h,
        megapixels=mp,
        passed=len(reasons) == 0,
        reasons=reasons,
    )
