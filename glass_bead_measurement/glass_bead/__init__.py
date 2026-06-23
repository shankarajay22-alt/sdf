"""Glass bead photographic measurement toolkit.

A calibrated computer-vision pipeline that measures the outer diameter, hole
diameter, wall thickness, roundness and a confidence score of glass beads from
a single photograph.  See :func:`glass_bead.pipeline.measure_image` for the
main entry point.
"""

from .calibration import Calibration, CalibrationError, calibrate
from .measurement import BeadMeasurement
from .pipeline import MeasurementResult, measure_image
from .quality import ImageQuality, assess_quality

__version__ = "1.0.0"

__all__ = [
    "Calibration",
    "CalibrationError",
    "calibrate",
    "BeadMeasurement",
    "MeasurementResult",
    "measure_image",
    "ImageQuality",
    "assess_quality",
    "__version__",
]
