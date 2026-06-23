"""Excel / CSV report generation for QC records.

Produces a spreadsheet with one row per bead plus a summary header, suitable for
archiving alongside the annotated image as part of a quality-control record.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from .pipeline import MeasurementResult


def build_dataframe(result: "MeasurementResult") -> pd.DataFrame:
    rows = [m.as_dict() for m in result.beads]
    if not rows:
        rows = [{"bead": None, "outer_diameter_mm": None}]
    return pd.DataFrame(rows)


def write_excel(
    result: "MeasurementResult",
    path: str | Path,
    *,
    source_name: str = "",
) -> Path:
    """Write a multi-sheet Excel report. Returns the written path."""
    path = Path(path)
    df = build_dataframe(result)

    summary = {
        "Generated": datetime.now().isoformat(timespec="seconds"),
        "Source image": source_name,
        "Calibration method": result.calibration.method,
        "Pixels per mm": round(result.calibration.pixels_per_mm, 3),
        "Calibration confidence": round(result.calibration.confidence, 3),
        "Beads detected": len(result.beads),
        "Beads reliable": sum(1 for b in result.beads if b.reliable),
        "Image passed QC": result.quality.passed,
    }
    summary_df = pd.DataFrame(
        {"Field": list(summary.keys()), "Value": list(summary.values())}
    )

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        df.to_excel(writer, sheet_name="Measurements", index=False)
    return path


def write_csv(result: "MeasurementResult", path: str | Path) -> Path:
    path = Path(path)
    build_dataframe(result).to_csv(path, index=False)
    return path
