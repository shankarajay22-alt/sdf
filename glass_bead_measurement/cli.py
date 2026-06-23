#!/usr/bin/env python3
"""Command-line interface for glass bead measurement.

Examples
--------
Measure with an ArUco calibration marker (10 mm side)::

    python cli.py photo.jpg --method aruco --marker-mm 10 \
        --out annotated.png --excel report.xlsx

Measure with a known calibration circle (e.g. a 5 mm printed dot)::

    python cli.py photo.jpg --method circle --circle-mm 5

Generate and measure a synthetic demo scene (no photo needed)::

    python cli.py --demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from glass_bead import measure_image
from glass_bead.report import write_csv, write_excel
from glass_bead.synthetic import default_demo_scene


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Measure glass bead dimensions from a calibrated photo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("image", nargs="?", help="Path to the input photograph.")
    p.add_argument("--demo", action="store_true",
                   help="Run on a built-in synthetic scene instead of a file.")
    p.add_argument("--method", default="aruco",
                   choices=["aruco", "circle", "manual"],
                   help="Calibration reference type.")
    p.add_argument("--marker-mm", type=float, default=None,
                   help="ArUco marker side length in mm (method=aruco).")
    p.add_argument("--circle-mm", type=float, default=None,
                   help="Calibration circle diameter in mm (method=circle).")
    p.add_argument("--px-per-mm", type=float, default=None,
                   help="Pixels per mm (method=manual).")
    p.add_argument("--min-mm", type=float, default=1.0,
                   help="Minimum bead outer diameter to accept (mm).")
    p.add_argument("--max-mm", type=float, default=8.0,
                   help="Maximum bead outer diameter to accept (mm).")
    p.add_argument("--min-confidence", type=float, default=90.0,
                   help="Reject beads below this confidence percentage.")
    p.add_argument("--no-quality-gate", action="store_true",
                   help="Skip the blur/glare/resolution rejection step.")
    p.add_argument("--out", default=None, help="Path to save annotated image.")
    p.add_argument("--excel", default=None, help="Path to save an Excel report.")
    p.add_argument("--csv", default=None, help="Path to save a CSV report.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.demo:
        image, gt = default_demo_scene()
        source_name = "synthetic-demo"
        if args.method == "aruco" and args.marker_mm is None:
            args.marker_mm = gt["marker_length_mm"]
    elif args.image:
        image = cv2.imread(args.image)
        if image is None:
            print(f"error: could not read image {args.image!r}", file=sys.stderr)
            return 2
        source_name = Path(args.image).name
    else:
        print("error: provide an image path or --demo", file=sys.stderr)
        return 2

    result = measure_image(
        image,
        calibration_method=args.method,
        marker_length_mm=args.marker_mm,
        diameter_mm=args.circle_mm,
        pixels_per_mm=args.px_per_mm,
        min_diameter_mm=args.min_mm,
        max_diameter_mm=args.max_mm,
        min_confidence=args.min_confidence,
        enforce_quality=not args.no_quality_gate,
    )

    print(result.summary_text())
    print()

    if not result.ok:
        return 1

    if args.out and result.annotated is not None:
        cv2.imwrite(args.out, result.annotated)
        print(f"Annotated image saved to {args.out}")
    if args.excel:
        write_excel(result, args.excel, source_name=source_name)
        print(f"Excel report saved to {args.excel}")
    if args.csv:
        write_csv(result, args.csv)
        print(f"CSV report saved to {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
