# Glass Bead Measurement

Calibrated photographic metrology for glass beads. From a single photo it
measures, for every bead in frame:

- **Outer Diameter (OD)** in mm
- **Hole Diameter (ID)** in mm
- **Wall Thickness** in mm
- **Roundness** (%)
- **Measurement Confidence** (%)
- an **annotated image** and a downloadable **Excel/CSV report**

The pipeline is built around one non-negotiable rule: **no calibration, no
measurement.** Every dimension is derived from a known-size reference placed in
the photo; nothing is ever guessed from pixels alone.

---

## A note on accuracy (please read)

A true **±0.01 mm (10 µm)** result from a phone photo is **only physically
meaningful when a high-quality calibration reference is in the frame** (an ArUco
marker, a calibration circle, a gauge block, or a ruler scale). Even then it
depends on focus, lighting, and the bead filling enough pixels.

In realistic conditions — good top-down lighting, a sharp 48 MP macro shot, and
a clean calibration marker — expect **practical accuracy of about 0.02–0.05 mm**.
This toolkit is engineered to get as close to that floor as the optics allow:

- **Sub-pixel edge localisation** (`subpixel.py`) walks an intensity profile
  across every edge and finds the gradient peak by parabolic interpolation, so
  edges are located to a fraction of a pixel rather than the nearest whole one.
- **Robust algebraic circle fitting** (`circle_fit.py`, Taubin + RANSAC) is
  unbiased and rejects outliers, unlike min-enclosing-circle approaches.
- On clean synthetic scenes (known ground truth) the pipeline measures OD to
  within **~20–30 µm** and ID to within **~10 µm**.

When the image cannot support a reliable measurement, the tool says so rather
than returning a confident wrong number:

> _"Measurement not reliable. Please retake photo."_

---

## Install

```bash
pip install -r requirements.txt
```

> `cv2.aruco` requires the **contrib** OpenCV build
> (`opencv-contrib-python`, or `opencv-contrib-python-headless` on a server).

## Quick start

No photo handy? Run the built-in synthetic demo (renders a calibrated 2-bead
scene with known ground truth and measures it):

```bash
python cli.py --demo --out annotated.png --excel report.xlsx
```

Measure a real photo with a 10 mm ArUco calibration marker:

```bash
python cli.py photo.jpg --method aruco --marker-mm 10 \
    --out annotated.png --excel report.xlsx
```

With a printed calibration circle (e.g. a 5 mm dot):

```bash
python cli.py photo.jpg --method circle --circle-mm 5
```

Manual scale (you measured 150 px/mm some other way):

```bash
python cli.py photo.jpg --method manual --px-per-mm 150
```

## Web app

```bash
streamlit run app.py
```

Upload a photo, choose the calibration reference, and read the per-bead results,
annotated overlay, and downloadable reports.

---

## How it works

The pipeline (`glass_bead/pipeline.py`) runs the spec workflow in order, and any
hard gate failing stops the run with an explicit reason:

1. **Quality gate** (`quality.py`) — rejects blurry, glary, or under-resolved
   photos. Blur is measured as variance-of-Laplacian **on edge pixels only**, so
   a large white background doesn't masquerade as "blur".
2. **Calibration** (`calibration.py`) — detects the reference and computes
   `pixels_per_mm`. Missing reference ⇒ hard stop.
3. **Bead detection** (`detection.py`) — Otsu segmentation + contour hierarchy;
   the outer contour is the bead, a nested contour is the hole (Hough-circle
   fallback if the hole is faint). The calibration reference region is excluded.
4. **Sub-pixel circle fitting** (`subpixel.py`, `circle_fit.py`) — refine edges,
   then fit outer and hole circles.
5. **Measurement** (`measurement.py`) — OD, ID, wall thickness, roundness.
6. **Confidence** (`confidence.py`) — a conservative weighted-geometric-mean of
   fit residual, roundness, resolution, edge contrast, and calibration quality.
   Beads below the threshold (default 90%) are flagged as not reliable.
7. **Annotation + report** (`visualization.py`, `report.py`).

### Module map

| File | Responsibility |
|------|----------------|
| `glass_bead/quality.py` | Blur / glare / resolution gate |
| `glass_bead/calibration.py` | ArUco / circle / manual → px-per-mm |
| `glass_bead/detection.py` | Bead + hole segmentation |
| `glass_bead/subpixel.py` | Sub-pixel edge refinement |
| `glass_bead/circle_fit.py` | Taubin + RANSAC circle fitting |
| `glass_bead/measurement.py` | Per-bead dimensions |
| `glass_bead/confidence.py` | Confidence scoring |
| `glass_bead/visualization.py` | Annotated output |
| `glass_bead/report.py` | Excel / CSV reports |
| `glass_bead/pipeline.py` | Orchestration / entry point |
| `glass_bead/synthetic.py` | Ground-truth test scenes |

## Tests

```bash
pytest tests/ -v
```

The tests render synthetic scenes with **exactly known** dimensions and assert
the measured OD/ID error stays within tolerance, plus that the quality and
calibration gates reject bad input.

---

## iPhone capture guide (iPhone 17 Pro)

- **Macro mode ON**, **48 MP** capture
- **Top-down** view, lens parallel to the tray (perspective tilt = scale error)
- **Plain white** matte background
- **Even, diffuse LED** lighting — avoid hard specular glints on the glass
- **Keep the calibration reference in frame** (ArUco marker or printed circle)
- Each bead should fill **≥ 300 px** in diameter for full confidence

A printable A4 calibration sheet with ArUco markers can be generated from any
ArUco library; a `DICT_4X4_50` marker at a precisely known side length is ideal.

---

## Industrial QC roadmap

The current tool is a single-photo measurement engine. To reach a production
line solution for automatically grading 1.7–4 mm beads, the natural extensions
are:

1. **Fixed rig** — camera on a copy-stand at fixed working distance with a
   bead tray and a permanent calibration marker, so scale is stable shot-to-shot.
2. **Telecentric or controlled optics** — removes perspective/magnification
   error and pushes accuracy toward the 10 µm floor.
3. **Batch + database** — per-bead records, pass/fail against a spec window,
   trend charts, and aggregated Excel/PDF QC reports (the `report.py` schema is
   already structured for this).
4. **Optional YOLOv8 detector** — swap the classical segmenter for a trained
   model when beads sit on cluttered or coloured backgrounds.

The classical pipeline here needs no GPU and runs on any laptop; the YOLO path
is an optional accuracy/robustness upgrade, not a requirement.
