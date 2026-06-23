"""Streamlit UI for glass bead measurement.

Run with::

    streamlit run app.py

Upload a calibrated photo, pick the calibration reference you placed in frame,
and the app returns per-bead OD / ID / wall thickness / roundness / confidence,
an annotated image, and a downloadable Excel/CSV report.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from glass_bead import measure_image
from glass_bead.report import build_dataframe, write_excel
from glass_bead.synthetic import default_demo_scene

st.set_page_config(page_title="Glass Bead Metrology", layout="wide")


def _read_upload(uploaded) -> np.ndarray:
    data = np.frombuffer(uploaded.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    st.title("🔬 Glass Bead Measurement")
    st.caption(
        "Calibrated photographic metrology for glass bead outer/hole diameter. "
        "A known reference (ArUco marker, calibration circle, or manual scale) "
        "is required — measurements are never estimated without one."
    )

    with st.sidebar:
        st.header("Calibration")
        method = st.selectbox(
            "Reference type",
            ["aruco", "circle", "manual"],
            help="What known-size reference is in the photo?",
        )
        marker_mm = circle_mm = px_per_mm = None
        if method == "aruco":
            marker_mm = st.number_input(
                "ArUco marker side (mm)", value=10.0, min_value=0.1, step=0.5
            )
        elif method == "circle":
            circle_mm = st.number_input(
                "Calibration circle diameter (mm)", value=5.0,
                min_value=0.1, step=0.5,
            )
        else:
            px_per_mm = st.number_input(
                "Pixels per mm", value=150.0, min_value=1.0, step=1.0
            )

        st.header("Bead size band")
        min_mm = st.number_input("Min outer diameter (mm)", value=1.0, step=0.1)
        max_mm = st.number_input("Max outer diameter (mm)", value=8.0, step=0.1)
        min_conf = st.slider("Min confidence to accept (%)", 50, 99, 90)
        enforce_quality = st.checkbox("Enforce blur/glare quality gate", value=True)

        st.divider()
        use_demo = st.button("Load synthetic demo scene")

    st.markdown(
        "**iPhone 17 Pro capture tips:** Macro mode ON · 48 MP · white "
        "background · top-down view · even LED lighting · keep the calibration "
        "reference in frame · each bead ≥ 300 px across."
    )

    uploaded = st.file_uploader(
        "Upload bead photo", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
    )

    image = None
    source_name = "uploaded-image"
    if use_demo:
        image, gt = default_demo_scene()
        source_name = "synthetic-demo"
        if method == "aruco":
            marker_mm = gt["marker_length_mm"]
        st.info("Loaded a synthetic 2-bead demo scene (OD 2.5 / 3.8 mm).")
    elif uploaded is not None:
        image = _read_upload(uploaded)
        source_name = uploaded.name

    if image is None:
        st.stop()

    with st.spinner("Measuring…"):
        result = measure_image(
            image,
            calibration_method=method,
            marker_length_mm=marker_mm,
            diameter_mm=circle_mm,
            pixels_per_mm=px_per_mm,
            min_diameter_mm=min_mm,
            max_diameter_mm=max_mm,
            min_confidence=float(min_conf),
            enforce_quality=enforce_quality,
        )

    col_img, col_res = st.columns([3, 2])

    with col_img:
        if result.annotated is not None:
            st.image(_bgr_to_rgb(result.annotated),
                     caption="Annotated measurement", use_container_width=True)
        else:
            st.image(_bgr_to_rgb(image), use_container_width=True)

    with col_res:
        if not result.ok:
            st.error(f"Measurement not reliable. {result.message}")
            with st.expander("Image quality detail"):
                st.json(result.quality.as_dict())
            st.stop()

        st.success(result.message)
        st.json(result.calibration.as_dict())

        for m in result.beads:
            badge = "✅" if m.reliable else "⚠️"
            with st.container(border=True):
                st.subheader(f"{badge} Bead #{m.index}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Outer Ø", f"{m.outer_diameter_mm:.2f} mm")
                if m.hole_diameter_mm is not None:
                    c2.metric("Hole Ø", f"{m.hole_diameter_mm:.2f} mm")
                if m.wall_thickness_mm is not None:
                    c3.metric("Wall", f"{m.wall_thickness_mm:.2f} mm")
                c4, c5 = st.columns(2)
                c4.metric("Roundness", f"{m.roundness_pct:.1f}%")
                c5.metric("Confidence", f"{m.confidence_pct:.0f}%")
                if m.notes:
                    st.caption(" · ".join(m.notes))

    st.divider()
    df = build_dataframe(result)
    st.dataframe(df, use_container_width=True)

    # Downloadable reports.
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv_bytes, file_name="bead_report.csv",
                       mime="text/csv")

    with tempfile.TemporaryDirectory() as td:
        xlsx_path = Path(td) / "bead_report.xlsx"
        write_excel(result, xlsx_path, source_name=source_name)
        st.download_button(
            "Download Excel",
            xlsx_path.read_bytes(),
            file_name="bead_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    if result.annotated is not None:
        ok, buf = cv2.imencode(".png", result.annotated)
        if ok:
            st.download_button("Download annotated image", buf.tobytes(),
                               file_name="annotated.png", mime="image/png")


if __name__ == "__main__":
    main()
