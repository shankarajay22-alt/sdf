# Glass Bead Measurement — Browser App

A self-contained, mobile-friendly web version of the measurement tool. It runs
**entirely in the browser** with [OpenCV.js](https://docs.opencv.org/4.x/) —
no server, no install, and **no image ever leaves your phone**.

## Open it on your phone

Once GitHub Pages is enabled for this repo, the app is served at:

```
https://shankarajay22-alt.github.io/sdf/
```

Open that link in **Chrome on mobile** (or any modern browser). The
`.github/workflows/deploy-pages.yml` workflow publishes the `web/` folder
automatically; the first deployment can take a couple of minutes after the
branch is pushed/merged.

> If the URL 404s, open the repo's **Settings → Pages** and confirm the source
> is **GitHub Actions** (the workflow tries to enable this automatically, but
> some org policies require a one-time manual confirmation).

## How to use

1. **Take / choose a photo** — on mobile this opens the camera directly.
2. **Calibrate** (required — nothing is measured without a known scale):
   - **Tap two points** a known distance apart on a ruler in the photo, then
     enter that distance in mm. *(Easiest on a phone.)*
   - **Manual** — type pixels-per-mm if you already know it.
   - **Calibration circle** — auto-detects a solid round dot of known diameter.
3. **Set the bead size range** and confidence threshold.
4. **Measure** — get per-bead OD / hole Ø / wall / roundness / confidence, an
   annotated image, and a CSV download.

## Accuracy note

This browser build downsizes very large photos for phone performance, so it is
best for **field/quick-check use** (roughly 0.05–0.1 mm typical). For the
tightest results (~0.02–0.05 mm) use the full Python pipeline in
`../glass_bead_measurement/`, which keeps full resolution and supports ArUco
calibration markers.

## Run locally

It's just static files — open `index.html` in a browser, or serve the folder:

```bash
cd web
python3 -m http.server 8000
# then visit http://localhost:8000
```
