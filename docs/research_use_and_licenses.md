# Research Use, Open Source, and Provenance

CytoTrack AI source code is licensed under MIT. The default paper-facing
pipeline is designed to run without non-commercial model backends:

1. classical microscopy detector and segmentation/contour repair,
2. centroid-based tracking of the cell center,
3. self-repair curators and frame-level QC,
4. migration metrics, plots, dashboards, and videos in `RESULT/`.

## What is safe for default paper results

- CytoTrack AI project source: MIT.
- Local result-generation scripts: MIT project code.
- Default detector/tracker/analyzer path: local project code plus common
  open-source scientific Python packages.
- Cell Tracking Challenge source movies/masks: external research dataset.
  Cite the Cell Tracking Challenge and obey its dataset terms.

## Components requiring extra care

- `requirements-locate.txt` enables NVIDIA LocateAnything-3B. This is
  optional and opt-in because the model license is non-commercial. Do not use
  it for commercial outputs, and record its use in the result manifest.
- LIVECell data/models are CC BY-NC 4.0. Do not redistribute LIVECell images
  in this repository. If users train from LIVECell, cite LIVECell and keep
  the run non-commercial unless separately licensed.
- PyQt5 is GPL/commercial. Source-code research use is open, but binary
  redistribution with PyQt5 must comply with GPL or use a commercial license.

## Result-folder requirements

For paper figures or quantitative tables, keep the exact `RESULT/<run>/`
folder with:

- `manifest.json` or equivalent run settings,
- input movie/dataset name and sequence,
- detector backend and model/training state,
- cell-line color mapping,
- tracking video and dashboard,
- migration CSVs,
- QC/coverage audit files,
- citation and license notes.

Do not move web research reports into `RESULT/` unless they are direct
run provenance. Architecture research belongs in `docs/`.
