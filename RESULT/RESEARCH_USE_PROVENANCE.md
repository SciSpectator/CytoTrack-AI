# RESULT Folder Research Provenance

This folder contains generated CytoTrack AI outputs intended for inspection,
validation, and paper-methods reporting.

## Software License

CytoTrack AI project code: MIT License, see `../LICENSE`.

## Default Backend Policy

Paper-facing default results use the open classical detector and centroid
tracking path. Optional non-commercial model backends, including NVIDIA
LocateAnything-3B, are not part of the default path and must be explicitly
declared in a run manifest if used.

## Data Provenance

Current generated validation bundles use Cell Tracking Challenge-style source
movies and manual masks stored under `../real_cell_movies/`. The output videos
draw the center of each labelled cell, not the cell edge, and include border
overlays for visual checking.

## Required Citations

When using these outputs in a manuscript, cite:

- CytoTrack AI, using `../CITATION.cff`.
- The source dataset or benchmark for each input movie.
- Any optional model or external training dataset listed in the run manifest.

## Generated Result Contract

Each completed run should include:

- `tracking_video.mp4` and/or `tracking_video.avi`,
- migration CSV files,
- plots and/or dashboard HTML,
- `manifest.json`,
- coverage and identity QC/audit CSV files,
- cell-line color mapping.
