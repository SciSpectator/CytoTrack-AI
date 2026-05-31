#!/usr/bin/env python3
"""Train two cell-line morphology prototypes, then compare migration.

This script uses public Cell Tracking Challenge movies already present in
``real_cell_movies``:

* DIC-C2DH-HeLa -> HeLa
* Fluo-C2DL-Huh7 -> Huh7

The morphology model is intentionally stored in ``model_cache``. Only final
tracking/comparison outputs are stored in ``RESULT``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from pipeline_architecture import default_model_cache_root  # noqa: E402


CELL_LINES = {
    "DIC-C2DH-HeLa": {
        "cell_line": "HeLa",
        "color": "#dc3cbe",
        "source": "Cell Tracking Challenge DIC-C2DH-HeLa",
        "condition": "DIC light microscopy migration movie",
    },
    "Fluo-C2DL-Huh7": {
        "cell_line": "Huh7",
        "color": "#2db95f",
        "source": "Cell Tracking Challenge Fluo-C2DL-Huh7",
        "condition": "fluorescence microscopy migration movie",
    },
}


FEATURE_COLUMNS = [
    "area_px",
    "perimeter_px",
    "bbox_w",
    "bbox_h",
    "aspect_ratio",
    "extent",
    "circularity",
    "intensity_mean",
    "intensity_std",
]


@dataclass
class PrototypeModel:
    classes: List[str]
    feature_columns: List[str]
    mean: List[float]
    std: List[float]
    prototypes: Dict[str, List[float]]
    training_accuracy: float
    samples_per_class: Dict[str, int]
    notes: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _frame_number(path: str) -> int:
    match = re.search(r"(\d+)", os.path.basename(path))
    if not match:
        raise ValueError(path)
    return int(match.group(1))


def _image_path(root: str, dataset: str, sequence: str, frame: int) -> str:
    return os.path.join(root, dataset, sequence, f"t{frame:03d}.tif")


def _mask_paths(root: str, dataset: str, sequence: str) -> List[str]:
    tra = os.path.join(root, dataset, f"{sequence}_GT", "TRA")
    return sorted(
        os.path.join(tra, name)
        for name in os.listdir(tra)
        if name.lower().endswith((".tif", ".tiff"))
    )


def _extract_features(image: np.ndarray, mask: np.ndarray, label: int) -> Optional[dict]:
    binary = np.uint8(mask == label) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area <= 0:
        return None
    perimeter = float(cv2.arcLength(contour, True))
    x, y, w, h = cv2.boundingRect(contour)
    if w <= 0 or h <= 0:
        return None
    crop_mask = binary[y:y + h, x:x + w] > 0
    crop = image[y:y + h, x:x + w]
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    pixels = crop[crop_mask]
    if pixels.size == 0:
        pixels = crop.reshape(-1)
    circularity = 0.0 if perimeter <= 0 else float(4.0 * np.pi * area / (perimeter ** 2))
    return {
        "area_px": area,
        "perimeter_px": perimeter,
        "bbox_w": float(w),
        "bbox_h": float(h),
        "aspect_ratio": float(w / max(1, h)),
        "extent": float(area / max(1, w * h)),
        "circularity": circularity,
        "intensity_mean": float(np.mean(pixels)),
        "intensity_std": float(np.std(pixels)),
    }


def collect_training_samples(root: str, sequences: Sequence[str],
                             max_samples_per_class: int) -> pd.DataFrame:
    rows: List[dict] = []
    for dataset, info in CELL_LINES.items():
        count = 0
        for sequence in sequences:
            for mask_path in _mask_paths(root, dataset, sequence):
                if count >= max_samples_per_class:
                    break
                frame = _frame_number(mask_path)
                image = cv2.imread(_image_path(root, dataset, sequence, frame),
                                   cv2.IMREAD_UNCHANGED)
                mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if image is None or mask is None:
                    continue
                for label in np.unique(mask):
                    label = int(label)
                    if label <= 0:
                        continue
                    feats = _extract_features(image, mask, label)
                    if feats is None:
                        continue
                    rows.append({
                        "dataset": dataset,
                        "sequence": sequence,
                        "frame": frame,
                        "track_id": label,
                        "cell_line": info["cell_line"],
                        **feats,
                    })
                    count += 1
                    if count >= max_samples_per_class:
                        break
            if count >= max_samples_per_class:
                break
    if not rows:
        raise RuntimeError("no morphology training samples extracted")
    return pd.DataFrame(rows)


def train_prototypes(samples: pd.DataFrame) -> PrototypeModel:
    x = samples[FEATURE_COLUMNS].astype(float).to_numpy()
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    z = (x - mean) / std
    classes = sorted(samples["cell_line"].unique())
    prototypes: Dict[str, List[float]] = {}
    for cls in classes:
        prototypes[cls] = z[samples["cell_line"].to_numpy() == cls].mean(axis=0).tolist()

    correct = 0
    for row, truth in zip(z, samples["cell_line"]):
        pred = min(
            classes,
            key=lambda cls: float(np.linalg.norm(row - np.asarray(prototypes[cls]))),
        )
        correct += int(pred == truth)
    counts = samples.groupby("cell_line").size().astype(int).to_dict()
    return PrototypeModel(
        classes=classes,
        feature_columns=FEATURE_COLUMNS,
        mean=mean.tolist(),
        std=std.tolist(),
        prototypes=prototypes,
        training_accuracy=float(correct / max(1, len(samples))),
        samples_per_class={str(k): int(v) for k, v in counts.items()},
        notes=[
            "Prototype morphology model trained before migration comparison.",
            "Features come from labelled CTC masks: shape, border geometry, and intensity.",
            "This model is for cell-line morphology/color provenance in the result dashboard.",
        ],
    )


def write_model_cache(samples: pd.DataFrame, model: PrototypeModel,
                      output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    samples.to_csv(os.path.join(output_dir, "training_samples.csv"), index=False)
    with open(os.path.join(output_dir, "morphology_model.json"), "w",
              encoding="utf-8") as f:
        json.dump(model.to_dict(), f, indent=2, sort_keys=True)
    source_research = {
        "purpose": "pre-tracking morphology training for two migration cell lines",
        "sources": [
            {
                "cell_line": "HeLa",
                "dataset": "DIC-C2DH-HeLa",
                "source": "Cell Tracking Challenge",
                "condition": CELL_LINES["DIC-C2DH-HeLa"]["condition"],
            },
            {
                "cell_line": "Huh7",
                "dataset": "Fluo-C2DL-Huh7",
                "source": "Cell Tracking Challenge",
                "condition": CELL_LINES["Fluo-C2DL-Huh7"]["condition"],
            },
            {
                "cell_line": "Huh7",
                "dataset": "LIVECell",
                "source": "Sartorius LIVECell",
                "condition": "phase-contrast morphology reference",
                "license_note": "research/reference only here; CC-BY-NC-4.0 is not auto-used for redistributable training",
            },
        ],
    }
    with open(os.path.join(output_dir, "source_research.json"), "w",
              encoding="utf-8") as f:
        json.dump(source_research, f, indent=2, sort_keys=True)


def _result_folders(result_root: str) -> List[str]:
    out = []
    for name in sorted(os.listdir(result_root)):
        path = os.path.join(result_root, name)
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "manifest.json")):
            out.append(path)
    return out


def build_comparison_dashboard(result_root: str, model_dir: str,
                               output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    summary_frames = []
    detailed_frames = []
    for folder in _result_folders(result_root):
        with open(os.path.join(folder, "manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        dataset = manifest["dataset"]
        if dataset not in CELL_LINES:
            continue
        info = CELL_LINES[dataset]
        seq = str(manifest["sequence"])
        summary = pd.read_csv(os.path.join(folder, "migration_summary.csv"))
        detailed = pd.read_csv(os.path.join(folder, "migration_detailed.csv"))
        for df in (summary, detailed):
            df["Dataset"] = dataset
            df["Sequence"] = seq
            df["Cell_Line"] = info["cell_line"]
            df["Color"] = info["color"]
        summary_frames.append(summary)
        detailed_frames.append(detailed)
    if not summary_frames:
        raise RuntimeError("no generated tracking summaries found in RESULT")

    summary_all = pd.concat(summary_frames, ignore_index=True)
    detailed_all = pd.concat(detailed_frames, ignore_index=True)
    summary_all.to_csv(os.path.join(output_dir, "two_cell_line_migration_summary.csv"),
                       index=False)
    detailed_all.to_csv(os.path.join(output_dir, "two_cell_line_migration_detailed.csv"),
                       index=False)

    with open(os.path.join(model_dir, "morphology_model.json"), encoding="utf-8") as f:
        model = json.load(f)
    with open(os.path.join(output_dir, "morphology_training_used.json"), "w",
              encoding="utf-8") as f:
        json.dump(model, f, indent=2, sort_keys=True)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.io as pio
    except Exception:
        go = None

    if go is not None:
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Average velocity by cell line",
                "Displacement by cell line",
                "Directionality / CDE",
                "Normalized trajectories",
            ),
        )
        for line, group in summary_all.groupby("Cell_Line"):
            color = group["Color"].iloc[0]
            fig.add_trace(go.Box(
                y=group["Avg_Velocity_um_min"],
                name=line,
                marker_color=color,
                boxpoints="all",
            ), row=1, col=1)
            fig.add_trace(go.Box(
                y=group["Displacement_um"],
                name=line,
                marker_color=color,
                boxpoints="all",
                showlegend=False,
            ), row=1, col=2)
            fig.add_trace(go.Box(
                y=group["CDE"],
                name=line,
                marker_color=color,
                boxpoints="all",
                showlegend=False,
            ), row=2, col=1)

        for (line, seq, tid), group in detailed_all.groupby(
                ["Cell_Line", "Sequence", "TrackID"]):
            color = group["Color"].iloc[0]
            fig.add_trace(go.Scatter(
                x=group["X_displacement_um"],
                y=group["Y_displacement_um"],
                mode="lines",
                line=dict(color=color, width=1),
                opacity=0.35,
                name=line,
                legendgroup=line,
                showlegend=False,
                hovertemplate=(
                    f"{line} seq {seq} ID {tid}<br>"
                    "X %{x:.2f} um<br>Y %{y:.2f} um<extra></extra>"
                ),
            ), row=2, col=2)
        fig.update_layout(template="plotly_white", height=820,
                          title="Two Cell-Line Migration After Morphology Training")
        fig.update_yaxes(title_text="um/min", row=1, col=1)
        fig.update_yaxes(title_text="um", row=1, col=2)
        fig.update_yaxes(title_text="CDE", row=2, col=1)
        fig.update_xaxes(title_text="X displacement (um)", row=2, col=2)
        fig.update_yaxes(title_text="Y displacement (um)", row=2, col=2)
        plot_html = pio.to_html(fig, include_plotlyjs=True, full_html=False)
    else:
        plot_html = "<p>Plotly is not available.</p>"

    kpis = summary_all.groupby("Cell_Line").agg(
        cells=("TrackID", "count"),
        mean_velocity=("Avg_Velocity_um_min", "mean"),
        mean_displacement=("Displacement_um", "mean"),
        mean_cde=("CDE", "mean"),
    ).reset_index()
    kpi_rows = "\n".join(
        f"<tr><td><span class='swatch' style='background:{CELL_LINES['DIC-C2DH-HeLa' if r.Cell_Line == 'HeLa' else 'Fluo-C2DL-Huh7']['color']}'></span>{r.Cell_Line}</td>"
        f"<td>{int(r.cells)}</td><td>{r.mean_velocity:.3f}</td>"
        f"<td>{r.mean_displacement:.2f}</td><td>{r.mean_cde:.3f}</td></tr>"
        for r in kpis.itertuples()
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Two Cell-Line Migration Comparison</title>
  <style>
    body {{ margin:0; font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:#f5f7f8; color:#1e2a2f; }}
    main {{ max-width:1280px; margin:0 auto; padding:20px; }}
    h1 {{ margin:0 0 8px; font-size:24px; }}
    .panel {{ background:#fff; border:1px solid #d8e0e3; border-radius:8px; padding:14px; margin:14px 0; }}
    table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }}
    th,td {{ padding:8px; border-bottom:1px solid #e2e8ea; text-align:right; }}
    th:first-child,td:first-child {{ text-align:left; }}
    .swatch {{ display:inline-block; width:12px; height:12px; border-radius:2px; margin-right:8px; vertical-align:-1px; }}
    code {{ background:#edf2f4; padding:2px 5px; border-radius:4px; }}
  </style>
</head>
<body>
<main>
  <h1>Two Cell-Line Migration Comparison</h1>
  <p>Before comparison, morphology prototypes were trained for HeLa and Huh7 from labelled public CTC frames. HeLa is magenta; Huh7 is green.</p>
  <section class="panel">
    <table>
      <thead><tr><th>Cell line</th><th>Tracks</th><th>Mean velocity</th><th>Mean displacement</th><th>Mean CDE</th></tr></thead>
      <tbody>{kpi_rows}</tbody>
    </table>
  </section>
  <section class="panel">{plot_html}</section>
  <section class="panel">
    <p>Training cache: <code>{os.path.relpath(model_dir, REPO_ROOT)}</code></p>
    <p>Downloads: <code>two_cell_line_migration_summary.csv</code>, <code>two_cell_line_migration_detailed.csv</code>, <code>morphology_training_used.json</code></p>
  </section>
</main>
</body>
</html>"""
    with open(os.path.join(output_dir, "dashboard.html"), "w",
              encoding="utf-8") as f:
        f.write(html)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.path.join(REPO_ROOT, "real_cell_movies"))
    parser.add_argument("--result-root", default=os.path.join(REPO_ROOT, "RESULT"))
    parser.add_argument("--model-name", default="hela_huh7_morphology")
    parser.add_argument("--max-samples-per-class", type=int, default=2000)
    parser.add_argument("--sequence", action="append", default=["01", "02"])
    parser.add_argument("--skip-tracking-regeneration", action="store_true",
                        help="Use existing RESULT folders instead of regenerating after training.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache_dir = os.path.join(default_model_cache_root(REPO_ROOT),
                             args.model_name)
    samples = collect_training_samples(
        args.root,
        sequences=args.sequence,
        max_samples_per_class=args.max_samples_per_class,
    )
    model = train_prototypes(samples)
    write_model_cache(samples, model, cache_dir)

    if not args.skip_tracking_regeneration:
        cmd = [
            sys.executable,
            os.path.join(REPO_ROOT, "tools", "generate_small_gt_results.py"),
            "--max-cells", "9999",
            "--output-dir", args.result_root,
            "--color-by-cell-line",
        ]
        print("[tracking] regenerating after morphology training", flush=True)
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    output_dir = os.path.join(args.result_root, "two_cell_line_migration_comparison")
    build_comparison_dashboard(args.result_root, cache_dir, output_dir)
    print(f"[morphology-training] samples={len(samples)} "
          f"accuracy={model.training_accuracy:.3f}")
    print(f"[morphology-training] model={cache_dir}")
    print(f"[migration-comparison] dashboard={os.path.join(output_dir, 'dashboard.html')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
