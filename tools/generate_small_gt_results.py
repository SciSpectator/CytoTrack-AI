#!/usr/bin/env python3
"""Generate small CTC result bundles with at most N labelled cells."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from analyzer import MigrationAnalyzer  # noqa: E402
from visualizer import TrajectoryVisualizer  # noqa: E402


Box = Tuple[int, int, int, int]
Point = Tuple[int, int]

CELL_LINE_COLORS = {
    "DIC-C2DH-HeLa": (220, 60, 190),   # HeLa: magenta
    "Fluo-C2DL-Huh7": (45, 185, 95),   # Huh7: green
    "PhC-C2DH-U373": (238, 134, 45),   # U373: amber/blue contrast
}


def _frame_number(path: str) -> int:
    match = re.search(r"(\d+)", os.path.basename(path))
    if not match:
        raise ValueError(f"could not parse frame number from {path}")
    return int(match.group(1))


def _image_paths(root: str, dataset: str, sequence: str) -> List[str]:
    seq_dir = os.path.join(root, dataset, sequence)
    return sorted(
        os.path.join(seq_dir, name)
        for name in os.listdir(seq_dir)
        if name.lower().endswith((".tif", ".tiff"))
    )


def _mask_path(root: str, dataset: str, sequence: str, frame: int) -> str:
    return os.path.join(root, dataset, f"{sequence}_GT", "TRA",
                        f"man_track{frame:03d}.tif")


def _normalize_to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        if image.dtype == np.uint8:
            return image.copy()
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return image.astype(np.uint8)
    if image.dtype == np.uint8:
        gray = image
    else:
        gray = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _color_for(label: int, dataset: Optional[str] = None,
               color_by_cell_line: bool = False) -> Tuple[int, int, int]:
    if color_by_cell_line and dataset in CELL_LINE_COLORS:
        return CELL_LINE_COLORS[dataset]
    rng = np.random.default_rng(label * 7919)
    return tuple(int(v) for v in rng.integers(70, 255, size=3))


def _load_masks(root: str, dataset: str, sequence: str,
                frames: Sequence[int]) -> Dict[int, np.ndarray]:
    masks = {}
    for frame in frames:
        mask = cv2.imread(_mask_path(root, dataset, sequence, frame),
                          cv2.IMREAD_UNCHANGED)
        if mask is not None:
            masks[frame] = mask
    return masks


def _longest_contiguous_run(frames: Sequence[int]) -> int:
    best = 0
    current = 0
    previous = None
    for frame in sorted(frames):
        if previous is None or frame == previous + 1:
            current += 1
        else:
            current = 1
        best = max(best, current)
        previous = frame
    return best


def _select_longest_tracks(masks: Dict[int, np.ndarray], max_cells: int) -> List[int]:
    by_label: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
    for frame, mask in sorted(masks.items()):
        for label in np.unique(mask):
            label = int(label)
            if label <= 0:
                continue
            geom = _label_geometry(mask, label)
            if geom is None:
                continue
            cx, cy = geom["centroid"]
            by_label[label].append((frame, cx, cy))

    scored = []
    for label, points in by_label.items():
        frames = [p[0] for p in points]
        longest_run = _longest_contiguous_run(frames)
        gap_count = sum(1 for a, b in zip(frames, frames[1:]) if b != a + 1)
        per_frame_steps = []
        for prev, curr in zip(points, points[1:]):
            frame_gap = max(1, curr[0] - prev[0])
            per_frame_steps.append(
                float(np.hypot(curr[1] - prev[1], curr[2] - prev[2])) / frame_gap
            )
        max_step = max(per_frame_steps) if per_frame_steps else 0.0
        median_step = float(np.median(per_frame_steps)) if per_frame_steps else 0.0
        threshold = max(30.0, median_step * 6.0)
        large_jumps = sum(1 for step in per_frame_steps if step > threshold)
        # Quality-first example selection:
        #   1. no large jumps,
        #   2. no/low gaps,
        #   3. longest contiguous tracks,
        #   4. longest total presence,
        #   5. smoother motion.
        #
        # The previous sort put max_step before length, which selected very
        # short low-motion labels in DIC-C2DH-HeLa/02. That made the video
        # look like tracking was disappearing or jumping even though the label
        # identity check passed.
        scored.append((
            large_jumps,
            gap_count,
            -longest_run,
            -len(points),
            max_step,
            label,
        ))

    scored.sort()
    if max_cells <= 0 or max_cells >= len(scored):
        return [label for *_, label in scored]
    clean = [item for item in scored if item[0] == 0]
    if clean:
        return [label for *_, label in clean[:max_cells]]
    return [label for *_, label in scored[:max_cells]]


def _label_geometry(mask: np.ndarray, label: int) -> Optional[dict]:
    binary = np.uint8(mask == label) * 255
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area <= 0:
        return None
    x, y, w, h = cv2.boundingRect(contour)
    moments = cv2.moments(contour)
    if moments["m00"]:
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
    else:
        cx = float(x + w / 2.0)
        cy = float(y + h / 2.0)
    return {
        "bbox": (int(x), int(y), int(w), int(h)),
        "centroid": (cx, cy),
        "area": area,
    }


def _extract_tracks(
    masks: Dict[int, np.ndarray],
    selected_ids: Sequence[int],
) -> Tuple[Dict[int, dict], List[dict]]:
    tracks: Dict[int, dict] = {}
    rows: List[dict] = []
    for label in selected_ids:
        boxes = []
        frame_numbers = []
        for frame in sorted(masks):
            geom = _label_geometry(masks[frame], label)
            if geom is None:
                continue
            cx, cy = geom["centroid"]
            # MigrationAnalyzer derives the tracked point as x + w/2, y + h/2.
            # Store a zero-size box at the true mask centroid so every metric
            # tracks the cell center, not the edge or bounding-box boundary.
            boxes.append((float(cx), float(cy), 0.0, 0.0))
            frame_numbers.append(frame)
            x, y, w, h = geom["bbox"]
            rows.append({
                "frame": frame,
                "track_id": label,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "centroid_x": cx,
                "centroid_y": cy,
                "area": geom["area"],
            })
        if len(boxes) >= 2:
            tracks[label] = {
                "boxes": boxes,
                "frames": frame_numbers,
                "cell_type": "Cell",
            }
        elif len(boxes) == 1:
            # Include single-frame cells in the video/count audit. Migration
            # metrics naturally skip them because velocity/displacement need
            # at least two positions, but dropping them from selected_ids made
            # all-cell videos under-count ground-truth labels.
            tracks[label] = {
                "boxes": boxes,
                "frames": frame_numbers,
                "cell_type": "Cell",
                "single_frame_only": True,
            }
    return tracks, rows


def _draw_selected_tracks(
    vis: np.ndarray,
    mask: np.ndarray,
    selected_ids: Sequence[int],
    trails: Dict[int, List[Tuple[int, Point]]],
    frame: int,
    dataset: str,
    color_by_cell_line: bool,
) -> int:
    active = 0
    for label in selected_ids:
        binary = np.uint8(mask == label) * 255
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) <= 0:
            continue
        active += 1
        color = _color_for(label, dataset=dataset,
                           color_by_cell_line=color_by_cell_line)
        overlay = vis.copy()
        cv2.drawContours(overlay, [contour], -1, color, -1)
        vis[:] = cv2.addWeighted(overlay, 0.22, vis, 0.78, 0)
        cv2.drawContours(vis, [contour], -1, color, 2)

        moments = cv2.moments(contour)
        if moments["m00"]:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = int(x + w / 2), int(y + h / 2)
        trails[label].append((frame, (cx, cy)))
        # Draw only adjacent-frame trail segments. Never connect across
        # missing labels or appearance/disappearance gaps.
        recent = trails[label][-80:]
        for (prev_frame, prev), (curr_frame, curr) in zip(recent, recent[1:]):
            if curr_frame == prev_frame + 1:
                cv2.line(vis, prev, curr, color, 2)
        cv2.circle(vis, (cx, cy), 4, color, -1)
        cv2.circle(vis, (cx, cy), 6, (0, 0, 0), 1)
        x, y, _, _ = cv2.boundingRect(contour)
        cv2.putText(vis, f"ID:{label}", (x, max(12, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        cv2.putText(vis, f"ID:{label}", (x, max(12, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return active


def _draw_hud(vis: np.ndarray, dataset: str, sequence: str, frame_idx: int,
              total: int, active: int, max_cells: int) -> np.ndarray:
    _, width = vis.shape[:2]
    hud_h = 96
    hud = np.full((hud_h, width, 3), (33, 33, 33), dtype=np.uint8)
    title = f"CytoTrack AI small example - {dataset}/{sequence}"
    cv2.putText(hud, title, (14, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (110, 210, 160), 2)
    cv2.putText(hud, f"Frame: {frame_idx + 1}/{total}", (14, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
    cv2.putText(hud, f"Tracked cells: {active}/{max_cells}", (175, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    bar_x0, bar_y = 14, 76
    bar_w = max(1, width - 28)
    cv2.rectangle(hud, (bar_x0, bar_y), (bar_x0 + bar_w, bar_y + 9),
                  (70, 70, 70), -1)
    filled = int(bar_w * (frame_idx + 1) / max(1, total))
    cv2.rectangle(hud, (bar_x0, bar_y), (bar_x0 + filled, bar_y + 9),
                  (110, 210, 160), -1)
    return np.vstack([hud, vis])


def _write_video(root: str, dataset: str, sequence: str, image_paths: Sequence[str],
                 masks: Dict[int, np.ndarray], selected_ids: Sequence[int],
                 output_dir: str, fps: float,
                 color_by_cell_line: bool = False) -> Tuple[str, str]:
    first = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED)
    first_vis = _normalize_to_bgr(first)
    height, width = first_vis.shape[:2]
    avi_path = os.path.join(output_dir, "tracking_video.avi")
    mp4_path = os.path.join(output_dir, "tracking_video.mp4")
    writers = [
        cv2.VideoWriter(avi_path, cv2.VideoWriter_fourcc(*"XVID"),
                        float(fps), (width, height + 96)),
        cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"),
                        float(fps), (width, height + 96)),
    ]
    if not all(w.isOpened() for w in writers):
        raise RuntimeError(f"could not open video writers in {output_dir}")

    trails: Dict[int, List[Tuple[int, Point]]] = defaultdict(list)
    for idx, image_path in enumerate(image_paths):
        frame = _frame_number(image_path)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        mask = masks.get(frame)
        if image is None or mask is None:
            continue
        vis = _normalize_to_bgr(image)
        active = _draw_selected_tracks(
            vis,
            mask,
            selected_ids,
            trails,
            frame,
            dataset,
            color_by_cell_line,
        )
        rendered = _draw_hud(
            vis, dataset, sequence, idx, len(image_paths), active, len(selected_ids))
        for writer in writers:
            writer.write(rendered)
    for writer in writers:
        writer.release()
    return avi_path, mp4_path


def _make_dashboard(
    output_path: str,
    dataset: str,
    sequence: str,
    selected_ids: Sequence[int],
    detailed_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    mp4_name: str,
    avi_name: str,
) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.io as pio
    except Exception:
        return

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Normalized trajectories",
            "Velocity by cell",
            "Displacement vs total path",
            "Directionality / CDE",
        ),
    )
    for track_id, group in detailed_df.groupby("TrackID"):
        x = group["X_displacement_um"]
        y = group["Y_displacement_um"]
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            name=f"ID {track_id}",
            marker=dict(size=4),
            hovertemplate=(
                f"ID {track_id}<br>Frame: %{{customdata}}"
                "<br>X: %{x:.2f} um<br>Y: %{y:.2f} um<extra></extra>"
            ),
            customdata=group["Frame"],
        ), row=1, col=1)

    fig.add_trace(go.Box(
        y=summary_df["Avg_Velocity_um_min"],
        boxpoints="all",
        text=[f"ID {v}" for v in summary_df["TrackID"]],
        name="Velocity",
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=summary_df["Total_Distance_um"],
        y=summary_df["Displacement_um"],
        mode="markers+text",
        text=summary_df["TrackID"].astype(str),
        textposition="top center",
        name="Tracks",
        hovertemplate=(
            "ID %{text}<br>Total: %{x:.2f} um"
            "<br>Displacement: %{y:.2f} um<extra></extra>"
        ),
    ), row=2, col=1)
    fig.add_trace(go.Bar(
        x=[str(v) for v in summary_df["TrackID"]],
        y=summary_df["CDE"],
        name="CDE",
    ), row=2, col=2)
    fig.update_layout(
        template="plotly_white",
        height=760,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.18),
        margin=dict(l=40, r=20, t=70, b=80),
    )
    fig.update_xaxes(title_text="X displacement (um)", row=1, col=1)
    fig.update_yaxes(title_text="Y displacement (um)", row=1, col=1)
    fig.update_yaxes(title_text="Velocity (um/min)", row=1, col=2)
    fig.update_xaxes(title_text="Total distance (um)", row=2, col=1)
    fig.update_yaxes(title_text="Displacement (um)", row=2, col=1)
    fig.update_xaxes(title_text="Track ID", row=2, col=2)
    fig.update_yaxes(title_text="CDE", row=2, col=2, range=[0, 1])

    plot_html = pio.to_html(fig, include_plotlyjs=True, full_html=False)
    kpis = {
        "cells": len(selected_ids),
        "mean_velocity": float(summary_df["Avg_Velocity_um_min"].mean()),
        "mean_displacement": float(summary_df["Displacement_um"].mean()),
        "mean_cde": float(summary_df["CDE"].mean()),
    }
    rows = "\n".join(
        "<tr>"
        f"<td>{int(r.TrackID)}</td>"
        f"<td>{int(r.Frames)}</td>"
        f"<td>{r.Avg_Velocity_um_min:.3f}</td>"
        f"<td>{r.Total_Distance_um:.2f}</td>"
        f"<td>{r.Displacement_um:.2f}</td>"
        f"<td>{r.CDE:.3f}</td>"
        "</tr>"
        for r in summary_df.itertuples()
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CytoTrack Small Example - {dataset}/{sequence}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1d2528;
      --muted: #667277;
      --line: #d9e0e2;
      --panel: #ffffff;
      --bg: #f4f7f8;
      --accent: #26745d;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      padding: 18px 24px 12px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }}
    h1 {{
      margin: 0 0 4px;
      font-size: 22px;
      font-weight: 650;
      letter-spacing: 0;
    }}
    .subtle {{ color: var(--muted); }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 18px;
      display: grid;
      gap: 16px;
    }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }}
    .kpi, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }}
    .kpi .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
    }}
    .kpi .value {{
      margin-top: 4px;
      font-size: 22px;
      font-weight: 650;
    }}
    video {{
      width: 100%;
      max-height: 640px;
      background: #111;
      border-radius: 6px;
    }}
    .video-actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }}
    .video-actions a {{
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px 10px;
      color: var(--accent);
      text-decoration: none;
      background: #f8fbfb;
      font-weight: 600;
    }}
    .video-note {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-variant-numeric: tabular-nums;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 7px 8px;
      text-align: right;
    }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ color: var(--muted); font-weight: 600; }}
    @media (max-width: 760px) {{
      .kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      main {{ padding: 10px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>CytoTrack Small Example - {dataset}/{sequence}</h1>
    <div class="subtle">Selected track IDs: {", ".join(str(i) for i in selected_ids)}</div>
    <div class="subtle">Trajectory metrics use the true cell center/centroid. Edges and masks are visual context only.</div>
  </header>
  <main>
    <section class="kpis">
      <div class="kpi"><div class="label">Cells</div><div class="value">{kpis["cells"]}</div></div>
      <div class="kpi"><div class="label">Mean velocity</div><div class="value">{kpis["mean_velocity"]:.3f} um/min</div></div>
      <div class="kpi"><div class="label">Mean displacement</div><div class="value">{kpis["mean_displacement"]:.2f} um</div></div>
      <div class="kpi"><div class="label">Mean CDE</div><div class="value">{kpis["mean_cde"]:.3f}</div></div>
    </section>
    <section class="panel">
      <video controls preload="metadata">
        <source src="{mp4_name}" type="video/mp4">
        <source src="{avi_name}" type="video/x-msvideo">
        Your browser cannot play this video inline. Use the download/open links below.
      </video>
      <div class="video-actions">
        <a href="{mp4_name}" target="_blank">Open MP4</a>
        <a href="{mp4_name}" download>Download MP4</a>
        <a href="{avi_name}" target="_blank">Open AVI</a>
        <a href="{avi_name}" download>Download AVI</a>
      </div>
      <div class="video-note">
        If inline playback fails, open or download the MP4/AVI file directly.
        Some browsers block local video codecs when dashboard.html is opened from disk.
      </div>
    </section>
    <section class="panel">{plot_html}</section>
    <section class="panel">
      <table>
        <thead>
          <tr><th>Track</th><th>Frames</th><th>Avg velocity</th><th>Total distance</th><th>Displacement</th><th>CDE</th></tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def _write_identity_qc(
    output_dir: str,
    masks: Dict[int, np.ndarray],
    gt_rows: Sequence[dict],
) -> dict:
    frame_rows = []
    by_track: Dict[int, List[dict]] = defaultdict(list)
    for row in gt_rows:
        mask = masks.get(int(row["frame"]))
        if mask is None:
            continue
        h, w = mask.shape[:2]
        cx = float(row["centroid_x"])
        cy = float(row["centroid_y"])
        px = min(max(int(round(cx)), 0), w - 1)
        py = min(max(int(round(cy)), 0), h - 1)
        label_at_centroid = int(mask[py, px])
        ok = label_at_centroid == int(row["track_id"])
        qc = {
            "frame": int(row["frame"]),
            "track_id": int(row["track_id"]),
            "centroid_x": cx,
            "centroid_y": cy,
            "label_at_centroid": label_at_centroid,
            "centroid_on_own_mask": ok,
        }
        frame_rows.append(qc)
        by_track[int(row["track_id"])].append(qc)

    summary_rows = []
    total_identity_errors = 0
    total_large_jumps = 0
    for track_id, rows in sorted(by_track.items()):
        rows = sorted(rows, key=lambda item: item["frame"])
        step_distances = []
        per_frame_steps = []
        for prev, curr in zip(rows, rows[1:]):
            dist = float(np.hypot(curr["centroid_x"] - prev["centroid_x"],
                                  curr["centroid_y"] - prev["centroid_y"]))
            frame_gap = max(1, int(curr["frame"]) - int(prev["frame"]))
            step_distances.append(dist)
            per_frame_steps.append(dist / frame_gap)
        max_step = max(per_frame_steps) if per_frame_steps else 0.0
        median_step = float(np.median(per_frame_steps)) if per_frame_steps else 0.0
        threshold = max(30.0, median_step * 6.0)
        large_jumps = sum(1 for dist in per_frame_steps if dist > threshold)
        identity_errors = sum(1 for row in rows if not row["centroid_on_own_mask"])
        total_identity_errors += identity_errors
        total_large_jumps += large_jumps
        summary_rows.append({
            "track_id": track_id,
            "frames_checked": len(rows),
            "centroid_identity_errors": identity_errors,
            "large_step_jumps": large_jumps,
            "max_step_px": max_step,
            "median_step_px": median_step,
            "large_step_threshold_px": threshold,
            "passed": identity_errors == 0 and large_jumps == 0,
        })

    with open(os.path.join(output_dir, "frame_identity_qc.csv"), "w",
              newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame", "track_id", "centroid_x", "centroid_y",
                "label_at_centroid", "centroid_on_own_mask",
            ],
        )
        writer.writeheader()
        writer.writerows(frame_rows)
    with open(os.path.join(output_dir, "identity_quality_report.csv"), "w",
              newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "track_id", "frames_checked", "centroid_identity_errors",
                "large_step_jumps", "max_step_px", "median_step_px",
                "large_step_threshold_px", "passed",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    return {
        "tracks_checked": len(summary_rows),
        "frame_checks": len(frame_rows),
        "centroid_identity_errors": total_identity_errors,
        "large_step_jumps": total_large_jumps,
        "passed": total_identity_errors == 0 and total_large_jumps == 0,
    }


def _write_manifest(output_dir: str, data: dict) -> None:
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    with open(os.path.join(output_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write("CytoTrack AI small example result bundle\n")
        f.write("========================================\n\n")
        f.write("This folder contains at most 15 labelled cells from one real CTC movie.\n")
        f.write("All velocity, displacement, CDE, MSD, and trajectory metrics track the true cell center/centroid.\n")
        f.write("Cell edges, contours, and bounding boxes are visual context only, not tracked points.\n")
        f.write("Files:\n")
        f.write("- tracking_video.mp4 / tracking_video.avi: overlay video\n")
        f.write("- dashboard.html: interactive local dashboard\n")
        f.write("- migration_detailed.csv: per-frame migration metrics\n")
        f.write("- migration_summary.csv: per-track migration metrics\n")
        f.write("- gt_tracks.csv: selected mask-derived boxes\n")
        f.write("- frame_identity_qc.csv: per-frame centroid identity checks\n")
        f.write("- identity_quality_report.csv: per-track jump/identity checks\n")
        f.write("- plot_*.png / plot_interactive.html: publication plots\n")


def generate_bundle(root: str, dataset: str, sequence: str,
                    output_root: str, args: argparse.Namespace) -> Optional[str]:
    image_paths = _image_paths(root, dataset, sequence)
    if not image_paths:
        return None
    frames = [_frame_number(path) for path in image_paths]
    masks = _load_masks(root, dataset, sequence, frames)
    if not masks:
        return None
    selected_ids = _select_longest_tracks(masks, args.max_cells)
    tracks, gt_rows = _extract_tracks(masks, selected_ids)
    if not tracks:
        return None
    selected_ids = list(tracks)

    suffix = "all_cells" if args.max_cells <= 0 or args.max_cells >= 9999 else f"max{args.max_cells}"
    output_dir = os.path.join(output_root, f"{dataset}_{sequence}_{suffix}")
    os.makedirs(output_dir, exist_ok=True)
    avi_path, mp4_path = _write_video(
        root, dataset, sequence, image_paths, masks, selected_ids,
        output_dir, args.fps, color_by_cell_line=args.color_by_cell_line)

    with open(os.path.join(output_dir, "gt_tracks.csv"), "w",
              newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame", "track_id", "x", "y", "w", "h",
                "centroid_x", "centroid_y", "area",
            ],
        )
        writer.writeheader()
        writer.writerows(gt_rows)
    qc_summary = _write_identity_qc(output_dir, masks, gt_rows)

    analyzer = MigrationAnalyzer(args.pixel_size, args.pixel_size, args.time_per_frame)
    detailed_df, summary_df = analyzer.analyze(tracks)
    detailed_df.to_csv(os.path.join(output_dir, "migration_detailed.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, "migration_summary.csv"), index=False)

    visualizer = TrajectoryVisualizer(args.pixel_size, args.pixel_size, args.time_per_frame)
    plot_files = visualizer.generate_all_plots(tracks, detailed_df, summary_df, output_dir)
    _make_dashboard(
        os.path.join(output_dir, "dashboard.html"),
        dataset,
        sequence,
        selected_ids,
        detailed_df,
        summary_df,
        os.path.basename(mp4_path),
        os.path.basename(avi_path),
    )
    manifest = {
        "dataset": dataset,
        "sequence": sequence,
        "source_root": root,
        "max_cells": args.max_cells,
        "all_cells_mode": args.max_cells <= 0 or args.max_cells >= 9999,
        "selected_track_ids": selected_ids,
        "frames": len(image_paths),
        "pixel_size_um": args.pixel_size,
        "time_per_frame_sec": args.time_per_frame,
        "tracked_point": (
            "true mask centroid / cell center; contours and boxes are visual "
            "context only"
        ),
        "cell_line_color_bgr": list(CELL_LINE_COLORS.get(dataset, (0, 255, 255)))
        if args.color_by_cell_line else None,
        "files": {
            "tracking_video_avi": os.path.basename(avi_path),
            "tracking_video_mp4": os.path.basename(mp4_path),
            "dashboard": "dashboard.html",
            "detailed_csv": "migration_detailed.csv",
            "summary_csv": "migration_summary.csv",
            "gt_tracks_csv": "gt_tracks.csv",
            "frame_identity_qc_csv": "frame_identity_qc.csv",
            "identity_quality_report_csv": "identity_quality_report.csv",
            "plots": [os.path.basename(path) for path in plot_files],
        },
        "quality_control": qc_summary,
    }
    _write_manifest(output_dir, manifest)
    return output_dir


def _discover(root: str, datasets: Optional[Iterable[str]],
              sequences: Optional[Iterable[str]]) -> List[Tuple[str, str]]:
    dataset_names = list(datasets) if datasets else [
        name for name in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, name))
    ]
    sequence_names = list(sequences) if sequences else ["01", "02"]
    pairs = []
    for dataset in dataset_names:
        for sequence in sequence_names:
            if os.path.isdir(os.path.join(root, dataset, f"{sequence}_GT", "TRA")):
                pairs.append((dataset, sequence))
    return pairs


def _write_result_root_index(output_root: str, outputs: Sequence[str]) -> None:
    rows = []
    links = []
    for out in outputs:
        manifest_path = os.path.join(out, "manifest.json")
        if not os.path.isfile(manifest_path):
            continue
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        rel = os.path.relpath(out, output_root)
        qc_passed = bool(manifest.get("quality_control", {}).get("passed", False))
        rows.append({
            "dataset": manifest["dataset"],
            "sequence": manifest["sequence"],
            "folder": rel,
            "dashboard": os.path.join(rel, "dashboard.html"),
            "video": os.path.join(rel, "tracking_video.mp4"),
            "cells": len(manifest["selected_track_ids"]),
            "qc_passed": qc_passed,
        })
        links.append(
            f"<li><a href='{rel}/dashboard.html'>{manifest['dataset']} "
            f"{manifest['sequence']}</a> - {len(manifest['selected_track_ids'])} "
            f"cells - QC passed: {qc_passed}</li>"
        )

    with open(os.path.join(output_root, "RESULT_INDEX.csv"), "w",
              newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset", "sequence", "folder", "dashboard", "video",
                "cells", "qc_passed",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(os.path.join(output_root, "README.txt"), "w", encoding="utf-8") as f:
        f.write("CytoTrack AI RESULT folder\n")
        f.write("==========================\n\n")
        f.write("All generated deliverables are kept inside this RESULT folder.\n")
        f.write("Each example subfolder contains video, dashboard, CSV metrics, plots, QC reports, and manifest.\n")
        f.write("Tracking metrics use cell center/centroid points, not cell edges.\n")
        f.write("Quality is prioritized over speed; frame-by-frame identity QC files are included.\n")

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CytoTrack AI RESULT</title>
  <style>
    body {{ margin: 0; font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f4f7f8; color: #1d2528; }}
    main {{ max-width: 980px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 26px; }}
    .panel {{ background: #fff; border: 1px solid #d9e0e2; border-radius: 8px; padding: 16px; }}
    li {{ margin: 8px 0; }}
  </style>
</head>
<body>
  <main>
    <h1>CytoTrack AI RESULT</h1>
    <p>All results are stored here. Metrics track the true cell center/centroid, not edges.</p>
    <section class="panel">
      <ul>{''.join(links)}</ul>
    </section>
  </main>
</body>
</html>
"""
    with open(os.path.join(output_root, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build small result bundles with at most N labelled cells.")
    parser.add_argument("--root", default=os.path.join(REPO_ROOT, "real_cell_movies"))
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--sequence", action="append")
    parser.add_argument("--output-dir", default=os.path.join(
        REPO_ROOT, "RESULT"))
    parser.add_argument("--max-cells", type=int, default=15)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--pixel-size", type=float, default=1.0)
    parser.add_argument("--time-per-frame", type=float, default=60.0)
    parser.add_argument("--color-by-cell-line", action="store_true",
                        help="Use one fixed color per cell line/dataset.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    outputs = []
    for dataset, sequence in _discover(args.root, args.dataset, args.sequence):
        print(f"[small-example] {dataset}/{sequence}", flush=True)
        out = generate_bundle(args.root, dataset, sequence, args.output_dir, args)
        if out:
            outputs.append(out)
            print(f"[small-example] wrote {out}", flush=True)
    print("Generated small result bundles:", flush=True)
    for out in outputs:
        print(out, flush=True)
    _write_result_root_index(args.output_dir, outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
