#!/usr/bin/env python3
"""Generate long-duration validation videos from real CTC movies.

The source movies in this repo are short (84 frames for HeLa, 30 frames for
Huh7). This tool creates longer validation movies by ping-ponging the real
time series: 0..N-1..1..0..N-1... . That avoids a hard jump from the final
frame back to the first frame while still using only real images and masks.

Outputs are written under RESULT/long_video_validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
TOOLS_DIR = os.path.join(REPO_ROOT, "tools")
for path in (SRC_DIR, TOOLS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from analyzer import MigrationAnalyzer  # noqa: E402
from generate_small_gt_results import (  # noqa: E402
    CELL_LINE_COLORS,
    _color_for,
    _draw_hud,
    _frame_number,
    _image_paths,
    _label_geometry,
    _load_masks,
    _normalize_to_bgr,
)
from visualizer import TrajectoryVisualizer  # noqa: E402


Point = Tuple[int, int]


def _pingpong_source_indices(source_frames: Sequence[int],
                             target_frames: int) -> List[int]:
    if not source_frames:
        return []
    if len(source_frames) == 1:
        return [source_frames[0]] * target_frames
    period = len(source_frames) * 2 - 2
    out = []
    for i in range(target_frames):
        r = i % period
        idx = r if r < len(source_frames) else period - r
        out.append(source_frames[idx])
    return out


def _all_labels(masks: Dict[int, np.ndarray]) -> List[int]:
    labels = set()
    for mask in masks.values():
        labels.update(int(x) for x in np.unique(mask) if int(x) > 0)
    return sorted(labels)


def _draw_frame(vis: np.ndarray, mask: np.ndarray, dataset: str,
                selected_ids: Sequence[int],
                trails: Dict[int, List[Tuple[int, Point]]],
                virtual_frame: int) -> int:
    active = 0
    for label in selected_ids:
        geom = _label_geometry(mask, label)
        if geom is None:
            continue
        active += 1
        color = _color_for(label, dataset=dataset, color_by_cell_line=True)
        binary = np.uint8(mask == label) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            overlay = vis.copy()
            cv2.drawContours(overlay, [contour], -1, color, -1)
            vis[:] = cv2.addWeighted(overlay, 0.18, vis, 0.82, 0)
            cv2.drawContours(vis, [contour], -1, color, 1)
        cx = int(round(geom["centroid"][0]))
        cy = int(round(geom["centroid"][1]))
        trails[label].append((virtual_frame, (cx, cy)))
        for (pf, pp), (cf, cp) in zip(trails[label][-120:],
                                      trails[label][-119:]):
            if cf == pf + 1:
                cv2.line(vis, pp, cp, color, 1)
        cv2.circle(vis, (cx, cy), 3, color, -1)
        cv2.circle(vis, (cx, cy), 5, (0, 0, 0), 1)
    return active


def _extract_virtual_tracks(source_map: Sequence[int],
                            masks: Dict[int, np.ndarray],
                            selected_ids: Sequence[int]) -> Tuple[Dict[int, dict], List[dict]]:
    tracks: Dict[int, dict] = {}
    rows: List[dict] = []
    for label in selected_ids:
        boxes = []
        frames = []
        for virtual_frame, source_frame in enumerate(source_map):
            geom = _label_geometry(masks[source_frame], label)
            if geom is None:
                continue
            cx, cy = geom["centroid"]
            boxes.append((float(cx), float(cy), 0.0, 0.0))
            frames.append(virtual_frame)
            x, y, w, h = geom["bbox"]
            rows.append({
                "frame": virtual_frame,
                "source_frame": source_frame,
                "track_id": label,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "centroid_x": cx,
                "centroid_y": cy,
                "area": geom["area"],
            })
        if boxes:
            tracks[label] = {
                "boxes": boxes,
                "frames": frames,
                "cell_type": "Cell",
            }
    return tracks, rows


def _write_identity_qc(output_dir: str, source_map: Sequence[int],
                       masks: Dict[int, np.ndarray],
                       rows: Sequence[dict]) -> dict:
    by_track: Dict[int, List[dict]] = defaultdict(list)
    frame_rows = []
    for row in rows:
        mask = masks[int(row["source_frame"])]
        h, w = mask.shape[:2]
        cx = float(row["centroid_x"])
        cy = float(row["centroid_y"])
        px = min(max(int(round(cx)), 0), w - 1)
        py = min(max(int(round(cy)), 0), h - 1)
        label_at_centroid = int(mask[py, px])
        ok = label_at_centroid == int(row["track_id"])
        item = {
            "frame": int(row["frame"]),
            "source_frame": int(row["source_frame"]),
            "track_id": int(row["track_id"]),
            "centroid_x": cx,
            "centroid_y": cy,
            "label_at_centroid": label_at_centroid,
            "centroid_on_own_mask": ok,
        }
        frame_rows.append(item)
        by_track[int(row["track_id"])].append(item)

    summary_rows = []
    total_identity_errors = 0
    total_large_jumps = 0
    for track_id, items in sorted(by_track.items()):
        items = sorted(items, key=lambda r: r["frame"])
        per_frame_steps = []
        for prev, curr in zip(items, items[1:]):
            gap = max(1, curr["frame"] - prev["frame"])
            dist = float(np.hypot(curr["centroid_x"] - prev["centroid_x"],
                                  curr["centroid_y"] - prev["centroid_y"]))
            per_frame_steps.append(dist / gap)
        max_step = max(per_frame_steps) if per_frame_steps else 0.0
        med_step = float(np.median(per_frame_steps)) if per_frame_steps else 0.0
        threshold = max(30.0, med_step * 6.0)
        large_jumps = sum(1 for step in per_frame_steps if step > threshold)
        identity_errors = sum(1 for item in items if not item["centroid_on_own_mask"])
        total_identity_errors += identity_errors
        total_large_jumps += large_jumps
        summary_rows.append({
            "track_id": track_id,
            "frames_checked": len(items),
            "centroid_identity_errors": identity_errors,
            "large_step_jumps": large_jumps,
            "max_step_px_per_frame": max_step,
            "median_step_px_per_frame": med_step,
            "large_step_threshold_px": threshold,
            "passed": identity_errors == 0,
        })

    with open(os.path.join(output_dir, "frame_identity_qc.csv"), "w",
              encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame", "source_frame", "track_id", "centroid_x", "centroid_y",
            "label_at_centroid", "centroid_on_own_mask",
        ])
        writer.writeheader()
        writer.writerows(frame_rows)
    with open(os.path.join(output_dir, "identity_quality_report.csv"), "w",
              encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "track_id", "frames_checked", "centroid_identity_errors",
            "large_step_jumps", "max_step_px_per_frame",
            "median_step_px_per_frame", "large_step_threshold_px", "passed",
        ])
        writer.writeheader()
        writer.writerows(summary_rows)
    return {
        "tracks_checked": len(summary_rows),
        "frame_checks": len(frame_rows),
        "centroid_identity_errors": total_identity_errors,
        "large_step_warning_count": total_large_jumps,
        "large_step_note": (
            "Large centroid steps are warnings for visual review. They do "
            "not fail identity QC when the centroid remains on the same "
            "manual instance label."
        ),
        "passed": total_identity_errors == 0,
    }


def _write_coverage_audit(output_dir: str, source_map: Sequence[int],
                          masks: Dict[int, np.ndarray],
                          selected_ids: Sequence[int]) -> dict:
    selected = set(int(x) for x in selected_ids)
    rows = []
    for virtual_frame, source_frame in enumerate(source_map):
        labels = {int(x) for x in np.unique(masks[source_frame]) if int(x) > 0}
        missing = labels - selected
        rows.append({
            "frame": virtual_frame,
            "source_frame": source_frame,
            "labels_present": len(labels),
            "labels_tracked": len(labels & selected),
            "labels_missing": len(missing),
            "missing_ids": " ".join(str(x) for x in sorted(missing)),
        })
    with open(os.path.join(output_dir, "all_cell_coverage_audit.csv"),
              "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame", "source_frame", "labels_present", "labels_tracked",
            "labels_missing", "missing_ids",
        ])
        writer.writeheader()
        writer.writerows(rows)
    return {
        "frames": len(rows),
        "max_missing_per_frame": max(r["labels_missing"] for r in rows),
        "frames_with_missing": sum(1 for r in rows if r["labels_missing"] > 0),
        "mean_labels_present_per_frame": float(np.mean([r["labels_present"] for r in rows])),
        "mean_labels_tracked_per_frame": float(np.mean([r["labels_tracked"] for r in rows])),
        "passed": all(r["labels_missing"] == 0 for r in rows),
    }


def generate_long_bundle(root: str, dataset: str, sequence: str,
                         output_root: str, target_frames: int,
                         fps: float, pixel_size: float,
                         time_per_frame: float) -> str:
    image_paths = _image_paths(root, dataset, sequence)
    source_frames = [_frame_number(path) for path in image_paths]
    masks = _load_masks(root, dataset, sequence, source_frames)
    source_map = _pingpong_source_indices(source_frames, target_frames)
    selected_ids = _all_labels(masks)

    out_dir = os.path.join(
        output_root,
        f"{dataset}_{sequence}_long{target_frames}_all_cells",
    )
    os.makedirs(out_dir, exist_ok=True)

    first = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED)
    first_vis = _normalize_to_bgr(first)
    height, width = first_vis.shape[:2]
    avi_path = os.path.join(out_dir, "tracking_video.avi")
    mp4_path = os.path.join(out_dir, "tracking_video.mp4")
    writers = [
        cv2.VideoWriter(avi_path, cv2.VideoWriter_fourcc(*"XVID"),
                        float(fps), (width, height + 96)),
        cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"),
                        float(fps), (width, height + 96)),
    ]
    if not all(writer.isOpened() for writer in writers):
        raise RuntimeError(f"could not open video writer in {out_dir}")

    image_by_frame = {frame: path for frame, path in zip(source_frames, image_paths)}
    trails: Dict[int, List[Tuple[int, Point]]] = defaultdict(list)
    for virtual_frame, source_frame in enumerate(source_map):
        image = cv2.imread(image_by_frame[source_frame], cv2.IMREAD_UNCHANGED)
        vis = _normalize_to_bgr(image)
        active = _draw_frame(
            vis, masks[source_frame], dataset, selected_ids, trails, virtual_frame)
        rendered = _draw_hud(
            vis, dataset, sequence, virtual_frame, len(source_map),
            active, len(selected_ids))
        for writer in writers:
            writer.write(rendered)
    for writer in writers:
        writer.release()

    tracks, gt_rows = _extract_virtual_tracks(source_map, masks, selected_ids)
    pd.DataFrame(gt_rows).to_csv(os.path.join(out_dir, "gt_tracks.csv"), index=False)
    analyzer = MigrationAnalyzer(pixel_size, pixel_size, time_per_frame)
    metric_tracks = {
        tid: track for tid, track in tracks.items() if len(track["boxes"]) >= 2
    }
    detailed_df, summary_df = analyzer.analyze(metric_tracks)
    detailed_df.to_csv(os.path.join(out_dir, "migration_detailed.csv"), index=False)
    summary_df.to_csv(os.path.join(out_dir, "migration_summary.csv"), index=False)
    plot_files = []
    if metric_tracks and not summary_df.empty:
        visualizer = TrajectoryVisualizer(pixel_size, pixel_size, time_per_frame)
        plot_files = visualizer.generate_all_plots(
            metric_tracks, detailed_df, summary_df, out_dir)
    qc_summary = _write_identity_qc(out_dir, source_map, masks, gt_rows)
    coverage_summary = _write_coverage_audit(out_dir, source_map, masks, selected_ids)
    manifest = {
        "dataset": dataset,
        "sequence": sequence,
        "software": "CytoTrack AI",
        "software_license": "MIT",
        "detector_backend": "manual-mask-derived validation overlay",
        "optional_noncommercial_backends_used": [],
        "source_frames": len(source_frames),
        "virtual_frames": len(source_map),
        "extension": "pingpong_real_frames",
        "selected_track_ids": selected_ids,
        "cell_line_color_bgr": list(CELL_LINE_COLORS.get(dataset, (0, 255, 255))),
        "coverage": coverage_summary,
        "quality_control": qc_summary,
        "tracked_point": "true mask centroid / cell center",
        "research_use_note": (
            "Generated from labelled CTC-style masks for validation. "
            "Use with the source dataset citation and CytoTrack AI citation."
        ),
        "files": {
            "tracking_video_mp4": "tracking_video.mp4",
            "tracking_video_avi": "tracking_video.avi",
            "coverage_audit": "all_cell_coverage_audit.csv",
            "identity_quality_report": "identity_quality_report.csv",
            "plots": [os.path.basename(path) for path in plot_files],
        },
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    with open(os.path.join(out_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write("Long video validation generated from real CTC frames.\n")
        f.write("Source frames are ping-ponged to avoid loop teleport jumps.\n")
        f.write("All labelled cells are rendered and audited per virtual frame.\n")
    return out_dir


def _write_index(output_root: str, outputs: Sequence[str]) -> None:
    rows = []
    links = []
    for out in outputs:
        with open(os.path.join(out, "manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        rel = os.path.relpath(out, output_root)
        rows.append({
            "folder": rel,
            "dataset": manifest["dataset"],
            "sequence": manifest["sequence"],
            "virtual_frames": manifest["virtual_frames"],
            "source_frames": manifest["source_frames"],
            "coverage_passed": manifest["coverage"]["passed"],
            "identity_qc_passed": manifest["quality_control"]["passed"],
            "video": os.path.join(rel, "tracking_video.mp4"),
        })
        links.append(
            f"<li><a href='{rel}/tracking_video.mp4'>{manifest['dataset']} "
            f"{manifest['sequence']} long {manifest['virtual_frames']} frames</a> "
            f"- coverage: {manifest['coverage']['passed']} "
            f"- identity: {manifest['quality_control']['passed']}</li>"
        )
    with open(os.path.join(output_root, "LONG_VIDEO_VALIDATION_INDEX.csv"),
              "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "folder", "dataset", "sequence", "virtual_frames",
            "source_frames", "coverage_passed", "identity_qc_passed", "video",
        ])
        writer.writeheader()
        writer.writerows(rows)
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Long Video Validation</title>
<style>body{{font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f5f7f8;color:#1e2a2f;margin:0}}main{{max-width:980px;margin:0 auto;padding:24px}}section{{background:white;border:1px solid #d9e1e4;border-radius:8px;padding:14px}}li{{margin:8px 0}}</style>
</head><body><main><h1>Long Video Validation</h1>
<p>Real CTC movies extended with ping-pong temporal sequencing. No synthetic cells are added.</p>
<section><ul>{''.join(links)}</ul></section></main></body></html>"""
    with open(os.path.join(output_root, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.path.join(REPO_ROOT, "real_cell_movies"))
    parser.add_argument("--output-dir", default=os.path.join(
        REPO_ROOT, "RESULT", "long_video_validation"))
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--pixel-size", type=float, default=1.0)
    parser.add_argument("--time-per-frame", type=float, default=60.0)
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--sequence", action="append")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    outputs = []
    datasets = args.dataset or ["DIC-C2DH-HeLa", "Fluo-C2DL-Huh7"]
    sequences = args.sequence or ["02"]
    for dataset in datasets:
        for sequence in sequences:
            print(f"[long-validation] {dataset}/{sequence} -> {args.frames} frames",
                  flush=True)
            out = generate_long_bundle(
                args.root,
                dataset,
                sequence,
                args.output_dir,
                args.frames,
                args.fps,
                args.pixel_size,
                args.time_per_frame,
            )
            outputs.append(out)
            print(f"[long-validation] wrote {out}", flush=True)
    _write_index(args.output_dir, outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
