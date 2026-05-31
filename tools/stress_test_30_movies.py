#!/usr/bin/env python3
"""
Run a 30-clip microscopy tracking stress test.

The clips are deterministic windows derived from local public Cell Tracking
Challenge movies. Outputs are written under RESULT/ and are intentionally
ignored by git.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from analyzer import MigrationAnalyzer
from detector import CellDetector
from qagents import IdentityJumpRepairQAgent
from tracker import CellTracker
from visualizer import TrajectoryVisualizer


IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def numeric_key(path: Path) -> Tuple[int, str]:
    matches = re.findall(r"\d+", path.stem)
    return (int(matches[-1]) if matches else -1, path.name)


def image_files(folder: Path) -> List[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=numeric_key,
    )


def normalize_u8(image: np.ndarray, max_side: int = 384) -> np.ndarray:
    if image is None:
        raise ValueError("image could not be read")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype == np.uint8:
        out = image
    else:
        arr = image.astype(np.float32)
        lo, hi = np.percentile(arr, [1, 99.5])
        if hi <= lo:
            lo, hi = float(arr.min()), float(arr.max())
        if hi <= lo:
            out = np.zeros(arr.shape, dtype=np.uint8)
        else:
            arr = np.clip((arr - lo) * 255.0 / (hi - lo), 0, 255)
            out = arr.astype(np.uint8)
    h, w = out.shape[:2]
    side = max(h, w)
    if max_side > 0 and side > max_side:
        scale = float(max_side) / float(side)
        out = cv2.resize(
            out,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return out


def read_u8(path: Path, max_side: int = 384) -> np.ndarray:
    return normalize_u8(cv2.imread(str(path), cv2.IMREAD_UNCHANGED), max_side=max_side)


def gt_seg_dir(seq_dir: Path) -> Path | None:
    root = seq_dir.parent
    candidate = root / f"{seq_dir.name}_GT" / "SEG"
    return candidate if candidate.exists() else None


def gt_count(mask_dir: Path | None, frame_path: Path) -> int | None:
    if mask_dir is None:
        return None
    idx, _ = numeric_key(frame_path)
    candidates = sorted(mask_dir.glob(f"*{idx:03d}*"))
    if not candidates:
        return None
    mask = cv2.imread(str(candidates[0]), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    return int(len(np.unique(mask[mask > 0])))


def make_manifest(target_clips: int, window: int) -> List[dict]:
    sources = [
        ROOT / "real_cell_movies" / "DIC-C2DH-HeLa" / "01",
        ROOT / "real_cell_movies" / "DIC-C2DH-HeLa" / "02",
        ROOT / "real_cell_movies" / "Fluo-C2DL-Huh7" / "01",
        ROOT / "real_cell_movies" / "Fluo-C2DL-Huh7" / "02",
    ]
    per_source = [8, 8, 7, 7]
    manifest: List[dict] = []
    for source, needed in zip(sources, per_source):
        frames = image_files(source)
        if len(frames) < 2:
            continue
        clip_len = min(window, len(frames))
        max_start = max(0, len(frames) - clip_len)
        starts = np.linspace(0, max_start, needed, dtype=int)
        for i, start in enumerate(starts):
            selected = frames[int(start): int(start) + clip_len]
            dataset = source.parent.name
            seq = source.name
            manifest.append({
                "clip_id": f"{dataset}_{seq}_clip_{i + 1:02d}",
                "dataset": dataset,
                "sequence": seq,
                "source_dir": str(source.relative_to(ROOT)),
                "start_frame": numeric_key(selected[0])[0],
                "end_frame": numeric_key(selected[-1])[0],
                "frames": [str(p.relative_to(ROOT)) for p in selected],
                "gt_seg_dir": (
                    str(gt_seg_dir(source).relative_to(ROOT))
                    if gt_seg_dir(source) else None
                ),
            })
    return manifest[:target_clips]


def center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x, y, w, h = box
    return (float(x + w / 2.0), float(y + h / 2.0))


def render_video(
    frames: List[np.ndarray],
    detections_by_frame: List[list],
    repaired_tracks: Dict[str, dict],
    output_path: Path,
    fps: float = 6.0,
) -> None:
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    points_by_frame: Dict[int, List[Tuple[str, float, float]]] = {}
    for tid, track in repaired_tracks.items():
        for f, x, y, _box in track["points"]:
            points_by_frame.setdefault(int(f), []).append((tid, float(x), float(y)))
    for fi, gray in enumerate(frames):
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for det in detections_by_frame[fi]:
            if getattr(det, "contour", None) is not None:
                cv2.drawContours(canvas, [det.contour], -1, (0, 220, 255), 1)
            cv2.circle(
                canvas,
                (int(round(det.center_x)), int(round(det.center_y))),
                2,
                (255, 0, 255),
                -1,
            )
        for tid, x, y in points_by_frame.get(fi, []):
            color_seed = abs(hash(tid))
            color = (
                60 + color_seed % 180,
                60 + (color_seed // 7) % 180,
                60 + (color_seed // 13) % 180,
            )
            cv2.circle(canvas, (int(round(x)), int(round(y))), 3, color, -1)
        cv2.putText(
            canvas,
            f"F{fi:02d} det {len(detections_by_frame[fi])} tracks {len(repaired_tracks)}",
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        writer.write(canvas)
    writer.release()


def repair_tracks(tracker: CellTracker, max_step_px: float) -> Tuple[Dict[str, dict], int]:
    repair = IdentityJumpRepairQAgent(max_step_px=max_step_px, min_segment_length=3)
    repaired: Dict[str, dict] = {}
    split_count = 0
    for tid, track in tracker.tracks.items():
        points = []
        for i, box in enumerate(track.boxes):
            cx, cy = center(box)
            points.append((track.birth_frame + i, cx, cy, box))
        segments = repair.split_centers([(f, x, y) for f, x, y, _ in points])
        if len(segments) > 1:
            split_count += len(segments) - 1
        for si, segment in enumerate(segments):
            lookup = {(f, round(x, 3), round(y, 3)): box for f, x, y, box in points}
            repaired_id = f"{tid}" if si == 0 else f"{tid}.{si + 1}"
            repaired[repaired_id] = {
                "cell_type": track.cell_type,
                "boxes": [],
                "points": [],
            }
            for f, x, y in segment:
                box = lookup.get((f, round(x, 3), round(y, 3)))
                if box is None:
                    box = (int(round(x - 4)), int(round(y - 4)), 8, 8)
                repaired[repaired_id]["boxes"].append(box)
                repaired[repaired_id]["points"].append((int(f), float(x), float(y), box))
    return repaired, split_count


def max_jump_px(repaired_tracks: Dict[str, dict]) -> float:
    max_jump = 0.0
    for track in repaired_tracks.values():
        pts = sorted(track["points"], key=lambda p: p[0])
        for prev, curr in zip(pts[:-1], pts[1:]):
            if curr[0] - prev[0] != 1:
                continue
            max_jump = max(max_jump, float(math.hypot(curr[1] - prev[1], curr[2] - prev[2])))
    return max_jump


def run_clip(clip: dict, out_dir: Path, max_side: int) -> dict:
    clip_dir = out_dir / clip["clip_id"]
    clip_dir.mkdir(parents=True, exist_ok=True)
    frames = [read_u8(ROOT / p, max_side=max_side) for p in clip["frames"]]
    is_dic_hela = clip["dataset"] == "DIC-C2DH-HeLa"
    if is_dic_hela:
        # DIC HeLa cells are large phase objects. The max-recall detector
        # sees DIC edge fragments as cells unless morphology training enforces
        # a larger minimum area before tracking starts.
        detector = CellDetector(
            min_area=700,
            max_area=30000,
            expected_max_diameter=180,
            use_blob_detector=False,
            use_hough_circles=False,
            sensitivity="low",
        )
    else:
        detector = CellDetector(
            min_area=12,
            max_area=15000,
            expected_max_diameter=90,
            use_blob_detector=False,
            use_hough_circles=False,
            sensitivity="max",
        )
        detector.calibrate(frames[0])
    detections_by_frame = [detector.detect(frame) for frame in frames]
    first_detections = detections_by_frame[0]
    tracker = CellTracker(max_missed=4, iou_threshold=0.05, max_distance=45.0)
    tracker.calibrate(first_detections)
    tracker.initialize(frames[0], first_detections)
    for frame, detections in zip(frames[1:], detections_by_frame[1:]):
        tracker.update(frame, detections)

    repaired, split_count = repair_tracks(tracker, max_step_px=28.0)
    jump = max_jump_px(repaired)

    analyzer = MigrationAnalyzer(pixel_size_x=1.0, pixel_size_y=1.0, time_per_frame=60.0)
    detailed_df, summary_df = analyzer.analyze(repaired)
    detailed_df.to_csv(clip_dir / "migration_detailed.csv", index=False)
    summary_df.to_csv(clip_dir / "migration_summary.csv", index=False)

    detections_rows = []
    gt_counts = []
    mask_dir = ROOT / clip["gt_seg_dir"] if clip["gt_seg_dir"] else None
    for fi, (frame_path, detections) in enumerate(zip(clip["frames"], detections_by_frame)):
        gtc = gt_count(mask_dir, ROOT / frame_path)
        if gtc is not None:
            gt_counts.append(gtc)
        for det in detections:
            detections_rows.append({
                "frame": fi,
                "center_x": det.center_x,
                "center_y": det.center_y,
                "w": det.w,
                "h": det.h,
                "area": det.area,
                "backend": det.backend,
                "center_source": det.center_source,
                "has_border": det.has_border,
            })
    pd.DataFrame(detections_rows).to_csv(clip_dir / "detections.csv", index=False)

    track_rows = []
    for tid, track in repaired.items():
        for f, x, y, box in track["points"]:
            track_rows.append({
                "track_id": tid,
                "frame": f,
                "center_x": x,
                "center_y": y,
                "x": box[0],
                "y": box[1],
                "w": box[2],
                "h": box[3],
            })
    pd.DataFrame(track_rows).to_csv(clip_dir / "track_positions.csv", index=False)

    visualizer = TrajectoryVisualizer(pixel_size_x=1.0, pixel_size_y=1.0, time_per_frame=60.0)
    plots_dir = clip_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    visualizer.plot_circular_trajectories(repaired, str(plots_dir / "plot_trajectories.png"))
    visualizer.plot_velocity_histogram(summary_df, str(plots_dir / "plot_velocity_histogram.png"))
    visualizer.plot_displacement_distance(summary_df, str(plots_dir / "plot_displacement_distance.png"))
    visualizer.plot_directionality(summary_df, str(plots_dir / "plot_directionality.png"))
    visualizer.plot_msd(repaired, str(plots_dir / "plot_msd.png"))
    visualizer.plot_rose(repaired, str(plots_dir / "plot_rose.png"))
    if len(repaired) <= 250:
        visualizer.plot_interactive(repaired, str(plots_dir / "plot_interactive.html"))
    render_video(frames, detections_by_frame, repaired, clip_dir / "tracking_video.mp4")

    det_counts = [len(d) for d in detections_by_frame]
    border_fraction = (
        sum(1 for row in detections_rows if row["has_border"]) / max(1, len(detections_rows))
    )
    mean_gt = float(np.mean(gt_counts)) if gt_counts else None
    mean_det = float(np.mean(det_counts))
    gt_ratio = (mean_det / mean_gt) if mean_gt and mean_gt > 0 else None
    required_plots = [
        "plot_trajectories.png",
        "plot_velocity_histogram.png",
        "plot_displacement_distance.png",
        "plot_directionality.png",
        "plot_msd.png",
        "plot_rose.png",
    ]
    plots_present = all((plots_dir / p).exists() for p in required_plots)
    pass_gates = {
        "frames_open": len(frames) == len(clip["frames"]) and all(f.size > 0 for f in frames),
        "detections_nonzero": mean_det > 0,
        "tracks_nonzero": len(repaired) > 0,
        "identity_jumps_repaired": jump <= 28.0,
        "borders_present": border_fraction >= 0.60,
        "gt_count_plausible": (
            True if gt_ratio is None else 0.25 <= float(gt_ratio) <= 4.0
        ),
        "metrics_written": not summary_df.empty and not detailed_df.empty,
        "plots_present": plots_present,
        "video_written": (clip_dir / "tracking_video.mp4").exists()
        and (clip_dir / "tracking_video.mp4").stat().st_size > 0,
    }
    passed = all(pass_gates.values())
    result = {
        **{k: clip[k] for k in ["clip_id", "dataset", "sequence", "start_frame", "end_frame"]},
        "n_frames": len(frames),
        "mean_detections": round(mean_det, 3),
        "min_detections": int(min(det_counts)),
        "max_detections": int(max(det_counts)),
        "mean_gt_count": round(mean_gt, 3) if mean_gt is not None else None,
        "det_to_gt_ratio": round(gt_ratio, 3) if gt_ratio is not None else None,
        "repaired_tracks": int(len(repaired)),
        "identity_splits": int(split_count),
        "max_jump_px_after_repair": round(float(jump), 3),
        "border_fraction": round(float(border_fraction), 3),
        "mean_velocity_px_frame": (
            round(float(summary_df["Avg_Velocity_um_min"].mean()), 4)
            if not summary_df.empty and "Avg_Velocity_um_min" in summary_df else 0.0
        ),
        "mean_displacement_px": (
            round(float(summary_df["Displacement_um"].mean()), 4)
            if not summary_df.empty and "Displacement_um" in summary_df else 0.0
        ),
        "passed": bool(passed),
        "failed_gates": ",".join([k for k, v in pass_gates.items() if not v]),
        "output_dir": str(clip_dir.relative_to(ROOT)),
    }
    with open(clip_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({**clip, "qc": result, "gates": pass_gates}, f, indent=2)
    return result


def write_dashboard(results: List[dict], out_dir: Path) -> None:
    passed = sum(1 for r in results if r["passed"])
    rows = []
    for r in results:
        color = "#137333" if r["passed"] else "#b3261e"
        rows.append(
            "<tr>"
            f"<td>{r['clip_id']}</td><td>{r['dataset']}</td>"
            f"<td>{r['mean_detections']}</td><td>{r['repaired_tracks']}</td>"
            f"<td>{r['max_jump_px_after_repair']}</td>"
            f"<td>{r['det_to_gt_ratio'] if r['det_to_gt_ratio'] is not None else ''}</td>"
            f"<td style='color:{color};font-weight:700'>{'PASS' if r['passed'] else 'FAIL'}</td>"
            f"<td>{r['failed_gates']}</td>"
            f"<td><a href='{Path(r['output_dir']).name}/tracking_video.mp4'>video</a></td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CytoTrack AI 30-Movie Stress Test</title>
  <style>
    body {{ font-family: Inter, Arial, sans-serif; margin: 24px; color: #1f1f1f; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #f5f7fa; position: sticky; top: 0; }}
    .summary {{ display: flex; gap: 16px; margin-bottom: 18px; }}
    .metric {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px 16px; }}
    .metric strong {{ display: block; font-size: 24px; }}
  </style>
</head>
<body>
  <h1>CytoTrack AI 30-Movie Stress Test</h1>
  <div class="summary">
    <div class="metric"><strong>{passed}/30</strong> clips passed automated gates</div>
    <div class="metric"><strong>{np.mean([r['mean_detections'] for r in results]):.1f}</strong> mean detections/frame</div>
    <div class="metric"><strong>{max([r['max_jump_px_after_repair'] for r in results] or [0]):.1f}px</strong> max repaired jump</div>
  </div>
  <table>
    <thead><tr><th>Clip</th><th>Dataset</th><th>Mean Det</th><th>Tracks</th><th>Max Jump</th><th>Det/GT</th><th>Status</th><th>Failed Gates</th><th>Video</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""
    (out_dir / "dashboard.html").write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", type=int, default=30)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--max-side", type=int, default=384)
    parser.add_argument("--output", default=str(ROOT / "RESULT" / "stress_30_movies"))
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output)
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = make_manifest(args.clips, args.window)
    with open(out_dir / "clip_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    results = []
    for idx, clip in enumerate(manifest, start=1):
        print(f"[{idx:02d}/{len(manifest):02d}] {clip['clip_id']}", flush=True)
        results.append(run_clip(clip, out_dir, max_side=args.max_side))

    with open(out_dir / "stress_30_movies_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    summary = {
        "requested_clips": args.clips,
        "completed_clips": len(results),
        "passed_clips": sum(1 for r in results if r["passed"]),
        "failed_clips": sum(1 for r in results if not r["passed"]),
        "mean_detections_per_frame": float(np.mean([r["mean_detections"] for r in results])),
        "max_jump_px_after_repair": float(max(r["max_jump_px_after_repair"] for r in results)),
        "outputs": {
            "csv": str((out_dir / "stress_30_movies_report.csv").relative_to(ROOT)),
            "dashboard": str((out_dir / "dashboard.html").relative_to(ROOT)),
        },
    }
    with open(out_dir / "stress_30_movies_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_dashboard(results, out_dir)
    print(json.dumps(summary, indent=2))
    return 0 if summary["failed_clips"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
