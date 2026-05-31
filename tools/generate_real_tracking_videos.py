#!/usr/bin/env python3
"""Generate tracking overlay videos for real Cell Tracking Challenge movies."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from detector import CellDetector  # noqa: E402
from tracker import CellTracker  # noqa: E402


Box = Tuple[int, int, int, int]


def _frame_paths(root: str, dataset: str, sequence: str) -> List[str]:
    seq_dir = os.path.join(root, dataset, sequence)
    return sorted(
        os.path.join(seq_dir, name)
        for name in os.listdir(seq_dir)
        if name.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg"))
    )


def _normalize_to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        if image.dtype == np.uint8:
            return image.copy()
        gray = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if image.dtype == np.uint8:
        gray = image
    else:
        gray = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _read_frame(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    return image


def _draw_hud(
    vis: np.ndarray,
    dataset: str,
    sequence: str,
    frame_idx: int,
    total: int,
    detection_count: int,
    tracker: CellTracker,
) -> np.ndarray:
    height, width = vis.shape[:2]
    hud_h = 92
    hud = np.full((hud_h, width, 3), (34, 34, 34), dtype=np.uint8)
    title = f"CytoTrack AI - {dataset}/{sequence}"
    cv2.putText(hud, title, (14, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (110, 210, 160), 2)
    cv2.putText(hud, f"Frame: {frame_idx + 1}/{total}", (14, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
    cv2.putText(hud, f"Detections: {detection_count}", (175, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(hud, f"Active: {tracker.active_count}", (365, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(hud, f"Lost: {tracker.lost_count}", (505, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 120, 255), 1)
    bar_x0 = 14
    bar_y = 72
    bar_w = max(1, width - 28)
    cv2.rectangle(hud, (bar_x0, bar_y), (bar_x0 + bar_w, bar_y + 9),
                  (70, 70, 70), -1)
    filled = int(bar_w * (frame_idx + 1) / max(1, total))
    cv2.rectangle(hud, (bar_x0, bar_y), (bar_x0 + filled, bar_y + 9),
                  (110, 210, 160), -1)
    return np.vstack([hud, vis])


def _draw_tracks(vis: np.ndarray, tracker: CellTracker) -> None:
    for tid, track in tracker.tracks.items():
        if not track.boxes:
            continue
        color = tuple(int(c) for c in track.color)
        centers = [
            (int(x + w / 2), int(y + h / 2))
            for x, y, w, h in track.boxes[-80:]
        ]
        for prev, curr in zip(centers, centers[1:]):
            cv2.line(vis, prev, curr, color, 2)
        if not track.is_active:
            continue
        box = track.display_bbox() if hasattr(track, "display_bbox") else track.boxes[-1]
        x, y, w, h = [int(v) for v in box]
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cv2.circle(vis, (int(x + w / 2), int(y + h / 2)), 3, color, -1)
        label = f"ID:{tid}"
        text_y = max(12, y - 5)
        cv2.putText(vis, label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (0, 0, 0), 2)
        cv2.putText(vis, label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, color, 1)


def _write_tracks_csv(path: str, tracker: CellTracker) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["track_id", "frame_index", "x", "y", "w", "h", "active"],
        )
        writer.writeheader()
        for tid, track in sorted(tracker.tracks.items()):
            for idx, (x, y, w, h) in enumerate(track.boxes):
                writer.writerow({
                    "track_id": tid,
                    "frame_index": idx,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "active": track.is_active,
                })


def generate_video(
    root: str,
    dataset: str,
    sequence: str,
    output_dir: str,
    args: argparse.Namespace,
) -> Optional[str]:
    paths = _frame_paths(root, dataset, sequence)
    if not paths:
        return None

    os.makedirs(output_dir, exist_ok=True)
    first_raw = _read_frame(paths[0])
    first_vis = _normalize_to_bgr(first_raw)
    height, width = first_vis.shape[:2]

    detector = CellDetector(
        min_area=args.min_area,
        max_area=args.max_area,
        expected_max_diameter=args.expected_max_diameter,
        use_blob_detector=not args.no_blob_detector,
        use_hough_circles=not args.no_hough_circles,
        sensitivity=args.sensitivity,
        whole_cell_border=args.whole_cell_border,
    )
    detector.calibrate(first_raw)
    first_dets = detector.detect(first_raw)

    tracker = CellTracker(
        max_missed=args.max_missed,
        suppress_duplicate_detections=args.sensitivity in ("locate", "high", "max"),
    )
    tracker.calibrate(first_dets)
    tracker.initialize(first_raw, first_dets)

    video_path = os.path.join(output_dir, f"{dataset}_{sequence}_tracking_video.avi")
    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*args.fourcc),
        float(args.fps),
        (width, height + 92),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {video_path}")

    detection_counts = []
    for idx, path in enumerate(paths):
        raw = _read_frame(path)
        vis = _normalize_to_bgr(raw)
        if idx == 0:
            dets = first_dets
        else:
            dets = detector.detect(raw)
            tracker.update(raw, detections=dets)
        detection_counts.append(len(dets))
        _draw_tracks(vis, tracker)
        writer.write(_draw_hud(vis, dataset, sequence, idx, len(paths),
                               len(dets), tracker))

    writer.release()
    _write_tracks_csv(os.path.join(output_dir, f"{dataset}_{sequence}_tracks.csv"),
                      tracker)
    with open(os.path.join(output_dir, f"{dataset}_{sequence}_summary.txt"),
              "w", encoding="utf-8") as f:
        f.write(f"dataset={dataset}\n")
        f.write(f"sequence={sequence}\n")
        f.write(f"frames={len(paths)}\n")
        f.write(f"video={video_path}\n")
        f.write(f"tracks_total={len(tracker.tracks)}\n")
        f.write(f"tracks_active={tracker.active_count}\n")
        f.write(f"detections_mean={float(np.mean(detection_counts)):.3f}\n")
        f.write(f"detections_min={int(np.min(detection_counts))}\n")
        f.write(f"detections_max={int(np.max(detection_counts))}\n")
    return video_path


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
            seq_dir = os.path.join(root, dataset, sequence)
            if os.path.isdir(seq_dir):
                pairs.append((dataset, sequence))
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CytoTrack tracking overlay videos for real movies.",
    )
    parser.add_argument("--root", default=os.path.join(REPO_ROOT, "real_cell_movies"))
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--sequence", action="append")
    parser.add_argument("--output-dir", default=os.path.join(
        REPO_ROOT, "tracking_results_real"))
    parser.add_argument("--sensitivity", default="normal",
                        choices=sorted(CellDetector.SENSITIVITY_PRESETS))
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--max-area", type=int, default=30000)
    parser.add_argument("--expected-max-diameter", type=int, default=60)
    parser.add_argument("--no-blob-detector", action="store_true")
    parser.add_argument("--no-hough-circles", action="store_true")
    parser.add_argument("--whole-cell-border", action="store_true",
                        help="repair fragment contours into full cell-body outlines")
    parser.add_argument("--max-missed", type=int, default=15)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--fourcc", default="XVID")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pairs = _discover(args.root, args.dataset, args.sequence)
    if not pairs:
        raise SystemExit("No dataset sequences found.")

    os.makedirs(args.output_dir, exist_ok=True)
    generated = []
    for dataset, sequence in pairs:
        sequence_out = os.path.join(args.output_dir, f"{dataset}_{sequence}")
        print(f"[video] generating {dataset}/{sequence} -> {sequence_out}")
        path = generate_video(args.root, dataset, sequence, sequence_out, args)
        if path:
            generated.append(path)
            print(f"[video] wrote {path}")

    print("Generated videos:")
    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
