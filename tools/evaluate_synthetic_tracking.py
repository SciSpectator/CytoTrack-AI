#!/usr/bin/env python3
"""Run a synthetic 32-cell tracking scenario and emit JSON metrics."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from detector import CellDetector  # noqa: E402
from synthetic_data import SyntheticDataGenerator  # noqa: E402
from tracker import CellTracker  # noqa: E402


Box = Tuple[int, int, int, int]
FrameTracks = Dict[int, Box]
GroundTruth = Dict[int, dict]


def _center(box: Box) -> Tuple[float, float]:
    return box[0] + box[2] / 2.0, box[1] + box[3] / 2.0


def _nearest_gt(
    gt_frame: GroundTruth,
    cx: float,
    cy: float,
    max_radius: float,
) -> Tuple[int, float]:
    best_id = -1
    best_dist = float("inf")
    for gt_id, entry in gt_frame.items():
        gx, gy = entry["center"]
        dist = math.hypot(cx - gx, cy - gy)
        if dist < best_dist:
            best_id = int(gt_id)
            best_dist = float(dist)
    if best_dist > max_radius:
        return -1, best_dist
    return best_id, best_dist


def _active_boxes(tracker: CellTracker) -> FrameTracks:
    return {
        int(tid): tuple(int(v) for v in track.boxes[-1])
        for tid, track in tracker.tracks.items()
        if track.is_active and track.boxes
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def evaluate_tracks(
    frames_tracks: List[FrameTracks],
    gt_frames: List[GroundTruth],
    match_radius: float,
    num_cells: int,
    total_tracks: int,
) -> dict:
    per_track_gt: Dict[int, List[int]] = defaultdict(list)
    per_track_dist: Dict[int, List[float]] = defaultdict(list)
    gt_to_track_ids: Dict[int, set] = defaultdict(set)

    for frame_idx, track_boxes in enumerate(frames_tracks):
        gt = gt_frames[frame_idx]
        for track_id, box in track_boxes.items():
            cx, cy = _center(box)
            gt_id, dist = _nearest_gt(gt, cx, cy, match_radius)
            per_track_gt[track_id].append(gt_id)
            per_track_dist[track_id].append(dist)
            if gt_id >= 0:
                gt_to_track_ids[gt_id].add(track_id)

    track_purities = {}
    for track_id, gt_sequence in per_track_gt.items():
        matched = [gt_id for gt_id in gt_sequence if gt_id >= 0]
        if not matched:
            track_purities[track_id] = 0.0
            continue
        dominant_count = Counter(matched).most_common(1)[0][1]
        track_purities[track_id] = dominant_count / len(gt_sequence)

    id_switches = 0
    for gt_sequence in per_track_gt.values():
        previous = None
        for gt_id in gt_sequence:
            if gt_id < 0:
                continue
            if previous is not None and gt_id != previous:
                id_switches += 1
            previous = gt_id

    matched_dists = [
        dist
        for distances in per_track_dist.values()
        for dist in distances
        if dist < match_radius
    ]
    ghost_frames = sum(
        1
        for distances in per_track_dist.values()
        for dist in distances
        if dist >= match_radius
    )
    emitted_frames = sum(len(distances) for distances in per_track_dist.values())

    gt_fragments = {
        str(gt_id): len(track_ids)
        for gt_id, track_ids in sorted(gt_to_track_ids.items())
    }

    return {
        "active_tracks": len(frames_tracks[-1]) if frames_tracks else 0,
        "lost_tracks": max(0, total_tracks - (len(frames_tracks[-1]) if frames_tracks else 0)),
        "total_tracks": int(total_tracks),
        "fragmentation_ratio": float(total_tracks / max(1, num_cells)),
        "excess_fragmentation_ratio": float(max(0, total_tracks - num_cells) / max(1, num_cells)),
        "approx_id_switches": int(id_switches),
        "mean_track_purity": _mean(track_purities.values()),
        "min_track_purity": float(min(track_purities.values())) if track_purities else 0.0,
        "mean_localization_error_px": float(np.mean(matched_dists)) if matched_dists else None,
        "ghost_frame_rate": float(ghost_frames / max(1, emitted_frames)),
        "gt_fragment_counts": gt_fragments,
    }


def run(args: argparse.Namespace) -> dict:
    generator = SyntheticDataGenerator(
        width=args.width,
        height=args.height,
        num_cells=args.cells,
        num_frames=args.frames,
        seed=args.seed,
        overlap_density=args.overlap_density,
    )
    generator.generate_cells()

    detector = CellDetector(
        min_area=args.min_area,
        max_area=args.max_area,
        expected_max_diameter=args.expected_max_diameter,
        use_blob_detector=not args.no_blob_detector,
        use_hough_circles=not args.no_hough_circles,
        sensitivity=args.sensitivity,
    )
    tracker = CellTracker(
        max_missed=args.max_missed,
        iou_threshold=args.iou_threshold,
        max_distance=args.max_distance,
        suppress_duplicate_detections=args.suppress_duplicate_detections,
    )

    gt_frames: List[GroundTruth] = []
    frames_tracks: List[FrameTracks] = []
    detection_counts: List[int] = []
    detector_calibration = None
    tracker_calibration = None

    for frame_idx in range(args.frames):
        frame, gt = generator.generate_frame(frame_idx)
        gt_frames.append(gt)

        if frame_idx == 0 and args.calibrate_detector:
            detector_calibration = detector.calibrate(frame)

        detections = detector.detect(frame)
        detection_counts.append(len(detections))

        if frame_idx == 0:
            if args.calibrate_tracker:
                tracker_calibration = tracker.calibrate(detections)
            tracker.initialize(frame, detections)
            frames_tracks.append(_active_boxes(tracker))
        else:
            frames_tracks.append(tracker.update(frame, detections=detections))

    median_diameter = None
    if tracker_calibration:
        median_diameter = tracker_calibration.get("median_diameter")
    if not median_diameter and detector_calibration:
        median_diameter = detector_calibration.get("median_diameter")
    match_radius = args.match_radius or max(float(median_diameter or 20.0) * 1.5, 20.0)

    track_metrics = evaluate_tracks(
        frames_tracks=frames_tracks,
        gt_frames=gt_frames,
        match_radius=match_radius,
        num_cells=args.cells,
        total_tracks=len(tracker.tracks),
    )

    return {
        "scenario": {
            "cells": args.cells,
            "frames": args.frames,
            "width": args.width,
            "height": args.height,
            "seed": args.seed,
            "overlap_density": args.overlap_density,
        },
        "detector_settings": {
            "sensitivity": args.sensitivity,
            "min_area": detector.min_area,
            "max_area": detector.max_area,
            "expected_max_diameter": detector.expected_max_diameter,
            "use_blob_detector": detector.use_blob_detector,
            "use_hough_circles": detector.use_hough_circles,
            "calibrated": bool(args.calibrate_detector),
        },
        "tracker_settings": {
            "max_missed": tracker.max_missed,
            "iou_threshold": tracker.iou_threshold,
            "max_distance": tracker.max_distance,
            "suppress_duplicate_detections": tracker.suppress_duplicate_detections,
            "calibrated": bool(args.calibrate_tracker),
        },
        "detection_counts": {
            "per_frame": detection_counts,
            "total": int(sum(detection_counts)),
            "first_frame": detection_counts[0] if detection_counts else 0,
            "last_frame": detection_counts[-1] if detection_counts else 0,
            "min": min(detection_counts) if detection_counts else 0,
            "max": max(detection_counts) if detection_counts else 0,
            "mean": _mean(detection_counts),
        },
        "metrics": track_metrics,
        "calibration": {
            "detector": detector_calibration,
            "tracker": tracker_calibration,
            "match_radius": match_radius,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CytoTrack on a synthetic 32-cell scenario.",
    )
    parser.add_argument("--cells", type=int, default=32)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overlap-density", type=float, default=0.0)
    parser.add_argument("--match-radius", type=float, default=None)

    parser.add_argument("--sensitivity", default="normal",
                        choices=sorted(CellDetector.SENSITIVITY_PRESETS))
    parser.add_argument("--min-area", type=int, default=40)
    parser.add_argument("--max-area", type=int, default=9000)
    parser.add_argument("--expected-max-diameter", type=int, default=60)
    parser.add_argument("--no-blob-detector", action="store_true")
    parser.add_argument("--no-hough-circles", action="store_true")
    parser.add_argument("--no-calibrate-detector", dest="calibrate_detector",
                        action="store_false")
    parser.set_defaults(calibrate_detector=True)

    parser.add_argument("--max-missed", type=int, default=15)
    parser.add_argument("--iou-threshold", type=float, default=0.1)
    parser.add_argument("--max-distance", type=float, default=80.0)
    parser.add_argument("--suppress-duplicate-detections", action="store_true")
    parser.add_argument("--no-calibrate-tracker", dest="calibrate_tracker",
                        action="store_false")
    parser.set_defaults(calibrate_tracker=True)
    parser.add_argument("--indent", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        metrics = run(args)
    print(json.dumps(metrics, indent=args.indent, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
