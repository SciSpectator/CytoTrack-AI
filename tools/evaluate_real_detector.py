#!/usr/bin/env python3
"""Evaluate detector counts against Cell Tracking Challenge SEG masks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from detector import CellDetector, Detection  # noqa: E402


@dataclass
class FrameMetrics:
    dataset: str
    sequence: str
    frame: int
    gt_objects: int
    detections: int
    matched_objects: int
    detections_inside_gt: int


def _frame_number(path: str) -> int:
    match = re.search(r"(\d+)", os.path.basename(path))
    if not match:
        raise ValueError(f"could not parse frame number from {path}")
    return int(match.group(1))


def _mask_paths(root: str, dataset: str, sequence: str) -> List[str]:
    seg_dir = os.path.join(root, dataset, f"{sequence}_GT", "SEG")
    if not os.path.isdir(seg_dir):
        raise FileNotFoundError(seg_dir)
    return sorted(
        os.path.join(seg_dir, name)
        for name in os.listdir(seg_dir)
        if name.lower().endswith((".tif", ".tiff"))
    )


def _image_path(root: str, dataset: str, sequence: str, frame: int) -> str:
    return os.path.join(root, dataset, sequence, f"t{frame:03d}.tif")


def _labels(mask: np.ndarray) -> np.ndarray:
    labels = np.unique(mask)
    return labels[labels > 0]


def _center_label(mask: np.ndarray, det: Detection) -> int:
    h, w = mask.shape[:2]
    x = min(max(int(round(det.center_x)), 0), w - 1)
    y = min(max(int(round(det.center_y)), 0), h - 1)
    return int(mask[y, x])


def _draw_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    detections: Sequence[Detection],
    out_path: str,
) -> None:
    if image.ndim == 2:
        if image.dtype != np.uint8:
            vis_gray = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            vis_gray = vis_gray.astype(np.uint8)
        else:
            vis_gray = image
        vis = cv2.cvtColor(vis_gray, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    for label in _labels(mask):
        binary = np.uint8(mask == label) * 255
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 1)

    for det in detections:
        cv2.rectangle(
            vis,
            (det.x, det.y),
            (det.x + det.w, det.y + det.h),
            (0, 0, 255),
            1,
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, vis)


def evaluate_sequence(
    root: str,
    dataset: str,
    sequence: str,
    args: argparse.Namespace,
    overlay_dir: Optional[str] = None,
) -> Tuple[List[FrameMetrics], dict]:
    masks = _mask_paths(root, dataset, sequence)
    if not masks:
        return [], {}

    detector = CellDetector(
        min_area=args.min_area,
        max_area=args.max_area,
        expected_max_diameter=args.expected_max_diameter,
        use_blob_detector=not args.no_blob_detector,
        use_hough_circles=not args.no_hough_circles,
        sensitivity=args.sensitivity,
    )

    calibration = None
    if args.calibrate:
        first_frame = _frame_number(masks[0])
        first_image = cv2.imread(
            _image_path(root, dataset, sequence, first_frame),
            cv2.IMREAD_UNCHANGED,
        )
        calibration = detector.calibrate(first_image)

    rows: List[FrameMetrics] = []
    for idx, mask_path in enumerate(masks):
        frame = _frame_number(mask_path)
        image = cv2.imread(
            _image_path(root, dataset, sequence, frame),
            cv2.IMREAD_UNCHANGED,
        )
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if image is None or mask is None:
            raise FileNotFoundError(f"missing image or mask for {mask_path}")

        detections = detector.detect(image)
        gt_labels = _labels(mask)
        matched = set()
        inside = 0
        for det in detections:
            label = _center_label(mask, det)
            if label > 0:
                inside += 1
                matched.add(label)

        rows.append(FrameMetrics(
            dataset=dataset,
            sequence=sequence,
            frame=frame,
            gt_objects=int(len(gt_labels)),
            detections=int(len(detections)),
            matched_objects=int(len(matched)),
            detections_inside_gt=int(inside),
        ))

        if overlay_dir and idx < args.overlay_limit:
            out_path = os.path.join(
                overlay_dir,
                f"{dataset}_{sequence}_t{frame:03d}.png",
            )
            _draw_overlay(image, mask, detections, out_path)

    return rows, calibration or {}


def _summarize(rows: Sequence[FrameMetrics], calibration: dict) -> dict:
    gt = sum(r.gt_objects for r in rows)
    detections = sum(r.detections for r in rows)
    matched = sum(r.matched_objects for r in rows)
    inside = sum(r.detections_inside_gt for r in rows)
    frames = len(rows)
    return {
        "dataset": rows[0].dataset if rows else "",
        "sequence": rows[0].sequence if rows else "",
        "frames": frames,
        "gt_objects": gt,
        "detections": detections,
        "mean_gt_per_frame": gt / frames if frames else 0.0,
        "mean_detections_per_frame": detections / frames if frames else 0.0,
        "object_recall": matched / gt if gt else 0.0,
        "center_precision": inside / detections if detections else 0.0,
        "count_ratio": detections / gt if gt else 0.0,
        "calibration": calibration,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CellDetector against real SEG masks.",
    )
    parser.add_argument("--root", default=os.path.join(REPO_ROOT, "real_cell_movies"))
    parser.add_argument("--dataset", action="append",
                        help="Dataset name. Repeatable. Default: all found.")
    parser.add_argument("--sequence", action="append",
                        help="Sequence id. Repeatable. Default: 01 and 02 if present.")
    parser.add_argument("--sensitivity", default="normal",
                        choices=sorted(CellDetector.SENSITIVITY_PRESETS))
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--max-area", type=int, default=30000)
    parser.add_argument("--expected-max-diameter", type=int, default=60)
    parser.add_argument("--no-blob-detector", action="store_true")
    parser.add_argument("--no-hough-circles", action="store_true")
    parser.add_argument("--no-calibrate", dest="calibrate",
                        action="store_false")
    parser.set_defaults(calibrate=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overlay-limit", type=int, default=0)
    parser.add_argument("--indent", type=int, default=2)
    return parser.parse_args()


def _discover(root: str, datasets: Optional[Iterable[str]],
              sequences: Optional[Iterable[str]]) -> List[Tuple[str, str]]:
    dataset_names = list(datasets) if datasets else [
        name for name in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, name))
    ]
    requested_sequences = list(sequences) if sequences else ["01", "02"]
    pairs = []
    for dataset in dataset_names:
        for sequence in requested_sequences:
            if os.path.isdir(os.path.join(root, dataset, f"{sequence}_GT", "SEG")):
                pairs.append((dataset, sequence))
    return pairs


def main() -> int:
    args = parse_args()
    pairs = _discover(args.root, args.dataset, args.sequence)
    summaries = []
    frame_rows: List[FrameMetrics] = []

    overlay_dir = None
    if args.output_dir and args.overlay_limit > 0:
        overlay_dir = os.path.join(args.output_dir, "overlays")

    for dataset, sequence in pairs:
        rows, calibration = evaluate_sequence(
            args.root, dataset, sequence, args, overlay_dir=overlay_dir)
        frame_rows.extend(rows)
        summaries.append(_summarize(rows, calibration))

    result = {
        "settings": {
            "sensitivity": args.sensitivity,
            "min_area": args.min_area,
            "max_area": args.max_area,
            "expected_max_diameter": args.expected_max_diameter,
            "use_blob_detector": not args.no_blob_detector,
            "use_hough_circles": not args.no_hough_circles,
            "calibrate": args.calibrate,
        },
        "summaries": summaries,
    }

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "real_detector_summary.json"),
                  "w", encoding="utf-8") as f:
            json.dump(result, f, indent=args.indent, sort_keys=True)
        with open(os.path.join(args.output_dir, "real_detector_frames.csv"),
                  "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(frame_rows[0]).keys()))
            writer.writeheader()
            for row in frame_rows:
                writer.writerow(asdict(row))

    print(json.dumps(result, indent=args.indent, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
