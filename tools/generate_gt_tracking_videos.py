#!/usr/bin/env python3
"""Generate clean tracking videos from CTC manual TRA masks."""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _frame_number(path: str) -> int:
    match = re.search(r"(\d+)", os.path.basename(path))
    if not match:
        raise ValueError(f"could not parse frame number from {path}")
    return int(match.group(1))


def _image_path(root: str, dataset: str, sequence: str, frame: int) -> str:
    return os.path.join(root, dataset, sequence, f"t{frame:03d}.tif")


def _mask_path(root: str, dataset: str, sequence: str, frame: int) -> str:
    return os.path.join(root, dataset, f"{sequence}_GT", "TRA",
                        f"man_track{frame:03d}.tif")


def _frame_paths(root: str, dataset: str, sequence: str) -> List[str]:
    seq_dir = os.path.join(root, dataset, sequence)
    return sorted(
        os.path.join(seq_dir, name)
        for name in os.listdir(seq_dir)
        if name.lower().endswith((".tif", ".tiff"))
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


def _color_for(label: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(label * 7919)
    return tuple(int(v) for v in rng.integers(70, 255, size=3))


def _draw_mask_tracks(
    vis: np.ndarray,
    mask: np.ndarray,
    trails: Dict[int, List[Tuple[int, int]]],
) -> int:
    labels = [int(v) for v in np.unique(mask) if v > 0]
    for label in labels:
        binary = np.uint8(mask == label) * 255
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area <= 0:
            continue
        color = _color_for(label)
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
        trails[label].append((cx, cy))
        for prev, curr in zip(trails[label][-80:], trails[label][-79:]):
            cv2.line(vis, prev, curr, color, 2)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(vis, f"ID:{label}", (x, max(12, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        cv2.putText(vis, f"ID:{label}", (x, max(12, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return len(labels)


def _draw_hud(
    vis: np.ndarray,
    dataset: str,
    sequence: str,
    frame_idx: int,
    total: int,
    active: int,
) -> np.ndarray:
    height, width = vis.shape[:2]
    hud_h = 92
    hud = np.full((hud_h, width, 3), (34, 34, 34), dtype=np.uint8)
    cv2.putText(hud, f"CytoTrack AI - Manual TRA Tracking - {dataset}/{sequence}",
                (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (110, 210, 160), 2)
    cv2.putText(hud, f"Frame: {frame_idx + 1}/{total}", (14, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
    cv2.putText(hud, f"Active labelled cells: {active}", (175, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    bar_x0, bar_y = 14, 72
    bar_w = max(1, width - 28)
    cv2.rectangle(hud, (bar_x0, bar_y), (bar_x0 + bar_w, bar_y + 9),
                  (70, 70, 70), -1)
    filled = int(bar_w * (frame_idx + 1) / max(1, total))
    cv2.rectangle(hud, (bar_x0, bar_y), (bar_x0 + filled, bar_y + 9),
                  (110, 210, 160), -1)
    return np.vstack([hud, vis])


def _write_summary(output_dir: str, dataset: str, sequence: str,
                   rows: List[dict], video_path: str) -> None:
    csv_path = os.path.join(output_dir, f"{dataset}_{sequence}_gt_tracks.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["frame", "track_id", "x", "y", "w", "h", "area"])
        writer.writeheader()
        writer.writerows(rows)
    unique_ids = sorted({row["track_id"] for row in rows})
    with open(os.path.join(output_dir, f"{dataset}_{sequence}_summary.txt"),
              "w", encoding="utf-8") as f:
        f.write(f"dataset={dataset}\n")
        f.write(f"sequence={sequence}\n")
        f.write(f"video={video_path}\n")
        f.write(f"frames={len(set(row['frame'] for row in rows))}\n")
        f.write(f"unique_track_ids={len(unique_ids)}\n")
        f.write(f"track_ids={','.join(str(i) for i in unique_ids)}\n")


def generate_video(root: str, dataset: str, sequence: str,
                   output_dir: str, fps: float, fourcc: str) -> str:
    image_paths = _frame_paths(root, dataset, sequence)
    if not image_paths:
        raise FileNotFoundError(f"no frames for {dataset}/{sequence}")

    first = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED)
    first_vis = _normalize_to_bgr(first)
    height, width = first_vis.shape[:2]
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, f"{dataset}_{sequence}_gt_tracking_video.avi")
    writer = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc(*fourcc), float(fps), (width, height + 92))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {video_path}")

    trails: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    rows: List[dict] = []
    for idx, image_path in enumerate(image_paths):
        frame = _frame_number(image_path)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(_mask_path(root, dataset, sequence, frame), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None:
            continue
        vis = _normalize_to_bgr(image)
        active = _draw_mask_tracks(vis, mask, trails)
        for label in [int(v) for v in np.unique(mask) if v > 0]:
            binary = np.uint8(mask == label) * 255
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contour)
            rows.append({
                "frame": frame,
                "track_id": label,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "area": float(cv2.contourArea(contour)),
            })
        writer.write(_draw_hud(vis, dataset, sequence, idx, len(image_paths), active))
    writer.release()
    _write_summary(output_dir, dataset, sequence, rows, video_path)
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
            if os.path.isdir(os.path.join(root, dataset, f"{sequence}_GT", "TRA")):
                pairs.append((dataset, sequence))
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate tracking videos from manual CTC TRA masks.")
    parser.add_argument("--root", default=os.path.join(REPO_ROOT, "real_cell_movies"))
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--sequence", action="append")
    parser.add_argument("--output-dir", default=os.path.join(
        REPO_ROOT, "tracking_results_real", "manual_gt"))
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--fourcc", default="XVID")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generated = []
    for dataset, sequence in _discover(args.root, args.dataset, args.sequence):
        out_dir = os.path.join(args.output_dir, f"{dataset}_{sequence}")
        print(f"[gt-video] {dataset}/{sequence} -> {out_dir}", flush=True)
        generated.append(generate_video(
            args.root, dataset, sequence, out_dir, args.fps, args.fourcc))
    print("Generated videos:", flush=True)
    for path in generated:
        print(path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
