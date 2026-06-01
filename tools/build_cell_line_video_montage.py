#!/usr/bin/env python3
"""Build a side-by-side cell-line tracking video montage.

The input videos are already produced after per-cell-line morphology training.
This tool creates a compact review movie where each panel keeps its own
cell-line label and overlay color so mixed-line experiments can be audited
without losing the per-line identity.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import cv2
import numpy as np


def _parse_input(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("input must be Label=/path/to/video.mp4")
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label:
        raise argparse.ArgumentTypeError("input label is empty")
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"video does not exist: {path}")
    return label, path


def _fit_panel(frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    out_w, out_h = size
    h, w = frame.shape[:2]
    scale = min(out_w / max(w, 1), out_h / max(h, 1))
    resized = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)
    panel = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    y = (out_h - resized.shape[0]) // 2
    x = (out_w - resized.shape[1]) // 2
    panel[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return panel


def build_montage(inputs: List[Tuple[str, str]], output: str,
                  frames: int, fps: float, panel_width: int,
                  panel_height: int) -> None:
    caps = [cv2.VideoCapture(path) for _, path in inputs]
    try:
        for label, cap in zip((x[0] for x in inputs), caps):
            if not cap.isOpened():
                raise RuntimeError(f"could not open video for {label}")
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        writer = cv2.VideoWriter(
            output,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (panel_width * len(inputs), panel_height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"could not open output writer: {output}")
        try:
            for i in range(frames):
                panels = []
                for label, cap in zip((x[0] for x in inputs), caps):
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = cap.read()
                    if not ok or frame is None:
                        frame = np.zeros((panel_height, panel_width, 3),
                                         dtype=np.uint8)
                    panel = _fit_panel(frame, (panel_width, panel_height))
                    cv2.rectangle(panel, (0, 0), (panel_width - 1, 34),
                                  (0, 0, 0), -1)
                    cv2.putText(panel, f"{label} | frame {i:04d}", (10, 23),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                                (255, 255, 255), 2, cv2.LINE_AA)
                    panels.append(panel)
                writer.write(np.hstack(panels))
        finally:
            writer.release()
    finally:
        for cap in caps:
            cap.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=_parse_input,
                        required=True,
                        help="Label=/path/to/tracking_video.mp4")
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--panel-width", type=int, default=420)
    parser.add_argument("--panel-height", type=int, default=520)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_montage(
        args.input,
        args.output,
        args.frames,
        args.fps,
        args.panel_width,
        args.panel_height,
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
