"""
Stress test: 150 cells across 8 frames.

Verifies that the Kalman + Hungarian tracker combined with the calibrated
detector keeps the overwhelming majority of IDs stable in a dense scene.
This is the scenario the user explicitly called out: 120+ cells, no IDs
should silently disappear between frames.
"""

import numpy as np

from detector import CellDetector
from tracker import CellTracker
from synthetic_data import SyntheticDataGenerator


def _run_dense(num_cells: int, num_frames: int = 8):
    gen = SyntheticDataGenerator(width=720, height=540,
                                 num_cells=num_cells, num_frames=num_frames,
                                 seed=7)
    gen.generate_cells()

    frames = []
    for i in range(num_frames):
        img, _ = gen.generate_frame(i)
        frames.append(img)

    detector = CellDetector(min_area=20, max_area=12000)
    detector.calibrate(frames[0])

    first_dets = detector.detect(frames[0])

    tracker = CellTracker(max_missed=6)
    tracker.calibrate(first_dets)
    tracker.initialize(frames[0], first_dets)

    initial_ids = set(tracker.tracks.keys())

    for f in frames[1:]:
        dets = detector.detect(f)
        tracker.update(f, detections=dets)

    return tracker, initial_ids, len(first_dets)


def test_detector_finds_most_of_150_cells():
    tracker, initial_ids, n_first = _run_dense(num_cells=150, num_frames=3)
    # Detector should find the lion's share of them on the first frame
    assert n_first >= 110, (
        f"Expected detector to find >= 110 of 150 synthetic cells, got {n_first}")


def test_dense_tracker_preserves_ids_for_120_cells():
    tracker, initial_ids, n_first = _run_dense(num_cells=125, num_frames=8)

    # After 8 frames, the initial tracks should mostly still be alive.
    survivors = [tid for tid in initial_ids
                 if tid in tracker.tracks and tracker.tracks[tid].is_active]
    survival_ratio = len(survivors) / max(1, len(initial_ids))
    assert survival_ratio >= 0.85, (
        f"Only {survival_ratio:.1%} of initial IDs survived "
        f"({len(survivors)}/{len(initial_ids)})")


def test_calibrate_keeps_parameters_consistent_in_dense_scene():
    gen = SyntheticDataGenerator(width=720, height=540,
                                 num_cells=140, num_frames=2, seed=11)
    gen.generate_cells()
    img, _ = gen.generate_frame(0)

    det = CellDetector(min_area=20, max_area=20000)
    stats = det.calibrate(img)

    assert stats["n"] >= 80
    # Cells are roughly 18-40px in the generator; median should fall in that band.
    assert 8 <= stats["median_diameter"] <= 60
    # expected_max_diameter must exceed the median, but not explode.
    assert det.expected_max_diameter >= stats["median_diameter"]
    assert det.expected_max_diameter <= 300
