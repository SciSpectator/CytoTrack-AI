"""End-to-end pipeline test: synthetic frames -> detect -> track -> analyze."""

import numpy as np

from analyzer import MigrationAnalyzer
from debris_reasoner import DebrisReasoner, filter_debris
from detector import CellDetector
from tracker import CellTracker


def test_full_pipeline_runs_on_synthetic_frames(synthetic_frames):
    frames, _ = synthetic_frames

    detector = CellDetector(min_area=40, max_area=6000)
    first = frames[0]

    raw = detector.detect(first)
    assert len(raw) > 0

    reasoner = DebrisReasoner(strategy="heuristic")
    dets0, _ = filter_debris(detector, first, raw, reasoner=reasoner)
    assert len(dets0) > 0

    tracker = CellTracker(max_missed=5)
    tracker.initialize(first, dets0)
    assert tracker.active_count == len(dets0)

    for f in frames[1:]:
        raw = detector.detect(f)
        dets, _ = filter_debris(detector, f, raw, reasoner=reasoner)
        tracker.update(f, detections=dets)

    tracks = tracker.get_tracks(min_length=3)
    assert len(tracks) > 0

    analyzer = MigrationAnalyzer(1.0, 1.0, 60.0)
    detailed, summary = analyzer.analyze(tracks)

    assert not summary.empty
    assert "Avg_Velocity_um_min" in summary.columns
    assert "CDE" in summary.columns
    assert (summary["Frames"] > 0).all()
    assert (summary["Avg_Velocity_um_min"] >= 0).all()
