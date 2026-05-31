"""Tests for self-repairing detector curators."""

import os

from self_repair import (CellBorderCuratorQAgent, CountStabilityQAgent,
                         SelfRepairingDetectorLoop)


def test_self_repair_loop_selects_border_detections(single_cell_frame, tmp_path):
    loop = SelfRepairingDetectorLoop(
        min_area=50,
        max_area=2000,
        sensitivities=["normal", "high"],
    )
    detector, detections, report = loop.run(
        single_cell_frame,
        output_dir=str(tmp_path),
    )

    assert detector is not None
    assert detections
    assert report.selected_count == len(detections)
    assert report.selected_sensitivity in {"normal", "high"}
    assert any(d.has_border for d in detections)
    assert os.path.exists(tmp_path / "qc" / "detector_self_repair_report.json")
    assert os.path.exists(tmp_path / "qc" / "detector_self_repair_first_frame.png")


def test_border_and_count_curators_score_detections(single_cell_frame):
    loop = SelfRepairingDetectorLoop(
        min_area=50,
        max_area=2000,
        sensitivities=["normal"],
    )
    _, detections, _ = loop.run(single_cell_frame)

    assert CellBorderCuratorQAgent().border_fraction(detections) > 0
    assert CountStabilityQAgent().duplicate_pairs(detections) >= 0
