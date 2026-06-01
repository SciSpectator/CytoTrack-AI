"""Tests for CellDetector."""

import numpy as np
import cv2

from detector import CellDetector, Detection


def test_detection_dataclass_bbox():
    d = Detection(x=10, y=20, w=30, h=40,
                  center_x=25.0, center_y=40.0, area=400.0)
    assert d.bbox == (10, 20, 30, 40)


def test_detector_handles_empty_input():
    det = CellDetector()
    assert det.detect(None) == []


def test_detect_single_cell(single_cell_frame):
    det = CellDetector(min_area=50, max_area=2000)
    detections = det.detect(single_cell_frame)
    # Should detect at least one object, and it should be roughly centered.
    assert len(detections) >= 1
    best = max(detections, key=lambda d: d.area)
    assert abs(best.center_x - 40) < 8
    assert abs(best.center_y - 40) < 8
    assert best.has_border
    assert best.contour is not None


def test_detect_multiple_cells_in_synthetic_frame(first_frame):
    det = CellDetector(min_area=40, max_area=6000)
    detections = det.detect(first_frame)
    # Synthetic generator creates 12 cells in the conftest fixture
    assert len(detections) >= 5


def test_detect_rejects_obvious_non_cell(debris_frame):
    det = CellDetector(min_area=50, max_area=4000, expected_max_diameter=30)
    detections = det.detect(debris_frame)
    # A thin line should not pass the aspect-ratio filter frequently.
    high_area = [d for d in detections if d.area > 200]
    assert len(high_area) == 0


def test_nms_prefers_higher_confidence():
    det = CellDetector()
    a = Detection(10, 10, 20, 20, 20, 20, 400, confidence=0.9)
    b = Detection(12, 12, 20, 20, 22, 22, 400, confidence=0.5)
    result = det._nms([a, b], iou_thr=0.1)
    assert len(result) == 1
    assert result[0] is a


def test_detection_centroid_aligned_bbox_uses_center_not_edge():
    d = Detection(x=0, y=0, w=20, h=20,
                  center_x=100.0, center_y=80.0, area=400.0)
    assert d.centroid_aligned_bbox() == (90, 70, 20, 20)


def test_whole_cell_border_repair_replaces_fragment_contour():
    det = CellDetector(whole_cell_border=True)
    fragment = np.array([[[10, 10]], [[80, 12]], [[82, 16]], [[12, 14]]],
                        dtype=np.int32)
    partial = Detection(
        x=10, y=10, w=80, h=50,
        center_x=50.0, center_y=35.0,
        area=float(cv2.contourArea(fragment)),
        contour=fragment,
    )

    repaired = det._repair_whole_cell_borders([partial])[0]

    assert repaired.has_border
    assert "whole_cell_border_repaired" in repaired.qc_flags
    assert repaired.area > partial.area
    assert repaired.area / (repaired.w * repaired.h) > 0.55
    assert abs(repaired.center_x - 50) < 3
    assert abs(repaired.center_y - 35) < 3


def test_morphology_scale_suppresses_duplicate_centers():
    det = CellDetector()
    det.set_morphology_constraints(
        median_area_px=400,
        median_diameter_px=24,
        duplicate_center_distance_px=14,
        min_valid_area_px=20,
        max_valid_area_px=2000,
    )
    a = Detection(20, 20, 24, 24, 32, 32, 450, confidence=0.9)
    b = Detection(25, 22, 24, 24, 37, 34, 430, confidence=0.8)

    result = det._nms([a, b], iou_thr=0.95)

    assert len(result) == 1
    assert result[0] is a
