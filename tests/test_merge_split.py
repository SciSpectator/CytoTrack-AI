"""Tests for merged-cell splitting via distance-transform watershed."""

import cv2
import numpy as np

from detector import CellDetector


def _bg_field(H=200, W=200, val=20):
    return np.full((H, W, 3), val, dtype=np.uint8)


def _draw_cell(img, cx, cy, r=12, color=220):
    cv2.circle(img, (cx, cy), r, (color, color, color), -1)


def test_calibrate_learns_cell_size():
    img = _bg_field()
    # 8 isolated, similarly sized cells
    centers = [(30, 40), (80, 40), (130, 40), (170, 40),
               (40, 150), (90, 150), (140, 150), (180, 150)]
    for cx, cy in centers:
        _draw_cell(img, cx, cy, r=11)

    det = CellDetector(min_area=20, max_area=10000)
    stats = det.calibrate(img)

    assert stats["n"] >= 6
    # Median diameter should be around 22 (2*r) — allow slack for threshold bloat
    assert 15 <= stats["median_diameter"] <= 40
    # Detector should have set internal bounds sensibly
    assert det.expected_max_diameter >= stats["median_diameter"]
    assert det.min_area > 0
    assert det.max_area > det.min_area


def test_splitter_separates_touching_cells():
    """
    Place two overlapping circles so their union contour is a single large
    peanut-shape. After calibrate() learns the typical cell area, the
    splitter should detect the oversized blob and break it into two.
    """
    img = _bg_field(H=200, W=200)
    # Reference cells (establish the 'typical' size)
    ref = [(30, 30), (80, 30), (130, 30), (170, 30),
           (30, 80), (80, 180), (150, 180), (170, 80)]
    for cx, cy in ref:
        _draw_cell(img, cx, cy, r=10)

    # A peanut of two touching cells at the centre
    _draw_cell(img, 90, 110, r=11)
    _draw_cell(img, 110, 110, r=11)

    det = CellDetector(min_area=20, max_area=20000)
    det.calibrate(img)

    out = det.detect(img)

    # Count detections inside a 40-px box around the merged pair
    near = [d for d in out
            if 70 <= d.center_x <= 130 and 90 <= d.center_y <= 130]
    assert len(near) >= 2, (
        f"Expected the merged pair to be split into >=2 detections, "
        f"got {len(near)} (total dets={len(out)})"
    )


def test_splitter_leaves_isolated_cells_alone():
    """A well-separated normal-sized cell must not be fragmented."""
    img = _bg_field()
    centers = [(40, 40), (100, 40), (160, 40),
               (40, 100), (100, 100), (160, 100),
               (40, 160), (100, 160), (160, 160)]
    for cx, cy in centers:
        _draw_cell(img, cx, cy, r=10)

    det = CellDetector(min_area=20, max_area=10000)
    det.calibrate(img)
    out = det.detect(img)

    # Should find ~9 cells, not dozens of fragments
    assert 7 <= len(out) <= 12
