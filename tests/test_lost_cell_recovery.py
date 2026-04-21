"""Tests for LostCellRecovery (classical path)."""

import numpy as np

from detector import Detection
from lost_cell_recovery import LostCellRecovery, _near_border, _ncc
from tracker import CellTracker


def _draw_cell(img, cx, cy, r=14, color=220):
    import cv2
    cv2.circle(img, (cx, cy), r, (color, color, color), -1)
    cv2.circle(img, (cx, cy), max(3, r // 3),
               (color - 80, color - 80, color - 80), -1)


def test_near_border_flags_edges():
    assert _near_border((0, 0, 20, 20), (100, 100, 3), margin=10) is True
    assert _near_border((40, 40, 20, 20), (100, 100, 3), margin=10) is False


def test_ncc_self_match_is_one():
    import cv2
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    cv2.circle(img, (20, 20), 10, (200, 200, 200), -1)
    # NCC of the same crop with itself should be close to 1.
    assert _ncc(img, img.copy()) > 0.95


def test_classical_recovery_finds_moved_cell():
    # Prev frame: single cell at (50,50)
    prev = np.full((200, 200, 3), 20, dtype=np.uint8)
    _draw_cell(prev, 50, 50, r=14)

    # Current frame: same cell appears at (65,60) — moved 15/10 px
    curr = np.full((200, 200, 3), 20, dtype=np.uint8)
    _draw_cell(curr, 65, 60, r=14)

    # Manually create a tracker with one track that got lost last frame.
    tr = CellTracker(max_missed=1)
    tr.initialize(prev, [Detection(36, 36, 28, 28, 50, 50, 784.0)])
    tid = list(tr.tracks.keys())[0]

    # Simulate a missed update: no detections passed -> track coasts.
    tr.update(curr, detections=[])
    # Force it inactive to mimic a "lost" state.
    tr.tracks[tid].is_active = False

    # New detection in the current frame at the moved location
    current_dets = [Detection(51, 46, 28, 28, 65, 60, 784.0)]

    rec = LostCellRecovery(strategy="heuristic",
                           appearance_threshold=0.5,
                           search_multiplier=4.0)
    results = rec.recover(tr, curr, prev, current_dets)

    assert any(r.recovered for r in results)
    # Track should be active again with bbox near (51, 46)
    assert tr.tracks[tid].is_active is True
    new = tr.tracks[tid].boxes[-1]
    assert abs(new[0] - 51) < 5
    assert abs(new[1] - 46) < 5


def test_recovery_respects_border_margin():
    # Cell at the frame border should be marked as 'border' exit, not recovered.
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    tr = CellTracker(max_missed=1)
    tr.initialize(frame, [Detection(2, 100, 20, 20, 12, 110, 400.0)])
    tid = list(tr.tracks.keys())[0]
    tr.update(frame, detections=[])
    tr.tracks[tid].is_active = False

    rec = LostCellRecovery(strategy="heuristic")
    results = rec.recover(tr, frame, frame, [])
    assert results[0].method == "border"
    assert results[0].recovered is False
