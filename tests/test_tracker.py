"""Tests for CellTracker (Kalman + Hungarian)."""

import math

import numpy as np

from detector import Detection
from tracker import CellTracker, KalmanBox, _bbox_iou


class _Approx:
    def __init__(self, value, abs_tol=1e-6):
        self.value = value
        self.abs_tol = abs_tol

    def __eq__(self, other):
        return math.isclose(float(other), float(self.value), abs_tol=self.abs_tol)


def _approx(v, abs=1e-6):
    return _Approx(v, abs_tol=abs)


class _PytestCompat:
    approx = staticmethod(_approx)


pytest = _PytestCompat()


def test_bbox_iou_identity():
    assert _bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_bbox_iou_disjoint():
    assert _bbox_iou((0, 0, 10, 10), (20, 20, 10, 10)) == pytest.approx(0.0)


def test_kalman_predict_shape():
    kf = KalmanBox((10, 20, 30, 40))
    out = kf.predict()
    assert len(out) == 4
    assert all(isinstance(v, int) for v in out)


def test_kalman_update_pulls_toward_measurement():
    kf = KalmanBox((0, 0, 20, 20))
    kf.predict()
    # Feed several consistent measurements
    for _ in range(6):
        kf.update((50, 50, 20, 20))
        kf.predict()
    cx, cy = kf.center
    assert 40 < cx < 60
    assert 40 < cy < 60


def test_tracker_initialize_spawns_tracks():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    dets = [
        Detection(10, 10, 20, 20, 20, 20, 400.0),
        Detection(60, 40, 20, 20, 70, 50, 400.0),
    ]
    tr = CellTracker()
    tr.initialize(frame, dets)
    assert tr.active_count == 2


def test_tracker_matches_moving_cell():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    dets0 = [Detection(10, 10, 20, 20, 20, 20, 400.0)]
    tr = CellTracker()
    tr.initialize(frame, dets0)
    tid = list(tr.tracks.keys())[0]

    # Same cell shifted 5px right and 3px down, frame-after-frame.
    for i in range(1, 5):
        x = 10 + 5 * i
        y = 10 + 3 * i
        dets = [Detection(x, y, 20, 20, x + 10, y + 10, 400.0)]
        tr.update(frame, detections=dets)

    assert tid in tr.tracks
    assert tr.tracks[tid].hits >= 4
    last_box = tr.tracks[tid].boxes[-1]
    # Should end roughly at (30, 22)+offset — let's just verify drift
    assert last_box[0] > 15
    assert last_box[1] > 12


def test_tracker_spawns_new_track_for_appearance():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    tr = CellTracker()
    tr.initialize(frame, [Detection(10, 10, 20, 20, 20, 20, 400.0)])
    starting_ids = set(tr.tracks.keys())

    # Second frame: original cell plus a new cell appears.
    tr.update(frame, detections=[
        Detection(12, 12, 20, 20, 22, 22, 400.0),
        Detection(90, 90, 20, 20, 100, 100, 400.0),
    ])
    new_ids = set(tr.tracks.keys()) - starting_ids
    assert len(new_ids) == 1


def test_tracker_coasts_then_loses_track():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    tr = CellTracker(max_missed=3)
    tr.initialize(frame, [Detection(50, 50, 20, 20, 60, 60, 400.0)])
    tid = list(tr.tracks.keys())[0]

    for _ in range(5):
        tr.update(frame, detections=[])

    assert tr.tracks[tid].is_active is False


def test_get_tracks_respects_min_length():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    tr = CellTracker()
    tr.initialize(frame, [Detection(50, 50, 20, 20, 60, 60, 400.0)])
    assert tr.get_tracks(min_length=10) == {}
    assert tr.get_tracks(min_length=1) != {}
