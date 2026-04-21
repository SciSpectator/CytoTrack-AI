"""Tests for MigrationAnalyzer."""

import math

import numpy as np
import pandas as pd

from analyzer import MigrationAnalyzer


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


def _straight_track(n: int = 10, dx: float = 2.0, dy: float = 0.0, w: int = 20):
    boxes = []
    for i in range(n):
        boxes.append((int(i * dx), int(i * dy), w, w))
    return {0: {"boxes": boxes, "cell_type": "Cell"}}


def test_analyze_produces_both_frames():
    a = MigrationAnalyzer(pixel_size_x=1.0, pixel_size_y=1.0, time_per_frame=60.0)
    detailed, summary = a.analyze(_straight_track())
    assert isinstance(detailed, pd.DataFrame)
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == 1


def test_straight_line_has_cde_near_one():
    a = MigrationAnalyzer(1.0, 1.0, 60.0)
    _, summary = a.analyze(_straight_track(n=20, dx=3.0, dy=0.0))
    assert summary["CDE"].iloc[0] == pytest.approx(1.0, abs=1e-6)


def test_velocity_units_um_per_minute():
    # 1 px/frame with 1 µm/px and 60s/frame = 1 µm/min
    a = MigrationAnalyzer(1.0, 1.0, 60.0)
    _, summary = a.analyze(_straight_track(n=10, dx=1.0, dy=0.0))
    assert summary["Avg_Velocity_um_min"].iloc[0] == pytest.approx(1.0, abs=1e-6)


def test_msd_increases_with_lag():
    a = MigrationAnalyzer(1.0, 1.0, 60.0)
    _, summary = a.analyze(_straight_track(n=30, dx=2.0, dy=0.0))
    assert summary["MSD_10"].iloc[0] < summary["MSD_20"].iloc[0]


def test_compare_types_returns_dataframe():
    a = MigrationAnalyzer(1.0, 1.0, 60.0)

    summary = pd.DataFrame({
        "TrackID": range(12),
        "Cell_Type": ["A"] * 6 + ["B"] * 6,
        "Avg_Velocity_um_min": [1, 1.1, 1.2, 1.05, 0.95, 1.0,
                                2.0, 2.1, 2.2, 2.05, 1.95, 2.0],
        "Displacement_um": [10] * 6 + [20] * 6,
        "Total_Distance_um": [12] * 6 + [22] * 6,
        "CDE": [0.8] * 6 + [0.9] * 6,
        "Persistence": [0.7] * 6 + [0.75] * 6,
    })

    cmp_df = a.compare_types(summary)
    assert isinstance(cmp_df, pd.DataFrame)
    assert (cmp_df["T_test_p_value"] < 0.05).any()


def test_type_summary_has_per_type_rows():
    a = MigrationAnalyzer(1.0, 1.0, 60.0)
    summary = pd.DataFrame({
        "TrackID": [0, 1, 2, 3],
        "Cell_Type": ["A", "A", "B", "B"],
        "Avg_Velocity_um_min": [1.0, 1.2, 2.0, 2.2],
        "Displacement_um": [5.0, 6.0, 8.0, 9.0],
        "Total_Distance_um": [7.0, 8.0, 10.0, 11.0],
        "CDE": [0.7, 0.75, 0.8, 0.85],
        "Persistence": [0.4, 0.5, 0.6, 0.7],
        "Duration_min": [10, 11, 12, 13],
    })
    ts = a.get_type_summary(summary)
    assert "Cell_Type" in ts.columns
    assert len(ts) == 2
