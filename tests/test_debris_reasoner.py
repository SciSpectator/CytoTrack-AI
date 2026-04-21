"""Tests for the DSPy-powered DebrisReasoner (fallback path)."""

import numpy as np

from debris_reasoner import (DebrisJudgement, DebrisReasoner,
                             extract_features, filter_debris)
from detector import Detection


def test_extract_features_for_round_cell(single_cell_frame):
    feats = extract_features(single_cell_frame, (20, 20, 40, 40))
    assert feats.circularity > 0.3
    assert 0.05 < feats.fg_fraction < 0.9
    assert feats.area > 50


def test_reasoner_judges_round_cell_as_cell(single_cell_frame):
    reasoner = DebrisReasoner(strategy="heuristic")
    verdict = reasoner.judge(single_cell_frame, (20, 20, 40, 40))
    assert isinstance(verdict, DebrisJudgement)
    assert verdict.method == "heuristic"
    assert verdict.is_cell is True


def test_reasoner_judges_debris_as_not_cell(debris_frame):
    reasoner = DebrisReasoner(strategy="heuristic")
    verdict = reasoner.judge(debris_frame, (0, 35, 80, 10))
    assert verdict.is_cell is False


def test_filter_debris_removes_obvious_junk(first_frame):
    dets = [
        Detection(5, 5, 6, 4, 8, 7, 20.0),                 # tiny / too small
        Detection(60, 60, 20, 20, 70, 70, 400.0),          # plausible cell
        Detection(30, 30, 80, 3, 70, 31, 240.0),           # very elongated
    ]
    reasoner = DebrisReasoner(strategy="heuristic")
    kept, rejected = filter_debris(None, first_frame, dets, reasoner=reasoner)
    assert any(r[1] == "obvious-debris" for r in rejected)
    assert len(kept) <= 2
