"""Tests for the VisualLLMHelper (heuristic fallback path)."""

import numpy as np

from ai_assistant import VisualLLMHelper, VerificationResult


def test_helper_selects_heuristic_when_nothing_installed():
    helper = VisualLLMHelper(prefer="heuristic")
    assert helper.backend == "heuristic"


def test_verify_round_cell_is_classified_cell(single_cell_frame):
    helper = VisualLLMHelper(prefer="heuristic")
    bbox = (20, 20, 40, 40)
    result = helper.verify_cell(single_cell_frame, bbox)
    assert isinstance(result, VerificationResult)
    assert result.backend == "heuristic"
    assert result.is_cell is True
    assert result.label in {"cell", "ambiguous"}


def test_verify_empty_crop_is_debris():
    helper = VisualLLMHelper(prefer="heuristic")
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    result = helper.verify_cell(frame, (0, 0, 2, 2))
    assert result.is_cell is False


def test_follow_returns_nearest_by_default():
    helper = VisualLLMHelper(prefer="heuristic")
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    prev_bbox = (50, 50, 20, 20)
    cands = [(10, 10, 20, 20), (52, 51, 20, 20), (90, 90, 20, 20)]
    idx = helper.follow_cell(frame, prev_bbox, cands)
    assert idx == 1  # nearest candidate
