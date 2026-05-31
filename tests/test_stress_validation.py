"""Tests for stress validation gates."""

from tools.stress_test_30_movies import gt_ratio_is_plausible


def test_gt_ratio_gate_allows_sparse_gt_audit_subset():
    assert gt_ratio_is_plausible(4.1, mean_gt=3.0) is True
    assert gt_ratio_is_plausible(8.5, mean_gt=3.0) is False


def test_gt_ratio_gate_stays_strict_for_dense_gt():
    assert gt_ratio_is_plausible(3.9, mean_gt=30.0) is True
    assert gt_ratio_is_plausible(4.1, mean_gt=30.0) is False
