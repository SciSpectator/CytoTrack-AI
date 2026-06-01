"""Tests for learned tracking-linker features."""

import numpy as np

from tracking_linker import FEATURE_NAMES, build_link_features, bbox_iou


def test_link_features_have_stable_order_and_values():
    features = build_link_features(
        (0, 0, 10, 10),
        (3, 4, 10, 10),
        source_area=100,
        target_area=100,
        appearance_similarity=0.9,
    )

    assert len(features) == len(FEATURE_NAMES)
    assert np.isclose(features[FEATURE_NAMES.index("distance")], 5.0)
    assert np.isclose(features[FEATURE_NAMES.index("normalized_distance")], 0.5)
    assert np.isclose(features[FEATURE_NAMES.index("appearance_similarity")], 0.9)
    assert features[FEATURE_NAMES.index("abs_log_area_ratio")] == 0


def test_bbox_iou_for_overlapping_boxes():
    iou = bbox_iou((0, 0, 10, 10), (5, 5, 10, 10))
    assert 0 < iou < 1
