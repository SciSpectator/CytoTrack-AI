"""Tests for executable pipeline architecture contracts."""

import os

from pipeline_architecture import (
    build_quality_first_run_plan,
    default_model_cache_root,
)


def test_quality_first_plan_contains_backend_registries(tmp_path):
    plan = build_quality_first_run_plan(
        ["HeLa"],
        condition_query="DIC light microscopy",
        project_root=str(tmp_path),
    )

    assert plan.quality_mode == "quality_first"
    assert plan.cell_lines == ["HeLa"]
    assert plan.model_cache_root == os.path.join(str(tmp_path), "model_cache")
    assert any(b["name"] == "cellpose-sam" for b in plan.segmentation_backends)
    assert any(b["name"] == "trackastra" for b in plan.tracking_backends)
    assert "TrackingCuratorQAgent" in plan.qagents
    assert "UserDataTrainingQAgent" in plan.qagents
    assert "CellBorderCuratorQAgent" in plan.qagents
    assert "SelfRepairCoordinatorQAgent" in plan.qagents
    assert "VisualBorderAgent" in plan.qagents
    assert plan.tracking_constraints["track_point"] == (
        "mask_centroid_or_distance_transform_center")
    assert plan.tracking_constraints["cell_count_source"] == (
        "instance_borders_required")
    assert plan.tracking_constraints["cell_line_required_before_tracking"] is True


def test_model_cache_root_is_not_result(tmp_path):
    root = default_model_cache_root(str(tmp_path))
    assert root.endswith("model_cache")
    assert "RESULT" not in root
