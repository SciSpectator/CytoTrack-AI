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
    assert "NoCellBaselineCuratorQAgent" in plan.qagents
    assert "VideoMorphologyTrainingQAgent" in plan.qagents
    assert "StaticArtifactCuratorQAgent" in plan.qagents
    assert "CellBirthCuratorQAgent" in plan.qagents
    assert "PerCellVisualAgentQAgent" in plan.qagents
    assert "FrameMemoryQAgent" in plan.qagents
    assert "BottomRegionCoverageQAgent" in plan.qagents
    assert "WallArtifactCuratorQAgent" in plan.qagents
    assert "MicroscopyInsetExtractionQAgent" in plan.qagents
    assert "IdentityJumpRepairQAgent" in plan.qagents
    assert "VisualBorderAgent" in plan.qagents
    assert plan.tracking_constraints["track_point"] == (
        "mask_centroid_or_distance_transform_center")
    assert plan.tracking_constraints["cell_count_source"] == (
        "instance_borders_required")
    assert plan.tracking_constraints["cell_line_required_before_tracking"] is True
    assert plan.tracking_constraints[
        "empty_baseline_frames_must_not_seed_tracks"] is True
    assert plan.tracking_constraints[
        "same_video_morphology_training_before_tracking"] is True
    assert plan.tracking_constraints[
        "new_track_requires_temporal_persistence_frames"] == 3
    assert plan.tracking_constraints["one_visual_agent_owns_one_cell"] is True
    assert plan.tracking_constraints[
        "frame_memory_required_for_dense_video"] is True
    assert plan.tracking_constraints["bottom_region_coverage_audit"] is True
    assert plan.tracking_constraints["wall_artifact_curator_required"] is True
    assert plan.tracking_constraints[
        "microscopy_inset_extraction_before_tracking"] is True
    assert plan.tracking_constraints[
        "split_tracks_on_identity_jump_before_metrics"] is True


def test_model_cache_root_is_not_result(tmp_path):
    root = default_model_cache_root(str(tmp_path))
    assert root.endswith("model_cache")
    assert "RESULT" not in root
