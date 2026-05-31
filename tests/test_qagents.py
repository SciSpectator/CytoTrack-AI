"""Tests for morphology-pretraining QAgents."""

from qagents import (ConditionMatcherQAgent, DetectorEnsembleQAgent,
                     CellBirthCuratorQAgent, NoCellBaselineCuratorQAgent,
                     MorphologyTrainingQAgent, TrackingCuratorQAgent,
                     PerCellVisualAgentQAgent, StaticArtifactCuratorQAgent,
                     VideoMorphologyTrainingQAgent, FrameMemoryQAgent,
                     UserDataTrainingQAgent,
                     parse_cell_lines)


def test_parse_cell_lines_splits_commas():
    assert parse_cell_lines("Huh7, MCF7,  HeLa ") == ["Huh7", "MCF7", "HeLa"]


def test_qagent_selects_open_large_public_candidate_for_mcf7():
    plan = MorphologyTrainingQAgent().plan(
        ["MCF7"],
        condition_query="light microscope phase contrast brightfield",
        min_images_per_cell_line=200,
    )
    assert len(plan.class_plans) == 1
    class_plan = plan.class_plans[0]
    assert class_plan.status == "ready"
    assert class_plan.selected_dataset_id == "BBBC021"
    assert class_plan.selected_dataset_licence.startswith("CC-BY")


def test_qagent_reports_blocked_livecell_but_does_not_select_it():
    plan = MorphologyTrainingQAgent().plan(
        ["Huh7"],
        condition_query="phase contrast light microscopy",
        min_images_per_cell_line=200,
    )
    class_plan = plan.class_plans[0]
    livecell = [c for c in class_plan.candidates if c.dataset_id == "LIVECell"]
    assert livecell
    assert livecell[0].downloadable is False
    assert "blocked licence" in livecell[0].reason
    assert class_plan.selected_dataset_id != "LIVECell"


def test_qagent_default_output_dir_uses_model_cache(tmp_path):
    agent = MorphologyTrainingQAgent(project_root=str(tmp_path))
    out = agent.default_output_dir()
    assert out.endswith("model_cache/qagent_morphology_pretraining")
    assert "RESULT" not in out


def test_condition_and_ensemble_qagents_are_executable():
    condition = ConditionMatcherQAgent().match("DIC sparse light microscopy")
    assert condition.modality == "label_free_dic_phase"
    assert condition.density == "sparse"

    detector_choice = DetectorEnsembleQAgent().select(condition)
    tracker_choice = TrackingCuratorQAgent().select(condition)
    assert detector_choice.selected
    assert tracker_choice.selected


def test_tracking_curator_flags_large_identity_jump():
    flags = TrackingCuratorQAgent().flag_suspicious_steps(
        {7: {"boxes": [(0, 0, 10, 10), (100, 0, 10, 10)]}},
        max_step_px=20,
    )
    assert flags
    assert flags[0]["track_id"] == 7


def test_user_data_training_qagent_requires_requested_cell_lines(tmp_path):
    hela = tmp_path / "HeLa"
    hela.mkdir()
    (hela / "frame.png").write_bytes(b"not-a-real-image")

    report = UserDataTrainingQAgent().inspect(str(tmp_path), ["HeLa", "Huh7"])

    assert report["ready"] is False
    assert report["missing_expected_cell_lines"] == ["Huh7"]


def test_guardrail_qagents_block_baseline_artifacts_and_long_jumps():
    baseline = NoCellBaselineCuratorQAgent()
    assert baseline.allow_detections(0, baseline_empty_until_frame=1) is False
    assert baseline.allow_detections(2, baseline_empty_until_frame=1) is True

    artifact = StaticArtifactCuratorQAgent()
    assert artifact.accept_candidate(
        temporal_delta=4.0,
        static_artifact_score=100.0,
        min_temporal_delta=10.0,
        max_static_artifact_score=70.0,
    ) is False
    assert artifact.accept_candidate(
        temporal_delta=20.0,
        static_artifact_score=20.0,
        min_temporal_delta=10.0,
        max_static_artifact_score=70.0,
    ) is True

    birth = CellBirthCuratorQAgent(required_persistence_frames=3)
    assert birth.confirmed([4, 5]) is False
    assert birth.confirmed([4, 5, 6]) is True

    visual = PerCellVisualAgentQAgent(track_id=7, max_center_step_px=15.0)
    assert visual.accept_step((10, 10), (20, 16)) is True
    assert visual.accept_step((10, 10), (40, 16)) is False


def test_video_morphology_training_qagent_skips_empty_baseline():
    agent = VideoMorphologyTrainingQAgent()
    frames = agent.select_training_frames(
        total_frames=67,
        baseline_empty_until_frame=1,
        max_training_frames=8,
    )
    assert frames
    assert min(frames) > 1
    assert max(frames) == 66

    manifest = agent.build_manifest("WM239A", total_frames=67)
    assert manifest["training_source"] == "same_video_before_final_tracking"
    assert manifest["center_policy"] == (
        "train morphology for center detections, not edges")


def test_frame_memory_qagent_predicts_and_gates_next_center():
    memory = FrameMemoryQAgent(max_center_step_px=5.0)
    memory.update(3, 10, (10.0, 10.0))
    memory.update(3, 11, (12.0, 11.0))

    assert memory.predict_center(3) == (14.0, 12.0)
    assert memory.accepts(3, (15.0, 12.0)) is True
    assert memory.accepts(3, (25.0, 12.0)) is False
