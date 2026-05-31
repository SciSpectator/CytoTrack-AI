"""Tests for morphology-pretraining QAgents."""

from qagents import (ConditionMatcherQAgent, DetectorEnsembleQAgent,
                     MorphologyTrainingQAgent, TrackingCuratorQAgent,
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
