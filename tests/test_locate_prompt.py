"""Tests for LocateAnything prompt construction."""

from detector import CellDetector
import numpy as np

from locate_prompt import (analyze_microscopy_frame,
                           build_adaptive_locate_question,
                           build_locate_question, deterministic_cell_query,
                           is_generic_cell_query)


def test_generic_cell_query_expands_to_microscopy_prompt():
    query = deterministic_cell_query("cell")
    lowered = query.lower()
    assert "microscopy" in lowered
    assert "individual" in lowered
    assert "overlapping" in lowered
    assert "one tight bounding box" in lowered
    assert "center/centroid" in lowered
    assert "not the cell edge" in lowered
    assert "ignore" in lowered


def test_specific_query_is_preserved():
    query = "fluorescent nuclei in the DAPI channel"
    assert deterministic_cell_query(query) == query
    assert not is_generic_cell_query(query)


def test_locate_question_demands_separate_boxes():
    question = build_locate_question("cells")
    lowered = question.lower()
    assert "every instance" in lowered
    assert "separate tight bounding box" in lowered
    assert "do not merge" in lowered
    assert "center/centroid" in lowered
    assert "do not target edges" in lowered
    assert "faint instances" in lowered


def test_detector_resolves_optimized_locate_query_without_model_load():
    detector = CellDetector(sensitivity="locate", locate_use_dspy_prompt=False)
    query = detector._get_locate_prompt_query()
    assert "microscopy" in query.lower()
    assert detector._locate_prompt_method == "deterministic"


def test_adaptive_prompt_for_fluorescence_mentions_bright_on_dark():
    img = np.zeros((80, 80), dtype=np.uint8)
    img[20:35, 20:35] = 220
    question = build_adaptive_locate_question("cells", img).lower()
    profile = analyze_microscopy_frame(img)
    assert profile.polarity == "bright-on-dark"
    assert "bright" in question
    assert "dark background" in question
    assert "every visible cell" in question


def test_adaptive_prompt_for_dic_mentions_texture_not_internal_patches():
    rng = np.random.default_rng(3)
    img = rng.normal(118, 8, size=(80, 80)).clip(0, 255).astype(np.uint8)
    question = build_adaptive_locate_question("cells", img).lower()
    profile = analyze_microscopy_frame(img)
    assert "dic" in profile.modality_hint.lower()
    assert "internal speckles" in question
    assert "same cell" in question
