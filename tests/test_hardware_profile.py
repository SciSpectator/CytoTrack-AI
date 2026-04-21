"""Tests for the hardware profiler / latency tuner.

These tests verify:
  * tier policy monotonicity — weaker hardware -> smaller latency knobs,
    never the reverse;
  * *accuracy-critical* attributes are NOT touched by the tuner;
  * the detector honours the latency toggles without losing accuracy on
    the core strategy set.
"""

import inspect
import numpy as np

from hardware_profile import (HardwareProfile, _assign_tier,
                              _knobs_for_tier, detect_hardware)
from detector import CellDetector


def test_tiers_assigned_sensibly():
    tier, _ = _assign_tier(has_cuda=True, vram_gb=24, ram_gb=32, cpu_count=16)
    assert tier == "extreme"
    tier, _ = _assign_tier(has_cuda=True, vram_gb=10, ram_gb=16, cpu_count=8)
    assert tier == "high"
    tier, _ = _assign_tier(has_cuda=True, vram_gb=5, ram_gb=16, cpu_count=8)
    assert tier == "mid"
    tier, _ = _assign_tier(has_cuda=False, vram_gb=0, ram_gb=16, cpu_count=12)
    assert tier == "mid"
    tier, _ = _assign_tier(has_cuda=False, vram_gb=0, ram_gb=4, cpu_count=2)
    assert tier == "low"


def test_knobs_monotonic_in_batch_and_workers():
    order = ["low", "mid", "high", "extreme"]
    # cpu_count = 16, has_cuda True for mid/high/extreme
    last_batch = 0
    last_workers = 0
    for t in order:
        k = _knobs_for_tier(t, has_cuda=(t != "low"),
                            vram_gb=16 if t in ("high", "extreme") else 4,
                            cpu_count=16)
        assert k["classifier_batch_size"] >= last_batch, (
            f"batch regressed at tier {t}")
        assert k["num_workers"] >= last_workers, (
            f"workers regressed at tier {t}")
        last_batch = k["classifier_batch_size"]
        last_workers = k["num_workers"]


def test_low_tier_preserves_core_strategies_turns_off_extras_only():
    """
    Critical rule: degrading hardware must NOT drop accuracy.  The low
    tier turns off optional/redundant strategies (blob, hough) but never
    the core adaptive/Otsu/watershed trio.
    """
    low = _knobs_for_tier("low", has_cuda=False, vram_gb=0, cpu_count=2)
    assert low["use_blob_detector"] is False
    assert low["use_hough_circles"] is False
    # Extreme tier keeps everything on
    ext = _knobs_for_tier("extreme", has_cuda=True, vram_gb=24, cpu_count=16)
    assert ext["use_blob_detector"] is True
    assert ext["use_hough_circles"] is True


def test_knobs_never_include_accuracy_parameters():
    """
    Regression guard: knob names must not include any accuracy-altering
    parameter. If someone adds e.g. `classifier_input_size` or
    `detection_threshold` to _knobs_for_tier, this test fails loud.
    """
    forbidden = {
        # Detector thresholds (changing these shifts recall/precision)
        "min_area", "max_area", "iou_threshold", "confidence_threshold",
        "expected_max_diameter", "merge_area_factor",
        # Classifier (changing these changes model behaviour)
        "classifier_input_size", "classifier_resolution",
        "classifier_model", "classifier_weights", "classifier_classes",
        # Tracker (these would shift ID-assignment logic)
        "max_missed", "max_distance", "iou_threshold_tracker",
        # Reasoning (strategy=react/cot is an accuracy-quality choice)
        "debris_strategy", "reasoner_strategy",
    }
    for tier in ("low", "mid", "high", "extreme"):
        k = _knobs_for_tier(tier, has_cuda=True, vram_gb=16, cpu_count=8)
        overlap = set(k.keys()) & forbidden
        assert not overlap, (
            f"knobs leak accuracy-critical parameter(s) at {tier}: {overlap}")


def test_detect_hardware_runs_on_this_machine():
    p = detect_hardware()
    assert isinstance(p, HardwareProfile)
    assert p.tier in ("low", "mid", "high", "extreme")
    assert p.cpu_count >= 1
    assert p.ram_gb >= 0.5
    # Summary/long description render without crashing
    assert "Tier:" in p.summary()
    assert "Runtime knobs" in p.long_description()


def test_detector_accepts_latency_toggles_without_losing_core_accuracy():
    """
    Turning off blob + hough (the latency-optional strategies) should
    still let the detector find a clean isolated cell via adaptive /
    Otsu / watershed.
    """
    import cv2
    img = np.full((200, 200, 3), 20, dtype=np.uint8)
    cv2.circle(img, (100, 100), 14, (220, 220, 220), -1)
    cv2.circle(img, (100, 100), 5, (140, 140, 140), -1)

    fast = CellDetector(min_area=50, max_area=5000,
                        use_blob_detector=False, use_hough_circles=False)
    full = CellDetector(min_area=50, max_area=5000,
                        use_blob_detector=True, use_hough_circles=True)

    dets_fast = fast.detect(img)
    dets_full = full.detect(img)

    # Both configurations must still detect the cell.
    assert len(dets_fast) >= 1, "fast config missed an isolated cell"
    assert len(dets_full) >= 1, "full config missed an isolated cell"
    # Centers should agree within a couple of pixels.
    cx_f = dets_fast[0].center_x
    cy_f = dets_fast[0].center_y
    assert abs(cx_f - 100) < 6 and abs(cy_f - 100) < 6
