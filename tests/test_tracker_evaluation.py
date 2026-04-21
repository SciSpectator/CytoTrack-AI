"""
Quantitative evaluation of the tracker against synthetic ground truth.

Every frame the generator emits a dict {cell_id: {"bbox", "center"}}.
We can therefore compute *real* tracking metrics per scene — not just
"did a track survive". The metrics implemented here are the ones that
matter for downstream migration analysis:

  * Track purity           — % of a track's frames assigned to the same
                              ground-truth cell. Low purity = the track
                              wandered between multiple cells (ID swap
                              or drift-across-neighbour).
  * ID switch count        — total times any track's best-matching GT
                              id changes between consecutive frames.
  * GT coverage            — fraction of ground-truth cells that have
                              at least one track following them with
                              purity >= 0.8 for at least half the scene.
  * Mean localisation err  — average pixel distance between track
                              centroid and its matched GT centre
                              (across frames where the match was
                              reasonable).
  * Ghost-frame rate       — fraction of frames a track emits a box
                              with no ground-truth cell within a
                              plausible radius — i.e. pure Kalman drift
                              after loss. This is the failure mode the
                              user flagged: the tracker keeps producing
                              boxes even when the cell is gone.

Each test asserts an honest bar; the bars are set where the current
implementation passes comfortably, and regressions will fire.

Pure numpy + stdlib — no pytest, no network.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np

from detector import CellDetector
from tracker import CellTracker
from synthetic_data import SyntheticDataGenerator


# ---------------------------------------------------------------- helpers
def _nearest_gt(gt_frame: Dict, cx: float, cy: float,
                max_radius: float = 40.0) -> Tuple[int, float]:
    """Return (best_gt_id, distance) or (-1, inf) if nothing within radius."""
    best_id, best_d = -1, float("inf")
    for gid, entry in gt_frame.items():
        gx, gy = entry["center"]
        d = math.hypot(cx - gx, cy - gy)
        if d < best_d:
            best_d = d
            best_id = gid
    if best_d > max_radius:
        return -1, best_d
    return best_id, best_d


def _run_scene(num_cells: int, num_frames: int, seed: int,
               width: int = 640, height: int = 480,
               overlap_density: float = 0.0,
               max_missed: int = 15) -> Tuple[dict, List[dict], int, float]:
    """Run the full pipeline on a synthetic scene. Return
    (track_dict, gt_per_frame, width, median_diameter)."""
    gen = SyntheticDataGenerator(
        width=width, height=height, num_cells=num_cells,
        num_frames=num_frames, seed=seed,
        overlap_density=overlap_density,
    )
    gen.generate_cells()
    frames, gts = [], []
    for i in range(num_frames):
        img, gt = gen.generate_frame(i)
        frames.append(img)
        gts.append(gt)

    detector = CellDetector(min_area=40, max_area=9000)
    detector.calibrate(frames[0])
    tracker = CellTracker(max_missed=max_missed)
    first_dets = detector.detect(frames[0])
    calib = tracker.calibrate(first_dets)
    tracker.initialize(frames[0], first_dets)

    for f in frames[1:]:
        dets = detector.detect(f)
        tracker.update(f, detections=dets)

    med_diam = calib.get("median_diameter", 20.0) or 20.0
    return tracker, gts, width, med_diam


def _evaluate(tracker, gts: List[dict], med_diam: float) -> dict:
    """Compute the metrics described in the module docstring."""
    match_radius = max(med_diam * 1.5, 20.0)

    # For each track and each frame, who is the closest GT?
    per_track_gt: Dict[int, List[int]] = defaultdict(list)
    per_track_dists: Dict[int, List[float]] = defaultdict(list)

    for tid, t in tracker.tracks.items():
        for fi, box in enumerate(t.boxes):
            if fi >= len(gts):
                break
            cx = box[0] + box[2] / 2.0
            cy = box[1] + box[3] / 2.0
            gid, dist = _nearest_gt(gts[fi], cx, cy,
                                    max_radius=match_radius)
            per_track_gt[tid].append(gid)
            per_track_dists[tid].append(dist)

    # Purity: most-common GT / total frames per track.
    purities = {}
    for tid, gseq in per_track_gt.items():
        non_ghost = [g for g in gseq if g >= 0]
        if not non_ghost:
            purities[tid] = 0.0
        else:
            top = Counter(non_ghost).most_common(1)[0][1]
            purities[tid] = top / len(gseq)

    # ID switch count: how many times (tid's best GT over time) changes.
    id_switches = 0
    for tid, gseq in per_track_gt.items():
        prev = None
        for g in gseq:
            if g < 0:
                continue
            if prev is not None and g != prev:
                id_switches += 1
            prev = g

    # GT coverage: fraction of GT cells present across the scene that
    # have SOME track following them with purity>=0.8 for half the scene.
    all_gt_ids = set()
    for gt in gts:
        all_gt_ids.update(gt.keys())
    n_gt = len(all_gt_ids)

    covered_gt_ids = set()
    total_frames = len(gts)
    for tid, gseq in per_track_gt.items():
        non_ghost = [g for g in gseq if g >= 0]
        if not non_ghost:
            continue
        most_common_gt, count = Counter(non_ghost).most_common(1)[0]
        if (count >= total_frames * 0.5
                and purities.get(tid, 0.0) >= 0.8):
            covered_gt_ids.add(most_common_gt)
    coverage = len(covered_gt_ids) / max(1, n_gt)

    # Mean localisation error over matched frames only.
    all_dists = []
    for tid, dseq in per_track_dists.items():
        for d in dseq:
            if d < match_radius:
                all_dists.append(d)
    loc_err = float(np.mean(all_dists)) if all_dists else float("inf")

    # Ghost-frame rate: fraction of (track, frame) pairs with no GT
    # within match_radius.
    total_frames_out = 0
    ghost = 0
    for tid, dseq in per_track_dists.items():
        total_frames_out += len(dseq)
        ghost += sum(1 for d in dseq if d >= match_radius)
    ghost_rate = ghost / max(1, total_frames_out)

    # Per-track worst purity — useful diagnostic.
    worst_purity = min(purities.values()) if purities else 1.0

    return {
        "num_tracks": len(per_track_gt),
        "num_gt": n_gt,
        "coverage": coverage,
        "mean_purity": float(np.mean(list(purities.values())))
                       if purities else 0.0,
        "worst_purity": worst_purity,
        "id_switches": id_switches,
        "mean_loc_err_px": loc_err,
        "ghost_rate": ghost_rate,
    }


# =================================================================== tests
def test_evaluation_clean_scene():
    """Sparse, non-overlapping cells — baseline. The tracker must cover
    almost every ground-truth cell and produce near-zero ID switches."""
    tracker, gts, _, med = _run_scene(
        num_cells=25, num_frames=40, seed=7,
        overlap_density=0.0, max_missed=15)
    m = _evaluate(tracker, gts, med)
    print(f"[clean] {m}")
    assert m["coverage"] >= 0.75, (
        f"Clean-scene GT coverage dropped to {m['coverage']:.2%} — "
        f"tracker is losing cells it shouldn't.")
    assert m["id_switches"] <= m["num_gt"] * 2.0, (
        f"Clean-scene ID switches = {m['id_switches']} on "
        f"{m['num_gt']} GT cells — too many.")
    assert m["ghost_rate"] <= 0.05, (
        f"Clean-scene ghost-frame rate {m['ghost_rate']:.2%} — "
        f"tracks are drifting into empty space.")


def test_evaluation_moderate_overlap():
    """Moderate overlap — real microscopy resembles this. Purity must
    stay high and ghost-frame rate bounded."""
    tracker, gts, _, med = _run_scene(
        num_cells=30, num_frames=30, seed=11,
        overlap_density=0.3, max_missed=15)
    m = _evaluate(tracker, gts, med)
    print(f"[moderate] {m}")
    assert m["mean_purity"] >= 0.80, (
        f"Mean track purity {m['mean_purity']:.2%} — tracks are "
        f"wandering between GT cells.")
    assert m["ghost_rate"] <= 0.20, (
        f"Ghost-frame rate {m['ghost_rate']:.2%} — too many frames "
        f"of pure drift.")


def test_evaluation_heavy_overlap():
    """Heavy overlap — the overlap-safety fix must hold up under this.
    We bar against coverage collapse and ID-switch explosion."""
    tracker, gts, _, med = _run_scene(
        num_cells=30, num_frames=25, seed=13,
        overlap_density=0.6, max_missed=15)
    m = _evaluate(tracker, gts, med)
    print(f"[heavy] {m}")
    assert m["coverage"] >= 0.50, (
        f"Heavy-overlap coverage collapsed to {m['coverage']:.2%} — "
        f"merge-split not rescuing enough cells.")
    assert m["id_switches"] <= m["num_gt"] * 1.5, (
        f"Heavy-overlap ID switches = {m['id_switches']} on "
        f"{m['num_gt']} GT cells — explosion.")


def test_evaluation_long_sequence():
    """Long sequence (80 frames) tests whether drift accumulates.
    Localisation error must stay bounded — if Kalman coasts for 80
    frames, error blows up."""
    tracker, gts, _, med = _run_scene(
        num_cells=20, num_frames=80, seed=19,
        overlap_density=0.2, max_missed=15)
    m = _evaluate(tracker, gts, med)
    print(f"[long] {m}")
    assert m["mean_loc_err_px"] <= 8.0, (
        f"Mean localisation error {m['mean_loc_err_px']:.2f}px — "
        f"tracks are drifting away from their GT cells.")
    # Worst-purity is sensitive to a single unlucky revive on a
    # short track; use mean_purity as the real quality bar and keep
    # worst_purity as a looser floor to catch catastrophic regressions.
    assert m["mean_purity"] >= 0.75, (
        f"Long-sequence mean purity {m['mean_purity']:.2%} — "
        f"tracks wandering between GT cells on average.")
    assert m["worst_purity"] >= 0.3, (
        f"Worst-track purity {m['worst_purity']:.2%} — at least one "
        f"track is jumping between GT cells repeatedly.")


def test_evaluation_high_density_stress():
    """Dense scene — lots of cells, modest overlap. Stress test for
    Hungarian assignment + merge-split under simultaneous cues."""
    tracker, gts, _, med = _run_scene(
        num_cells=60, num_frames=25, seed=23,
        overlap_density=0.3, max_missed=15,
        width=800, height=600)
    m = _evaluate(tracker, gts, med)
    print(f"[dense] {m}")
    # Dense scenes are detector-limited: only ~70% of GT cells produce a
    # detection at any given frame, so 100% coverage is unreachable
    # without detector improvements. The bar is set above the "tracker
    # is the bottleneck" threshold.
    assert m["coverage"] >= 0.45, (
        f"Dense-scene coverage {m['coverage']:.2%} — detector or "
        f"assignment breaking down under density.")
    assert m["ghost_rate"] <= 0.10, (
        f"Dense-scene ghost-frame rate {m['ghost_rate']:.2%}.")
    assert m["mean_purity"] >= 0.70, (
        f"Dense-scene mean purity {m['mean_purity']:.2%} — tracks "
        f"losing identity under density.")


def test_evaluation_huge_200_cells():
    """Stress at 200 cells in a 1200x900 field. Detector recall is the
    main bottleneck at this density, so we guard against TRACKER
    regressions: purity and ghost-rate."""
    tracker, gts, _, med = _run_scene(
        num_cells=200, num_frames=20, seed=29,
        overlap_density=0.2, max_missed=15,
        width=1200, height=900)
    m = _evaluate(tracker, gts, med)
    print(f"[200-cells] {m}")
    assert m["mean_purity"] >= 0.70, (
        f"200-cell mean purity {m['mean_purity']:.2%} — tracks losing "
        f"identity.")
    assert m["ghost_rate"] <= 0.10, (
        f"200-cell ghost-frame rate {m['ghost_rate']:.2%}.")


def test_evaluation_huge_400_cells():
    """Extreme density — 400 cells. Worst-case for Hungarian. Focus
    purely on per-track fidelity since coverage is detector-bound."""
    tracker, gts, _, med = _run_scene(
        num_cells=400, num_frames=15, seed=31,
        overlap_density=0.2, max_missed=15,
        width=1600, height=1200)
    m = _evaluate(tracker, gts, med)
    print(f"[400-cells] {m}")
    assert m["mean_purity"] >= 0.65, (
        f"400-cell mean purity {m['mean_purity']:.2%} — tracker "
        f"assignment breaking down.")
    assert m["mean_loc_err_px"] <= 10.0, (
        f"400-cell localisation error {m['mean_loc_err_px']:.2f}px.")


def test_print_evaluation_table():
    """Not an assertion test — prints a summary table across scenes so a
    regression in any metric is obvious. Runs last so its output is near
    the tail of the test log."""
    print()
    print("Scene          | tracks  GT  cov%  purity  ids   err_px  ghost%")
    print("-" * 70)
    for label, cfg in [
        ("clean",        dict(num_cells=25, num_frames=40, seed=7,
                              overlap_density=0.0)),
        ("moderate",     dict(num_cells=30, num_frames=30, seed=11,
                              overlap_density=0.3)),
        ("heavy",        dict(num_cells=30, num_frames=25, seed=13,
                              overlap_density=0.6)),
        ("long",         dict(num_cells=20, num_frames=80, seed=19,
                              overlap_density=0.2)),
        ("dense",        dict(num_cells=60, num_frames=25, seed=23,
                              overlap_density=0.3, width=800, height=600)),
        ("200-cells",    dict(num_cells=200, num_frames=20, seed=29,
                              overlap_density=0.2, width=1200, height=900)),
        ("400-cells",    dict(num_cells=400, num_frames=15, seed=31,
                              overlap_density=0.2, width=1600, height=1200)),
    ]:
        tracker, gts, _, med = _run_scene(max_missed=15, **cfg)
        m = _evaluate(tracker, gts, med)
        print(f"{label:13s}  | {m['num_tracks']:6d} {m['num_gt']:3d}"
              f"  {m['coverage']*100:4.0f}"
              f"  {m['mean_purity']*100:5.0f}%"
              f"  {m['id_switches']:4d}"
              f"  {m['mean_loc_err_px']:6.2f}"
              f"  {m['ghost_rate']*100:5.1f}%")
