# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**CytoTrack AI** — desktop (PyQt5) application for cell migration tracking and phenotype analysis on time-lapse microscopy. SORT-style tracker (Kalman + Hungarian + appearance) over a multi-strategy detector, with optional phenotype-classifier training from local folders or open-licensed image databases.

Entry point is `main.py`, which dispatches from a menu (`desktop_gui.FancyGUI`) into tracking, training, synthetic data generation, or analysis flows. The app expects to be run from its own repo root; `main.py` prepends `src/` to `sys.path`, so **every module under `src/` imports as a top-level name** (e.g. `from tracker import CellTracker`, not `src.tracker`). Preserve this when editing imports or tests.

## Commands

All Python work assumes the project venv:

```bash
source cell_track_venv/bin/activate
```

The venv is created by `./setup.sh` (minimal) or `./install_linux.sh` (with desktop integration). On Windows: `Setup_Windows.bat`.

### Running the app

```bash
./launch.sh                        # Linux
python3 main.py                    # any platform, from repo root
```

### Tests

The venv does not include `pytest`. Use the custom runner:

```bash
python tests/run_all.py            # runs all test_*.py files, no network
```

The runner discovers every `test_*` function across `tests/`, injects fixtures that `conftest.py` would normally supply (synthetic frames, single-cell/debris images), and prints `ok/FAIL/ERROR` per test with a final pass/fail count.

To run a single test module, use `unittest`-style dispatch or edit `run_all.py` to filter, since `pytest` is unavailable:

```bash
PYTHONPATH=src python -c "import tests.test_tracker_evaluation as m; m.test_evaluation_clean_scene()"
```

`tests/test_tracker_evaluation.py::test_print_evaluation_table` prints a scene-by-scene accuracy table (coverage / purity / id-switches / loc-err / ghost-rate) across seven difficulty levels — use this as the primary dashboard when tuning tracker or detector parameters.

### Building Windows installer

```bat
build_windows_exe.bat
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\installer.iss
```

Output: `installer_output\CytoTrackAI-1.0-setup.exe`.

## Architecture

### Pipeline

1. `CellDetector` (`src/detector.py`) — fuses **adaptive threshold + Otsu + distance-transform watershed + (optional) blob + (optional) Hough-circle** strategies, then runs a custom NMS that rejects on IoU *and* on center-distance vs `min(w,h)/2`. `calibrate()` sets `min_area` / `max_area` / `expected_max_diameter` from the first frame; `detect()` returns `Detection` objects with `bbox`, `area`, `confidence`, `center_x/y`.

2. `CellTracker` (`src/tracker.py`) — SORT-style. Each track owns a 6D constant-velocity `KalmanBox` (`[cx, cy, w, h, vx, vy]`) plus an **EMA-smoothed appearance thumbnail** (12×12 zero-mean unit-variance grayscale with 50% contextual padding). The cost matrix combines four cues: `(1-IoU)·1.0 + (dist/max_distance)·0.5 + appearance·0.7 + log_size_ratio·0.5`, cutoff 2.5. Hungarian is solved with `scipy.optimize.linear_sum_assignment` (greedy fallback if scipy missing). After the solve, three guarded fallbacks handle unmatched tracks in this order: **(1) last-chance match** (widened 1.5× radius, appearance ≤ 0.35) → **(2) merge-share** (track coasts inside an oversized blob already claimed by another track *without* incrementing `missed_frames`) → **(3) pure coast**. Unmatched detections first try **inactive-track revival** before spawning a new ID. Near-neighbour swap guard refuses a Hungarian match if a neighbour's appearance fits the same detection by ≥ 0.15 better.

3. `LostCellRecovery` (`src/lost_cell_recovery.py`) — NCC template matching (or optional visual-LLM check) resurrecting tracks that drifted off detection.

4. `MigrationAnalyzer` + `TrajectoryVisualizer` — per-track metrics (distance / displacement / CDE / persistence / MSD), statistical comparison across cell types (t-test, Mann-Whitney, Cohen's d), and matplotlib/plotly plots. Writes the `tracking_YYYYMMDD_HHMMSS/` output bundle documented in the README.

### Accuracy vs latency separation

`HardwareProfile` (`src/hardware_profile.py`) is **allowed to tune only throughput knobs** (batch size, worker count, enabling optional detector strategies). It must never touch accuracy-affecting parameters (areas, thresholds, cost weights, classifier weights). This invariant is enforced by a regression test — respect it when adding new config.

### Soft dependencies

`DSPy`, `vLLM`, and `anthropic` are all optional. `debris_reasoner.py` and `ai_assistant.py` fall back to heuristics if the heavy dep is missing. Never make these hard requirements.

### Phenotype image DB

`src/cell_image_library.py` enforces a **permissive-license allow-list** (`CC-0`, `CC-BY-*`, `CC-BY-SA-*`, `MIT`, `Apache-2.0`, `BSD-*`, public-domain). `register_dataset()` raises `ValueError` on anything else. Every download writes a `manifest.json`; multi-class builds write a top-level `LICENSES.json`. Do not bypass this gate.

## Tracker tuning guardrails

When touching `tracker.py` — especially `calibrate()`, `_cost_matrix`, the near-neighbour guard, or the fallback order:

- Always run `tests/run_all.py` after each change. The honest accuracy bars live in `tests/test_tracker_evaluation.py` and `tests/test_overlap_tracking.py`.
- Trade-offs are real: changes that help the 200/400-cell stress cases often regress the clean/heavy scenes. Check the full evaluation table, not just one scene.
- `calibrate()` sets `max_distance = max(30.0, median_diameter * 2.5)`. Tightening the multiplier below 2.5× has historically regressed clean/long/dense — see `feedback_tracker_max_distance.md` in auto-memory.
- Overlap safety is a correctness requirement, not a tuning preference. The synthetic generator has an `overlap_density` knob used by `test_overlap_tracking.py` specifically to catch ID swaps when bounding boxes merge.

## Qt plugin path

`main.py` strips `QT_PLUGIN_PATH` / `QT_QPA_PLATFORM_PLUGIN_PATH` and points them at the system Qt5 plugins before any `cv2` or `PyQt5` import. `opencv-python` ships its own Qt5 plugins that conflict on Linux. Preserve this block if reorganising startup.
