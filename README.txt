================================================================
  CytoTrack AI v1.0 — Cell Migration Tracking & Analysis (AI-assisted)
================================================================

FEATURES
--------
1. New identity: CytoTrack AI (unique, not already taken by existing
   cell-tracking tools like TrackMate, CellTracker, usiigaci, etc.).

2. Modern tracker: SORT-style pipeline — per-track constant-velocity
   Kalman filter + Hungarian assignment (scipy.optimize.linear_sum_
   assignment) against fresh detections every frame. Graceful
   fall-back to a greedy assigner when scipy is absent.

3. Smarter detector: fuses adaptive thresholding, Otsu, distance-
   transform watershed, OpenCV SimpleBlobDetector, and Hough circles.
   IoU-based non-max suppression. Auto polarity detection (bright
   cells on dark background vs. dark on bright).

4. Visual-LLM helper (src/ai_assistant.py): can verify cell crops
   with a vision-language model.
     * vLLM backend (e.g. Qwen2-VL-7B-Instruct) when `vllm` is
       installed and a local GPU is available.
     * Anthropic Claude vision as a cloud fall-back when the
       ANTHROPIC_API_KEY env var is set.
     * Classical heuristic when neither is available, so the
       pipeline always works.

5. DSPy reasoning for debris judgement (src/debris_reasoner.py):
     * Uses `dspy.ChainOfThought` when available for structured
       "is this debris or a real cell?" reasoning with explanation.
     * Can switch to `dspy.ReAct` with tools (cell-shape tool,
       intensity-profile tool, neighbor-context tool) for deeper
       analysis.
     * Deterministic heuristic fallback when DSPy absent.

6. Comprehensive unit tests (tests/) using synthetic data —
   cover detector, tracker, analyzer, synthetic generator, AI
   helpers and reasoner.


WORKFLOW
--------
Launch:
    ./launch.sh       (Linux/Mac)
    LaunchCellTracker.bat (Windows)

Menu:
  * Track Cells          — full migration-tracking pipeline
  * Train AI Classifier  — ViT / CNN transfer learning
  * Analyze Existing Data
  * Generate Test Data

Classification modes:
  * Fast Mode           — one type for all cells
  * Manual              — classify each cell interactively
  * Auto-Classify       — AI classifies each cell
  * No Classification


OUTPUT FILES
------------
tracking_video.avi
plot_trajectories.png            plot_trajectories_interactive.html
plot_velocity_histogram.png      plot_displacement_distance.png
plot_directionality.png          plot_msd.png
plot_rose.png
migration_detailed.csv           migration_summary.csv
statistical_comparison.csv       cell_type_summary.csv
settings_used.txt


METRICS
-------
Per track:
  Total_Distance_um, Displacement_um, CDE (confinement ratio),
  Avg_Velocity_um_min, Max_Velocity_um_min, Persistence, MSD(τ).

Between types:
  T-test, Mann-Whitney U, Cohen's d, significance markers.


INSTALL / RUN
-------------
Linux/Mac:
    chmod +x setup.sh launch.sh
    ./setup.sh
    ./launch.sh

Windows:
    Double-click Setup_Windows.bat
    Double-click LaunchCellTracker.bat

Run unit tests:
    python3 -m pytest tests/ -v
    or
    python3 tests/run_all.py

Optional AI-helper dependencies (everything stays functional
without them — the helpers use clean fallbacks):
    pip install dspy-ai
    pip install vllm                                 # GPU required
    pip install anthropic                            # cloud fallback
    export ANTHROPIC_API_KEY=...                     # for Claude vision
