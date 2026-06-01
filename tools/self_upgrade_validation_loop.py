#!/usr/bin/env python3
"""
Quality gate loop for CytoTrack AI.

This is intentionally broader than unit tests. It runs code tests, short and
long real-movie checks, frame-folder result generation, morphology training,
dashboard/video export checks, and machine-readable QC audits. Outputs stay
under RESULT/ and are ignored by git.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

import cv2
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "RESULT"


class GateError(RuntimeError):
    pass


def run_command(name: str, cmd: List[str], cwd: Path = ROOT) -> dict:
    started = time.time()
    print(f"[gate] {name}: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = time.time() - started
    print(proc.stdout, end="", flush=True)
    return {
        "name": name,
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 3),
        "stdout_tail": proc.stdout[-8000:],
    }


def assert_video_readable(path: Path, min_frames: int = 1) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise GateError(f"missing or empty video: {path}")
    cap = cv2.VideoCapture(str(path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise GateError(f"video is not readable: {path}")
    if count and count < min_frames:
        raise GateError(f"video has too few frames: {path} ({count})")


def validate_stress(out_dir: Path, expected_clips: int) -> dict:
    summary_path = out_dir / "stress_30_movies_summary.json"
    report_path = out_dir / "stress_30_movies_report.csv"
    if not summary_path.exists() or not report_path.exists():
        raise GateError(f"stress report missing in {out_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report = pd.read_csv(report_path)
    if int(summary["completed_clips"]) != expected_clips:
        raise GateError(f"stress completed {summary['completed_clips']}, expected {expected_clips}")
    if int(summary["failed_clips"]) != 0:
        raise GateError(f"stress failed clips: {summary['failed_clips']}")
    if "median_border_extent" not in report.columns:
        raise GateError("stress report missing whole-cell border extent")
    if float(report["median_border_extent"].min()) < 0.55:
        raise GateError("whole-cell border extent gate failed")
    if report["passed"].astype(bool).sum() != expected_clips:
        raise GateError("not all stress rows passed")
    for rel in report["output_dir"].head(4):
        assert_video_readable(ROOT / rel / "tracking_video.mp4", min_frames=2)
    return {
        "completed": int(summary["completed_clips"]),
        "failed": int(summary["failed_clips"]),
        "min_border_extent": float(report["median_border_extent"].min()),
        "mean_detections_per_frame": float(summary["mean_detections_per_frame"]),
    }


def validate_long(out_dir: Path) -> dict:
    index = out_dir / "LONG_VIDEO_VALIDATION_INDEX.csv"
    if not index.exists():
        raise GateError(f"long validation index missing: {index}")
    df = pd.read_csv(index)
    if df.empty:
        raise GateError("long validation produced no rows")
    if not df["coverage_passed"].astype(bool).all():
        raise GateError("long validation coverage failed")
    if not df["identity_qc_passed"].astype(bool).all():
        raise GateError("long validation identity QC failed")
    for video in df["video"]:
        assert_video_readable(out_dir / video, min_frames=30)
    return {
        "movies": int(len(df)),
        "coverage_passed": bool(df["coverage_passed"].astype(bool).all()),
        "identity_qc_passed": bool(df["identity_qc_passed"].astype(bool).all()),
    }


def validate_small_gt(out_dir: Path) -> dict:
    folders = [p for p in out_dir.iterdir() if p.is_dir() and (p / "manifest.json").exists()]
    if not folders:
        raise GateError("small GT result folders missing")
    for folder in folders:
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if not manifest.get("quality_control", {}).get("passed", False):
            raise GateError(f"identity QC failed in {folder}")
        if manifest.get("tracked_point") and "centroid" not in manifest["tracked_point"]:
            raise GateError(f"tracked point is not centroid based in {folder}")
        if int(manifest.get("max_cells", 0)) > 0:
            selected = len(manifest.get("selected_track_ids", []))
            if selected > int(manifest["max_cells"]):
                raise GateError(f"selected too many cells in {folder}: {selected}")
        assert_video_readable(folder / "tracking_video.mp4", min_frames=2)
        for required in ["migration_summary.csv", "migration_detailed.csv", "gt_tracks.csv"]:
            if not (folder / required).exists():
                raise GateError(f"{required} missing in {folder}")
        for plot in manifest.get("files", {}).get("plots", []):
            if not (folder / plot).exists():
                raise GateError(f"plot missing in {folder}: {plot}")
    return {"folders": len(folders)}


def validate_training_compare(out_dir: Path) -> dict:
    dashboard = out_dir / "cell_line_migration_comparison" / "dashboard.html"
    model = ROOT / "model_cache" / "cell_line_morphology" / "morphology_model.json"
    summary = out_dir / "cell_line_migration_comparison" / "cell_line_migration_summary.csv"
    if not dashboard.exists() or dashboard.stat().st_size == 0:
        raise GateError("training comparison dashboard missing")
    if not model.exists():
        raise GateError("morphology model missing")
    model_data = json.loads(model.read_text(encoding="utf-8"))
    if float(model_data.get("training_accuracy", 0.0)) < 0.80:
        raise GateError(f"low morphology training accuracy: {model_data.get('training_accuracy')}")
    df = pd.read_csv(summary)
    required = {"HeLa", "Huh7"}
    present = set(df["Cell_Line"].dropna().unique())
    if not required.issubset(present):
        raise GateError(f"training comparison missing required groups: {required - present}")
    return {
        "training_accuracy": float(model_data["training_accuracy"]),
        "classes": model_data["classes"],
        "tracks": int(len(df)),
    }


def write_report(out_dir: Path, iterations: List[dict], final_status: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": final_status,
        "iterations": iterations,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "self_upgrade_validation_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    rows = []
    for iteration in iterations:
        for gate in iteration["gates"]:
            rows.append({
                "iteration": iteration["iteration"],
                "gate": gate["name"],
                "passed": gate["passed"],
                "elapsed_sec": gate.get("elapsed_sec", 0),
                "details": json.dumps(gate.get("details", {}), sort_keys=True),
            })
    pd.DataFrame(rows).to_csv(out_dir / "self_upgrade_validation_report.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--frames-long", type=int, default=180)
    parser.add_argument("--stress-window-short", type=int, default=8)
    parser.add_argument("--stress-window-long", type=int, default=20)
    parser.add_argument("--output-dir", default=str(RESULT_ROOT / "self_upgrade_validation"))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    iterations: List[dict] = []
    validators: Dict[str, Callable[[], dict]] = {}

    for iteration in range(1, args.iterations + 1):
        iteration_record = {"iteration": iteration, "gates": []}
        iterations.append(iteration_record)

        gates = [
            (
                "pytest_full",
                [sys.executable, "-m", "pytest", "-q"],
                lambda: {"tests": "pytest full suite completed"},
            ),
            (
                "stress_30_short_movies",
                [
                    sys.executable, "tools/stress_test_30_movies.py",
                    "--clips", "30",
                    "--window", str(args.stress_window_short),
                    "--max-side", "384",
                    "--clean",
                    "--output", str(out_dir / "stress_30_short"),
                ],
                lambda: validate_stress(out_dir / "stress_30_short", 30),
            ),
            (
                "stress_30_longer_windows",
                [
                    sys.executable, "tools/stress_test_30_movies.py",
                    "--clips", "30",
                    "--window", str(args.stress_window_long),
                    "--max-side", "384",
                    "--clean",
                    "--output", str(out_dir / "stress_30_long_window"),
                ],
                lambda: validate_stress(out_dir / "stress_30_long_window", 30),
            ),
            (
                "long_movies",
                [
                    sys.executable, "tools/generate_long_video_validation.py",
                    "--frames", str(args.frames_long),
                    "--output-dir", str(out_dir / "long_movies"),
                ],
                lambda: validate_long(out_dir / "long_movies"),
            ),
            (
                "frame_folder_small_gt",
                [
                    sys.executable, "tools/generate_small_gt_results.py",
                    "--max-cells", "15",
                    "--output-dir", str(out_dir / "small_gt"),
                    "--color-by-cell-line",
                ],
                lambda: validate_small_gt(out_dir / "small_gt"),
            ),
            (
                "morphology_training_cell_lines",
                [
                    sys.executable, "tools/train_and_compare_two_cell_lines.py",
                    "--result-root", str(out_dir / "training_compare"),
                    "--max-samples-per-class", "120",
                ],
                lambda: validate_training_compare(out_dir / "training_compare"),
            ),
        ]

        all_passed = True
        for name, cmd, validator in gates:
            result = run_command(name, cmd)
            gate = {
                "name": name,
                "passed": result["returncode"] == 0,
                "elapsed_sec": result["elapsed_sec"],
                "details": {},
                "command_returncode": result["returncode"],
            }
            if result["returncode"] != 0:
                gate["details"] = {"stdout_tail": result["stdout_tail"]}
                all_passed = False
                iteration_record["gates"].append(gate)
                break
            try:
                gate["details"] = validator()
            except Exception as exc:
                gate["passed"] = False
                gate["details"] = {"error": str(exc)}
                all_passed = False
                iteration_record["gates"].append(gate)
                break
            iteration_record["gates"].append(gate)

        if all_passed:
            write_report(out_dir, iterations, "passed")
            print(f"[self-upgrade] all gates passed; report={out_dir}", flush=True)
            return 0

        if iteration == args.iterations:
            write_report(out_dir, iterations, "failed")
            print(f"[self-upgrade] failed; report={out_dir}", flush=True)
            return 2

        print("[self-upgrade] gate failed; rerunning after repairs/next iteration", flush=True)

    write_report(out_dir, iterations, "failed")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
