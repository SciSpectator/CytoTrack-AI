#!/usr/bin/env python3
"""Train and evaluate a frame-to-frame cell association model.

This trains the tracking part of CytoTrack AI: given an object in frame t and
candidate objects in frame t+1, predict which candidate is the same cell. The
training labels come from public CTC TRA masks.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used",
    category=UserWarning,
)

from tracking_linker import (FEATURE_NAMES, appearance_similarity,
                             build_link_features, bbox_iou, crop_appearance)


IMAGE_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def frame_number(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if not match:
        raise ValueError(path)
    return int(match.group(1))


def image_path(root: Path, dataset: str, sequence: str, frame: int) -> Path:
    return root / dataset / sequence / f"t{frame:03d}.tif"


def mask_paths(root: Path, dataset: str, sequence: str) -> List[Path]:
    tra = root / dataset / f"{sequence}_GT" / "TRA"
    return sorted([p for p in tra.iterdir() if p.suffix.lower() in IMAGE_EXTS],
                  key=frame_number)


def normalize_u8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype == np.uint8:
        return image
    arr = image.astype(np.float32)
    lo, hi = np.percentile(arr, [1, 99.5])
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def object_geometry(mask: np.ndarray, label: int) -> dict | None:
    binary = np.uint8(mask == label) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area <= 0:
        return None
    x, y, w, h = cv2.boundingRect(contour)
    moments = cv2.moments(contour)
    if moments["m00"]:
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
    else:
        cx = float(x + w / 2.0)
        cy = float(y + h / 2.0)
    return {
        "label": int(label),
        "bbox": (float(x), float(y), float(w), float(h)),
        "center": (cx, cy),
        "area": area,
    }


def frame_objects(mask: np.ndarray) -> Dict[int, dict]:
    out = {}
    for label in np.unique(mask):
        label = int(label)
        if label <= 0:
            continue
        geom = object_geometry(mask, label)
        if geom is not None:
            out[label] = geom
    return out


def load_sequence(root: Path, dataset: str, sequence: str) -> List[dict]:
    frames = []
    for mask_path in mask_paths(root, dataset, sequence):
        frame = frame_number(mask_path)
        image = cv2.imread(str(image_path(root, dataset, sequence, frame)),
                           cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None:
            continue
        image = normalize_u8(image)
        objects = frame_objects(mask)
        appearances = {
            label: crop_appearance(image, geom["bbox"])
            for label, geom in objects.items()
        }
        frames.append({
            "dataset": dataset,
            "sequence": sequence,
            "frame": frame,
            "image": image,
            "objects": objects,
            "appearances": appearances,
        })
    return frames


def pair_features(source: dict, target: dict, source_app, target_app) -> np.ndarray:
    return build_link_features(
        source["bbox"],
        target["bbox"],
        source_area=source["area"],
        target_area=target["area"],
        appearance_similarity=appearance_similarity(source_app, target_app),
    )


def candidate_pairs(frames: Sequence[dict], max_negatives_per_object: int = 8) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    xs: List[np.ndarray] = []
    ys: List[int] = []
    rows: List[dict] = []
    for a, b in zip(frames[:-1], frames[1:]):
        for label, source in a["objects"].items():
            candidates = []
            for target_label, target in b["objects"].items():
                feat = pair_features(
                    source,
                    target,
                    a["appearances"].get(label),
                    b["appearances"].get(target_label),
                )
                distance = float(feat[FEATURE_NAMES.index("distance")])
                is_positive = int(label == target_label)
                candidates.append((distance, target_label, target, feat, is_positive))
            candidates.sort(key=lambda item: (0 if item[4] else 1, item[0]))
            positives = [c for c in candidates if c[4]]
            negatives = [c for c in candidates if not c[4]][:max_negatives_per_object]
            for _distance, target_label, _target, feat, is_positive in positives + negatives:
                xs.append(feat)
                ys.append(is_positive)
                rows.append({
                    "dataset": a["dataset"],
                    "sequence": a["sequence"],
                    "frame": a["frame"],
                    "source_label": label,
                    "target_label": target_label,
                    "is_positive": is_positive,
                    **{name: float(value) for name, value in zip(FEATURE_NAMES, feat)},
                })
    if not xs:
        raise RuntimeError("no training pairs generated")
    return np.vstack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64), pd.DataFrame(rows)


def baseline_score(features: np.ndarray) -> float:
    f = {name: float(value) for name, value in zip(FEATURE_NAMES, features)}
    return (
        -f["normalized_distance"]
        + 2.5 * f["bbox_iou"]
        - 0.8 * f["abs_log_area_ratio"]
        + 0.4 * f["appearance_similarity"]
    )


def evaluate_linker(frames: Sequence[dict], model=None) -> dict:
    persistent = 0
    baseline_correct = 0
    model_correct = 0
    baseline_errors = []
    model_errors = []
    rows = []
    for a, b in zip(frames[:-1], frames[1:]):
        if a["dataset"] != b["dataset"] or a["sequence"] != b["sequence"]:
            continue
        next_labels = set(b["objects"])
        candidates_by_label: Dict[int, List[dict]] = {}
        feature_rows = []
        candidate_refs = []
        for label, source in a["objects"].items():
            if label not in next_labels:
                continue
            for target_label, target in b["objects"].items():
                feat = pair_features(
                    source,
                    target,
                    a["appearances"].get(label),
                    b["appearances"].get(target_label),
                )
                feature_rows.append(feat)
                candidate_refs.append((label, target_label, baseline_score(feat)))
        if not feature_rows:
            continue
        if model is not None:
            trained_scores = model.predict_proba(np.vstack(feature_rows))[:, 1]
        else:
            trained_scores = np.asarray([ref[2] for ref in candidate_refs],
                                        dtype=np.float32)
        for (label, target_label, base), trained in zip(candidate_refs, trained_scores):
            candidates_by_label.setdefault(label, []).append({
                "target_label": target_label,
                "baseline_score": float(base),
                "trained_score": float(trained),
            })
        for label, candidates in candidates_by_label.items():
            persistent += 1
            base_pick = max(candidates, key=lambda item: item["baseline_score"])["target_label"]
            model_pick = max(candidates, key=lambda item: item["trained_score"])["target_label"]
            if base_pick == label:
                baseline_correct += 1
            else:
                baseline_errors.append((a["frame"], label, base_pick))
            if model_pick == label:
                model_correct += 1
            else:
                model_errors.append((a["frame"], label, model_pick))
            rows.append({
                "dataset": a["dataset"],
                "sequence": a["sequence"],
                "frame": a["frame"],
                "source_label": label,
                "baseline_pick": base_pick,
                "model_pick": model_pick,
                "correct_target": label,
                "baseline_correct": base_pick == label,
                "model_correct": model_pick == label,
            })
    return {
        "persistent_links": int(persistent),
        "baseline_correct": int(baseline_correct),
        "model_correct": int(model_correct),
        "baseline_accuracy": float(baseline_correct / max(1, persistent)),
        "model_accuracy": float(model_correct / max(1, persistent)),
        "baseline_identity_errors": int(len(baseline_errors)),
        "model_identity_errors": int(len(model_errors)),
        "error_delta": int(len(baseline_errors) - len(model_errors)),
        "rows": pd.DataFrame(rows),
    }


def discover(root: Path, datasets: Iterable[str], sequences: Iterable[str]) -> List[Tuple[str, str]]:
    pairs = []
    for dataset in datasets:
        for sequence in sequences:
            if (root / dataset / sequence).exists() and (root / dataset / f"{sequence}_GT" / "TRA").exists():
                pairs.append((dataset, sequence))
    return pairs


def write_dashboard(output_dir: Path, train_metrics: dict, eval_metrics: dict) -> None:
    improvement = eval_metrics["model_accuracy"] - eval_metrics["baseline_accuracy"]
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Tracking Linker Training</title>
<style>
body {{ font-family: Inter, Arial, sans-serif; margin: 24px; color: #1f2933; }}
table {{ border-collapse: collapse; min-width: 760px; }}
td, th {{ padding: 8px 10px; border-bottom: 1px solid #d7dde2; text-align: right; }}
td:first-child, th:first-child {{ text-align: left; }}
.ok {{ color: #137333; font-weight: 700; }}
.bad {{ color: #b3261e; font-weight: 700; }}
</style></head><body>
<h1>Tracking Linker Training</h1>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Training pairs</td><td>{train_metrics['training_pairs']}</td></tr>
<tr><td>Positive pairs</td><td>{train_metrics['positive_pairs']}</td></tr>
<tr><td>Validation ROC-AUC</td><td>{train_metrics['roc_auc']:.4f}</td></tr>
<tr><td>Validation average precision</td><td>{train_metrics['average_precision']:.4f}</td></tr>
<tr><td>Baseline link accuracy</td><td>{eval_metrics['baseline_accuracy']:.4f}</td></tr>
<tr><td>Trained link accuracy</td><td>{eval_metrics['model_accuracy']:.4f}</td></tr>
<tr><td>Accuracy delta</td><td class="{'ok' if improvement >= 0 else 'bad'}">{improvement:+.4f}</td></tr>
<tr><td>Baseline identity errors</td><td>{eval_metrics['baseline_identity_errors']}</td></tr>
<tr><td>Model identity errors</td><td>{eval_metrics['model_identity_errors']}</td></tr>
</table>
</body></html>"""
    (output_dir / "dashboard.html").write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT / "real_cell_movies"))
    parser.add_argument("--datasets", nargs="+",
                        default=["DIC-C2DH-HeLa", "Fluo-C2DL-Huh7"])
    parser.add_argument("--train-sequences", nargs="+", default=["01"])
    parser.add_argument("--test-sequences", nargs="+", default=["02"])
    parser.add_argument("--output-dir", default=str(ROOT / "RESULT" / "tracking_linker_training"))
    parser.add_argument("--model-dir", default=str(ROOT / "model_cache" / "tracking_linker"))
    parser.add_argument("--trees", type=int, default=240)
    parser.add_argument("--fail-on-regression", action="store_true",
                        help="Return non-zero when the trained model is worse than the baseline.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    if not root.is_absolute():
        root = ROOT / root
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_frames = []
    test_frames = []
    for dataset, sequence in discover(root, args.datasets, args.train_sequences):
        train_frames.extend(load_sequence(root, dataset, sequence))
    for dataset, sequence in discover(root, args.datasets, args.test_sequences):
        test_frames.extend(load_sequence(root, dataset, sequence))
    if not train_frames or not test_frames:
        raise RuntimeError("missing train/test tracking sequences")

    x_train, y_train, pairs_df = candidate_pairs(train_frames)
    x_test, y_test, test_pairs_df = candidate_pairs(test_frames)

    model = RandomForestClassifier(
        n_estimators=args.trees,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=17,
        n_jobs=1,
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    train_metrics = {
        "training_pairs": int(len(y_train)),
        "positive_pairs": int(y_train.sum()),
        "test_pairs": int(len(y_test)),
        "test_positive_pairs": int(y_test.sum()),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "feature_names": FEATURE_NAMES,
        "train_sequences": args.train_sequences,
        "test_sequences": args.test_sequences,
        "datasets": args.datasets,
    }
    eval_metrics = evaluate_linker(test_frames, model)
    eval_rows = eval_metrics.pop("rows")
    report = {
        "training": train_metrics,
        "evaluation": eval_metrics,
        "model_path": str((model_dir / "tracking_linker.joblib").relative_to(ROOT)),
        "purpose": "frame-to-frame tracking association model",
        "source_data": "public Cell Tracking Challenge training movies with TRA masks",
        "deployment_decision": (
            "deploy_trained_model"
            if eval_metrics["model_accuracy"] > eval_metrics["baseline_accuracy"]
            else "keep_existing_tracker_no_accuracy_gain"
        ),
    }

    joblib.dump({
        "model": model,
        "feature_names": FEATURE_NAMES,
        "report": report,
    }, model_dir / "tracking_linker.joblib")
    (model_dir / "tracking_linker_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    pairs_df.to_csv(output_dir / "train_pairs.csv", index=False)
    test_pairs_df.to_csv(output_dir / "test_pairs.csv", index=False)
    eval_rows.to_csv(output_dir / "link_predictions_test.csv", index=False)
    (output_dir / "tracking_linker_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    write_dashboard(output_dir, train_metrics, eval_metrics)

    print(json.dumps(report, indent=2))
    if args.fail_on_regression and eval_metrics["model_accuracy"] < eval_metrics["baseline_accuracy"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
