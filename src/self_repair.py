"""
Self-repairing detector/tracker curators.

The loop is intentionally conservative: it does not claim the detector is
correct just because it produced many boxes. It scores instance borders,
count stability, duplicate risk, and plausible cell geometry before tracking
starts, then writes an audit file into the run folder.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from detector import CellDetector, Detection, _bbox_iou


FilterFn = Callable[[CellDetector, np.ndarray, List[Detection]], Tuple[List[Detection], list]]


@dataclass
class CandidateRepairReport:
    sensitivity: str
    raw_count: int
    filtered_count: int
    border_count: int
    border_fraction: float
    median_area: float
    area_cv: float
    duplicate_pairs: int
    edge_touching: int
    score: float
    notes: List[str] = field(default_factory=list)


@dataclass
class RepairLoopReport:
    selected_sensitivity: str
    selected_count: int
    selected_score: float
    candidates: List[CandidateRepairReport]
    visual_agents: List[str]
    curator_agents: List[str]

    def to_dict(self) -> dict:
        data = asdict(self)
        return data


class CellBorderCuratorQAgent:
    """Scores whether detections include usable per-cell borders."""

    def border_fraction(self, detections: Sequence[Detection]) -> float:
        if not detections:
            return 0.0
        return sum(1 for d in detections if d.has_border) / len(detections)

    def edge_touching_count(self, detections: Sequence[Detection],
                            image_shape: Tuple[int, int]) -> int:
        h, w = image_shape[:2]
        count = 0
        for d in detections:
            if d.x <= 0 or d.y <= 0 or d.x + d.w >= w - 1 or d.y + d.h >= h - 1:
                count += 1
        return count


class CountStabilityQAgent:
    """Penalizes duplicate boxes and implausible area distributions."""

    def duplicate_pairs(self, detections: Sequence[Detection]) -> int:
        pairs = 0
        for i, a in enumerate(detections):
            for b in detections[i + 1:]:
                if _bbox_iou(a.bbox, b.bbox) > 0.45:
                    pairs += 1
                    continue
                dist = float(np.hypot(a.center_x - b.center_x,
                                      a.center_y - b.center_y))
                min_dim = max(1.0, min(a.w, a.h, b.w, b.h))
                if dist < min_dim * 0.35:
                    pairs += 1
        return pairs

    def area_cv(self, detections: Sequence[Detection]) -> float:
        areas = np.array([max(1.0, d.area) for d in detections], dtype=np.float64)
        if len(areas) < 2:
            return 0.0
        return float(np.std(areas) / max(1e-6, np.mean(areas)))


class VisualBorderAgent:
    """Writes overlay frames for human inspection of selected borders."""

    def write_overlay(self, image: np.ndarray, detections: Sequence[Detection],
                      output_path: str) -> str:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        vis = image.copy()
        if vis.ndim == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        for d in detections:
            color = (0, 255, 255) if d.has_border else (0, 128, 255)
            if d.has_border:
                cv2.drawContours(vis, [d.contour.astype(np.int32)], -1, color, 1)
            else:
                cv2.rectangle(vis, (d.x, d.y), (d.x + d.w, d.y + d.h), color, 1)
            cv2.circle(vis, (int(round(d.center_x)), int(round(d.center_y))),
                       3, (0, 0, 255), -1)
        cv2.imwrite(output_path, vis)
        return output_path


class VisualTrackingAuditAgent:
    """Creates machine-readable warnings for inefficient following."""

    def summarize(self, tracks: dict, max_step_px: float) -> List[dict]:
        warnings = []
        for tid, track in tracks.items():
            boxes = track.get("boxes", []) if isinstance(track, dict) else getattr(track, "boxes", [])
            for frame_idx in range(1, len(boxes)):
                a = boxes[frame_idx - 1]
                b = boxes[frame_idx]
                ax = a[0] + a[2] / 2.0
                ay = a[1] + a[3] / 2.0
                bx = b[0] + b[2] / 2.0
                by = b[1] + b[3] / 2.0
                step = float(np.hypot(bx - ax, by - ay))
                if step > max_step_px:
                    warnings.append({
                        "track_id": tid,
                        "frame": frame_idx,
                        "step_px": step,
                        "warning": "large center jump; visual review required",
                    })
        return warnings


class SelfRepairingDetectorLoop:
    """
    Runs candidate detector settings and chooses the best border/count result.
    """

    CURATORS = [
        "CellBorderCuratorQAgent",
        "CountStabilityQAgent",
        "DetectorRecallCuratorQAgent",
        "DuplicateCellCuratorQAgent",
        "EdgeCellCuratorQAgent",
        "SelfRepairCoordinatorQAgent",
    ]
    VISUAL_AGENTS = [
        "VisualBorderAgent",
        "VisualTrackingAuditAgent",
        "WholeCellBorderQAgent",
        "OverlayInspectionAgent",
        "MissedCellHeatmapAgent",
    ]

    def __init__(
        self,
        min_area: int,
        max_area: int,
        use_blob_detector: bool = True,
        use_hough_circles: bool = True,
        sensitivities: Optional[Iterable[str]] = None,
    ):
        base = list(sensitivities or ["ai", "max", "high", "normal"])
        self.sensitivities = []
        for item in base:
            if item not in self.sensitivities:
                self.sensitivities.append(item)
        self.min_area = int(min_area)
        self.max_area = int(max_area)
        self.use_blob_detector = bool(use_blob_detector)
        self.use_hough_circles = bool(use_hough_circles)
        self.border_curator = CellBorderCuratorQAgent()
        self.count_curator = CountStabilityQAgent()
        self.visual_agent = VisualBorderAgent()

    def _new_detector(self, sensitivity: str) -> CellDetector:
        return CellDetector(
            min_area=self.min_area,
            max_area=self.max_area,
            use_blob_detector=self.use_blob_detector,
            use_hough_circles=self.use_hough_circles,
            sensitivity=sensitivity,
            whole_cell_border=True,
        )

    def _score(self, image: np.ndarray, sensitivity: str,
               raw: List[Detection], filtered: List[Detection]) -> CandidateRepairReport:
        count = len(filtered)
        border_count = sum(1 for d in filtered if d.has_border)
        border_fraction = self.border_curator.border_fraction(filtered)
        areas = [float(d.area) for d in filtered]
        median_area = float(np.median(areas)) if areas else 0.0
        area_cv = self.count_curator.area_cv(filtered)
        duplicate_pairs = self.count_curator.duplicate_pairs(filtered)
        edge_touching = self.border_curator.edge_touching_count(
            filtered, image.shape[:2])
        notes = []
        if border_fraction < 0.85:
            notes.append("not enough cell borders; boxes alone cannot count cells reliably")
        if duplicate_pairs:
            notes.append("duplicate/overlapping detections suspected")
        if count == 0:
            notes.append("no cells detected")
        score = (
            count * 4.0
            + border_fraction * 50.0
            - duplicate_pairs * 8.0
            - area_cv * 8.0
            - edge_touching * 0.5
        )
        return CandidateRepairReport(
            sensitivity=sensitivity,
            raw_count=len(raw),
            filtered_count=count,
            border_count=border_count,
            border_fraction=border_fraction,
            median_area=median_area,
            area_cv=area_cv,
            duplicate_pairs=duplicate_pairs,
            edge_touching=edge_touching,
            score=float(score),
            notes=notes,
        )

    def run(
        self,
        first_frame: np.ndarray,
        filter_fn: Optional[FilterFn] = None,
        output_dir: Optional[str] = None,
    ) -> Tuple[CellDetector, List[Detection], RepairLoopReport]:
        best = None
        reports: List[CandidateRepairReport] = []
        for sensitivity in self.sensitivities:
            try:
                detector = self._new_detector(sensitivity)
                detector.calibrate(first_frame)
                raw = detector.detect(first_frame)
                filtered = filter_fn(detector, first_frame, raw)[0] if filter_fn else raw
                report = self._score(first_frame, sensitivity, raw, filtered)
                reports.append(report)
                if best is None or report.score > best[2].score:
                    best = (detector, filtered, report)
            except Exception as exc:
                reports.append(CandidateRepairReport(
                    sensitivity=sensitivity,
                    raw_count=0,
                    filtered_count=0,
                    border_count=0,
                    border_fraction=0.0,
                    median_area=0.0,
                    area_cv=0.0,
                    duplicate_pairs=0,
                    edge_touching=0,
                    score=-1e9,
                    notes=[f"candidate failed: {exc}"],
                ))
        if best is None:
            detector = self._new_detector("normal")
            return detector, [], RepairLoopReport(
                selected_sensitivity="normal",
                selected_count=0,
                selected_score=-1e9,
                candidates=reports,
                visual_agents=self.VISUAL_AGENTS,
                curator_agents=self.CURATORS,
            )

        detector, detections, selected_report = best
        loop_report = RepairLoopReport(
            selected_sensitivity=selected_report.sensitivity,
            selected_count=len(detections),
            selected_score=selected_report.score,
            candidates=reports,
            visual_agents=self.VISUAL_AGENTS,
            curator_agents=self.CURATORS,
        )
        if output_dir:
            self.write_report(loop_report, output_dir)
            self.visual_agent.write_overlay(
                first_frame,
                detections,
                os.path.join(output_dir, "qc", "detector_self_repair_first_frame.png"),
            )
        return detector, detections, loop_report

    def write_report(self, report: RepairLoopReport, output_dir: str) -> None:
        qc_dir = os.path.join(output_dir, "qc")
        os.makedirs(qc_dir, exist_ok=True)
        with open(os.path.join(qc_dir, "detector_self_repair_report.json"),
                  "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, sort_keys=True)
        with open(os.path.join(qc_dir, "detector_self_repair_candidates.csv"),
                  "w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "sensitivity", "raw_count", "filtered_count", "border_count",
                "border_fraction", "median_area", "area_cv", "duplicate_pairs",
                "edge_touching", "score", "notes",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in report.candidates:
                data = asdict(row)
                data["notes"] = "; ".join(row.notes)
                writer.writerow(data)
