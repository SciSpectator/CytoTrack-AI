"""
Quality-first morphology pretraining agents.

These agents prepare phenotype training data before tracking starts. They are
deliberately licence-aware: public web research may identify candidate sources,
but only open-licensed datasets from the curated catalogue are downloaded
automatically.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from cell_image_library import (Dataset, build_phenotype_folders, catalogue,
                                is_licence_open, search)
from pipeline_architecture import (SEGMENTATION_BACKENDS, TRACKING_BACKENDS,
                                   BackendSpec, default_model_cache_root)


LIGHT_MICROSCOPY_TERMS = {
    "brightfield", "bright-field", "phase", "phase-contrast",
    "phase contrast", "dic", "differential interference contrast",
    "label-free", "light microscopy", "wound", "migration",
}

BLOCKED_RESEARCH_ONLY_SOURCES = [
    {
        "dataset_id": "LIVECell",
        "name": "LIVECell phase-contrast cell lines",
        "phenotype": "A172, BT474, BV2, Huh7, MCF7, SH-SY5Y, SK-OV-3, SkBr3",
        "licence": "CC-BY-NC-4.0",
        "homepage": "https://sartorius-research.github.io/LIVECell/",
        "approx_image_count": 3180,
        "keywords": [
            "phase contrast", "label-free", "light microscopy", "A172",
            "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SKOV3", "SkBr3",
        ],
    }
]


@dataclass
class ResearchCandidate:
    cell_line: str
    dataset_id: str
    name: str
    phenotype: str
    licence: str
    homepage: str
    approx_image_count: int
    condition_score: int
    downloadable: bool
    reason: str


@dataclass
class TrainingClassPlan:
    class_label: str
    cell_line: str
    min_required_images: int
    selected_dataset_id: Optional[str] = None
    selected_dataset_name: Optional[str] = None
    selected_dataset_homepage: Optional[str] = None
    selected_dataset_licence: Optional[str] = None
    approx_available_images: int = 0
    status: str = "missing"
    notes: List[str] = field(default_factory=list)
    candidates: List[ResearchCandidate] = field(default_factory=list)


@dataclass
class MorphologyTrainingPlan:
    generated_utc: str
    condition_query: str
    min_images_per_cell_line: int
    class_plans: List[TrainingClassPlan]
    architecture: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConditionProfile:
    query: str
    modality: str
    density: str
    notes: List[str] = field(default_factory=list)


@dataclass
class BackendSelection:
    selected: List[str]
    available: List[str]
    unavailable: List[str]
    notes: List[str] = field(default_factory=list)


def _norm(text: str) -> str:
    return (text or "").strip().lower().replace("_", "-")


def _condition_score(dataset: Dataset, condition_query: str) -> int:
    hay = " ".join([
        dataset.name,
        dataset.phenotype,
        dataset.description,
        " ".join(dataset.keywords),
    ]).lower()
    score = 0
    for term in LIGHT_MICROSCOPY_TERMS:
        if term in hay:
            score += 2
    for term in (condition_query or "").lower().replace(",", " ").split():
        if term and term in hay:
            score += 3
    return score


def _cell_line_score(dataset: Dataset, cell_line: str) -> int:
    needle = _norm(cell_line)
    hay = _norm(" ".join([
        dataset.id, dataset.name, dataset.phenotype,
        dataset.organism, dataset.description, " ".join(dataset.keywords),
    ]))
    aliases = {needle, needle.replace("-", ""), needle.replace(" ", "")}
    score = 0
    for alias in aliases:
        if alias and alias in hay.replace("-", "").replace(" ", ""):
            score += 8
    for token in needle.replace("-", " ").split():
        if token and token in hay:
            score += 2
    return score


class WebsiteResearchQAgent:
    """
    Looks up candidate public datasets for a requested cell line.

    The current implementation searches the curated open dataset catalogue.
    It also records why non-open or mismatched sources cannot be used
    automatically. This keeps the training pipeline deterministic and testable.
    """

    def research(
        self,
        cell_line: str,
        condition_query: str = "light microscope phase contrast brightfield DIC",
    ) -> List[ResearchCandidate]:
        hits: Dict[str, Dataset] = {}
        for query in {
            cell_line,
            f"{cell_line} {condition_query}",
            "phase contrast",
            "brightfield",
            "DIC",
            "migration",
            "nuclei",
        }:
            for dataset in search(query):
                hits[dataset.id] = dataset
        # Fall back to the whole catalogue so the report can say exactly why
        # nothing matched a rare cell line.
        for dataset in catalogue():
            hits.setdefault(dataset.id, dataset)

        candidates = []
        for dataset in hits.values():
            line_score = _cell_line_score(dataset, cell_line)
            condition_score = _condition_score(dataset, condition_query)
            licence_ok = is_licence_open(dataset.licence)
            downloadable = bool(licence_ok and line_score > 0 and
                                dataset.approx_image_count > 0)
            if not licence_ok:
                reason = f"blocked licence: {dataset.licence}"
            elif line_score <= 0:
                reason = "cell line not matched"
            elif condition_score <= 0:
                reason = "cell line matched, but microscopy condition is weak"
            else:
                reason = "usable open candidate"
            candidates.append(ResearchCandidate(
                cell_line=cell_line,
                dataset_id=dataset.id,
                name=dataset.name,
                phenotype=dataset.phenotype,
                licence=dataset.licence,
                homepage=dataset.homepage,
                approx_image_count=int(dataset.approx_image_count),
                condition_score=int(condition_score),
                downloadable=downloadable,
                reason=reason,
            ))
        for source in BLOCKED_RESEARCH_ONLY_SOURCES:
            hay = _norm(" ".join([
                source["dataset_id"], source["name"], source["phenotype"],
                " ".join(source["keywords"]),
            ]))
            cell_hit = any(
                token in hay for token in _norm(cell_line).replace("-", " ").split()
                if token
            )
            if not cell_hit:
                continue
            condition_score = sum(
                2 for term in LIGHT_MICROSCOPY_TERMS if term in hay
            )
            candidates.append(ResearchCandidate(
                cell_line=cell_line,
                dataset_id=source["dataset_id"],
                name=source["name"],
                phenotype=source["phenotype"],
                licence=source["licence"],
                homepage=source["homepage"],
                approx_image_count=int(source["approx_image_count"]),
                condition_score=int(condition_score),
                downloadable=False,
                reason=(
                    f"research candidate only; blocked licence: "
                    f"{source['licence']}"
                ),
            ))
        candidates.sort(
            key=lambda c: (
                c.downloadable,
                c.condition_score,
                c.approx_image_count,
                c.dataset_id,
            ),
            reverse=True,
        )
        return candidates


class CellLineResolverQAgent:
    """Normalizes user-provided cell-line names before research/training."""

    ALIASES = {
        "mcf7": "MCF7",
        "mcf-7": "MCF7",
        "hela": "HeLa",
        "huh7": "Huh7",
        "huh-7": "Huh7",
        "u2os": "U2OS",
        "skov3": "SK-OV-3",
        "sk-ov-3": "SK-OV-3",
    }

    def resolve(self, cell_lines: Iterable[str]) -> List[str]:
        resolved = []
        for raw in cell_lines:
            key = _norm(raw).replace(" ", "")
            value = self.ALIASES.get(key, raw.strip())
            if value and value not in resolved:
                resolved.append(value)
        return resolved


class DatasetResearchQAgent(WebsiteResearchQAgent):
    """Compatibility name for the dataset-search agent in the architecture."""


class UserDataTrainingQAgent:
    """Validates a user-provided class-folder training dataset."""

    def inspect(self, data_dir: str,
                expected_cell_lines: Optional[Iterable[str]] = None) -> dict:
        expected = [x.strip() for x in (expected_cell_lines or []) if x and x.strip()]
        result = {
            "data_dir": data_dir,
            "expected_cell_lines": expected,
            "class_folders": [],
            "missing_expected_cell_lines": [],
            "ready": False,
            "notes": [],
        }
        if not data_dir or not os.path.isdir(data_dir):
            result["notes"].append("training folder does not exist")
            return result

        image_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
        folders = []
        for name in sorted(os.listdir(data_dir)):
            path = os.path.join(data_dir, name)
            if not os.path.isdir(path) or name.startswith("."):
                continue
            count = sum(
                1 for item in os.listdir(path)
                if item.lower().endswith(image_exts)
            )
            folders.append({"class_label": name, "image_count": count})
        result["class_folders"] = folders

        labels_norm = {_norm(item["class_label"]).replace(" ", "")
                       for item in folders}
        missing = []
        for line in expected:
            key = _norm(line).replace(" ", "")
            if key not in labels_norm:
                missing.append(line)
        result["missing_expected_cell_lines"] = missing
        if not folders:
            result["notes"].append("no class folders with images were found")
        if missing:
            result["notes"].append(
                "some requested cell lines are not represented by folder names")
        if any(item["image_count"] <= 0 for item in folders):
            result["notes"].append("one or more class folders contain no images")

        result["ready"] = bool(folders and not missing and
                               all(item["image_count"] > 0 for item in folders))
        return result


class ConditionMatcherQAgent:
    """Classifies the requested microscopy condition for backend selection."""

    def match(self, condition_query: str) -> ConditionProfile:
        q = (condition_query or "").lower()
        notes = []
        if any(term in q for term in ("fluorescence", "fluo", "dark background")):
            modality = "fluorescence_bright_on_dark"
        elif any(term in q for term in ("dic", "phase", "phase-contrast")):
            modality = "label_free_dic_phase"
        elif any(term in q for term in ("brightfield", "bright-field")):
            modality = "brightfield_dark_on_bright"
        else:
            modality = "unknown_light_microscopy"
            notes.append("modality was not explicit; detector calibration must decide")

        if any(term in q for term in ("confluent", "dense", "crowded", "1000")):
            density = "dense"
        elif any(term in q for term in ("sparse", "few", "15")):
            density = "sparse"
        else:
            density = "unknown_density"
        return ConditionProfile(condition_query, modality, density, notes)


def _select_available(backends: Iterable[BackendSpec],
                      preferred: Iterable[str]) -> BackendSelection:
    specs = list(backends)
    by_name = {b.name: b for b in specs}
    available = [b.name for b in specs if b.available]
    unavailable = [b.name for b in specs if not b.available]
    selected = [name for name in preferred
                if name in by_name and by_name[name].available]
    if not selected:
        selected = [b.name for b in sorted(
            specs, key=lambda s: s.priority, reverse=True) if b.available][:1]
    return BackendSelection(
        selected=selected,
        available=available,
        unavailable=unavailable,
        notes=[] if selected else ["no available preferred backend"],
    )


class DetectorEnsembleQAgent:
    """Chooses detector backend candidates for calibration frames."""

    def select(self, condition: ConditionProfile) -> BackendSelection:
        if condition.modality == "fluorescence_bright_on_dark":
            preferred = ["cellpose-sam", "stardist", "monai-vista2d",
                         "classical-fallback"]
        elif condition.modality == "label_free_dic_phase":
            preferred = ["cellpose-sam", "micro-sam", "omnipose",
                         "classical-fallback"]
        else:
            preferred = ["cellpose-sam", "micro-sam", "classical-fallback"]
        return _select_available(SEGMENTATION_BACKENDS, preferred)


class TrackingCuratorQAgent:
    """Scores tracks for likely identity jumps after linking."""

    def flag_suspicious_steps(self, tracks: Dict[int, object],
                              max_step_px: float) -> List[dict]:
        flags = []
        for tid, track in tracks.items():
            boxes = track.get("boxes", []) if isinstance(track, dict) else getattr(track, "boxes", [])
            for idx in range(1, len(boxes)):
                a = boxes[idx - 1]
                b = boxes[idx]
                ax = a[0] + a[2] / 2.0
                ay = a[1] + a[3] / 2.0
                bx = b[0] + b[2] / 2.0
                by = b[1] + b[3] / 2.0
                step = float(((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5)
                if step > max_step_px:
                    flags.append({
                        "track_id": tid,
                        "frame": idx,
                        "step_px": step,
                        "reason": "possible identity jump or missed detection",
                    })
        return flags

    def select(self, condition: ConditionProfile) -> BackendSelection:
        preferred = ["trackastra", "ultrack", "btrack",
                     "local-kalman-hungarian"]
        if condition.density == "dense":
            preferred = ["ultrack", "trackastra", "btrack",
                         "local-kalman-hungarian"]
        return _select_available(TRACKING_BACKENDS, preferred)


class VisualAuditQAgent:
    """Writes suspicious-transition audit rows for dashboard consumption."""

    def write_flags_csv(self, flags: List[dict], output_path: str) -> str:
        import csv

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fieldnames = ["track_id", "frame", "step_px", "reason"]
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in flags:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        return output_path


class NoCellBaselineCuratorQAgent:
    """
    Blocks first-frame chamber/background texture from seeding tracks.

    Some migration videos begin with an empty channel and cells enter later.
    In that case frame-zero phase halos, chamber walls, timestamps, and scale
    bars must be treated as baseline artifacts, not cells.
    """

    def allow_detections(self, frame_index: int,
                         baseline_empty_until_frame: int = 0) -> bool:
        return int(frame_index) > int(baseline_empty_until_frame)


class StaticArtifactCuratorQAgent:
    """Rejects candidates that look unchanged from the no-cell baseline."""

    def accept_candidate(self,
                         temporal_delta: float,
                         static_artifact_score: float,
                         min_temporal_delta: float,
                         max_static_artifact_score: float) -> bool:
        if temporal_delta < min_temporal_delta:
            return False
        if (static_artifact_score > max_static_artifact_score and
                temporal_delta < min_temporal_delta * 1.35):
            return False
        return True


class CellBirthCuratorQAgent:
    """
    Requires repeated observations before a new track/agent is created.

    This prevents one-frame debris or wall texture from becoming a permanent
    cell identity.
    """

    def __init__(self, required_persistence_frames: int = 3):
        self.required_persistence_frames = int(required_persistence_frames)

    def confirmed(self, observed_frame_indices: Iterable[int]) -> bool:
        frames = sorted({int(f) for f in observed_frame_indices})
        if len(frames) < self.required_persistence_frames:
            return False
        tail = frames[-self.required_persistence_frames:]
        return all((b - a) == 1 for a, b in zip(tail, tail[1:]))


class VideoMorphologyTrainingQAgent:
    """
    Plans morphology calibration from the same video before final tracking.

    This is used when public/local training data are not enough for the exact
    acquisition format. It deliberately excludes baseline frames and samples
    later frames where the requested cell line is visible.
    """

    def select_training_frames(
        self,
        total_frames: int,
        baseline_empty_until_frame: int = 1,
        max_training_frames: int = 8,
    ) -> List[int]:
        total = int(total_frames)
        start = int(baseline_empty_until_frame) + 1
        if total <= start:
            return []
        usable = list(range(start, total))
        if len(usable) <= max_training_frames:
            return usable
        # Bias toward later frames because migration videos often begin empty.
        positions = np.linspace(
            max(start, total // 4),
            total - 1,
            int(max_training_frames),
        )
        return sorted({int(round(p)) for p in positions})

    def build_manifest(
        self,
        cell_line: str,
        total_frames: int,
        baseline_empty_until_frame: int = 1,
    ) -> Dict[str, object]:
        frames = self.select_training_frames(
            total_frames,
            baseline_empty_until_frame=baseline_empty_until_frame,
        )
        return {
            "cell_line": cell_line,
            "training_source": "same_video_before_final_tracking",
            "baseline_empty_until_frame": int(baseline_empty_until_frame),
            "selected_training_frames": frames,
            "center_policy": "train morphology for center detections, not edges",
        }


class PerCellVisualAgentQAgent:
    """
    Owns one cell identity and refuses impossible ownership transfers.

    The tracker may have many visual agents, but each agent is responsible for
    only one cell. New detections are accepted only when their center movement
    stays inside the configured gate and, when supplied, their appearance
    similarity is high enough.
    """

    def __init__(self,
                 track_id: int,
                 max_center_step_px: float = 15.0,
                 min_appearance_similarity: float = 0.15):
        self.track_id = int(track_id)
        self.max_center_step_px = float(max_center_step_px)
        self.min_appearance_similarity = float(min_appearance_similarity)

    def accept_step(self,
                    previous_center: Tuple[float, float],
                    candidate_center: Tuple[float, float],
                    appearance_similarity: Optional[float] = None) -> bool:
        dx = float(candidate_center[0]) - float(previous_center[0])
        dy = float(candidate_center[1]) - float(previous_center[1])
        step = (dx ** 2 + dy ** 2) ** 0.5
        if step > self.max_center_step_px:
            return False
        if (appearance_similarity is not None and
                appearance_similarity < self.min_appearance_similarity):
            return False
        return True


class FrameMemoryQAgent:
    """
    Maintains per-cell memory across frames for quality-first tracking.

    Memory contains the last accepted center, an estimated velocity, and a
    compact appearance vector. It lets the next-frame linker reason from the
    previous frame instead of treating each frame independently.
    """

    def __init__(self, max_center_step_px: float = 15.0):
        self.max_center_step_px = float(max_center_step_px)
        self._memory: Dict[int, Dict[str, object]] = {}

    def update(
        self,
        track_id: int,
        frame_index: int,
        center: Tuple[float, float],
        appearance_vector: Optional[Iterable[float]] = None,
    ) -> Dict[str, object]:
        center_arr = np.asarray(center, dtype=float)
        prev = self._memory.get(int(track_id))
        velocity = np.zeros(2, dtype=float)
        if prev is not None:
            prev_center = np.asarray(prev["center"], dtype=float)
            velocity = center_arr - prev_center
        memory = {
            "track_id": int(track_id),
            "frame_index": int(frame_index),
            "center": (float(center_arr[0]), float(center_arr[1])),
            "velocity": (float(velocity[0]), float(velocity[1])),
            "appearance_vector": (
                list(appearance_vector) if appearance_vector is not None else None),
        }
        self._memory[int(track_id)] = memory
        return memory

    def predict_center(self, track_id: int) -> Optional[Tuple[float, float]]:
        memory = self._memory.get(int(track_id))
        if memory is None:
            return None
        c = np.asarray(memory["center"], dtype=float)
        v = np.asarray(memory["velocity"], dtype=float)
        p = c + v
        return (float(p[0]), float(p[1]))

    def accepts(self, track_id: int,
                candidate_center: Tuple[float, float]) -> bool:
        predicted = self.predict_center(track_id)
        if predicted is None:
            return True
        step = float(np.hypot(
            float(candidate_center[0]) - predicted[0],
            float(candidate_center[1]) - predicted[1],
        ))
        return step <= self.max_center_step_px


class BottomRegionCoverageQAgent:
    """Audits whether late-frame lower cells are represented by tracks."""

    def __init__(self, bottom_y_fraction: float = 0.72,
                 min_bottom_fraction: float = 0.12):
        self.bottom_y_fraction = float(bottom_y_fraction)
        self.min_bottom_fraction = float(min_bottom_fraction)

    def audit(self, frame_height: int,
              centers: Iterable[Tuple[float, float]]) -> Dict[str, object]:
        centers = list(centers)
        if not centers:
            return {
                "bottom_count": 0,
                "total_count": 0,
                "bottom_fraction": 0.0,
                "passes": False,
            }
        threshold = float(frame_height) * self.bottom_y_fraction
        bottom_count = sum(1 for _, y in centers if float(y) >= threshold)
        fraction = bottom_count / max(1, len(centers))
        return {
            "bottom_count": bottom_count,
            "total_count": len(centers),
            "bottom_fraction": fraction,
            "passes": fraction >= self.min_bottom_fraction,
        }


class WholeCellBorderQAgent:
    """
    Rejects partial edge-fragment borders.

    A contour can exist but still be unusable if it traces only a DIC rim,
    lamellipodia edge, or texture island. This curator requires the instance
    contour to occupy enough of its bbox to be a whole-cell body outline.
    """

    def __init__(self, min_extent: float = 0.55):
        self.min_extent = float(min_extent)

    def border_extent(self, contour, bbox: Tuple[int, int, int, int]) -> float:
        if contour is None:
            return 0.0
        _, _, w, h = [int(v) for v in bbox]
        area = float(max(1, w * h))
        contour_area = float(cv2.contourArea(contour))
        return contour_area / area

    def accepts(self, contour, bbox: Tuple[int, int, int, int]) -> bool:
        return self.border_extent(contour, bbox) >= self.min_extent


class WallArtifactCuratorQAgent:
    """Rejects detections too close to known chamber-wall artifact bands."""

    def __init__(self, left_margin_px: int = 58, right_margin_px: int = 312):
        self.left_margin_px = int(left_margin_px)
        self.right_margin_px = int(right_margin_px)

    def accept_center(self, center_x: float) -> bool:
        x = float(center_x)
        return self.left_margin_px <= x <= self.right_margin_px


class MicroscopyInsetExtractionQAgent:
    """
    Locates a microscopy inset inside presentation/screen-recorded videos.

    Some public videos embed the actual microscope field inside a larger
    slide, social-media frame, or lecture screen. Tracking must run on the
    microscope inset, not the full presentation frame.
    """

    def accept_crop(self,
                    bbox: Tuple[int, int, int, int],
                    frame_shape: Tuple[int, int],
                    min_area_fraction: float = 0.04) -> bool:
        h, w = int(frame_shape[0]), int(frame_shape[1])
        x, y, bw, bh = [int(v) for v in bbox]
        if bw <= 0 or bh <= 0:
            return False
        if x < 0 or y < 0 or x + bw > w or y + bh > h:
            return False
        area_fraction = (bw * bh) / max(1.0, float(w * h))
        return area_fraction >= float(min_area_fraction)

    def expand_crop(self,
                    bbox: Tuple[int, int, int, int],
                    frame_shape: Tuple[int, int],
                    pad_fraction: float = 0.04) -> Tuple[int, int, int, int]:
        h, w = int(frame_shape[0]), int(frame_shape[1])
        x, y, bw, bh = [int(v) for v in bbox]
        pad = int(round(max(bw, bh) * float(pad_fraction)))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)
        return (x1, y1, x2 - x1, y2 - y1)


class IdentityJumpRepairQAgent:
    """Splits trajectories when center motion exceeds the accepted gate."""

    def __init__(self, max_step_px: float = 18.0,
                 min_segment_length: int = 8):
        self.max_step_px = float(max_step_px)
        self.min_segment_length = int(min_segment_length)

    def split_centers(
        self,
        points: Iterable[Tuple[int, float, float]],
    ) -> List[List[Tuple[int, float, float]]]:
        ordered = sorted(
            [(int(f), float(x), float(y)) for f, x, y in points],
            key=lambda p: p[0],
        )
        if not ordered:
            return []
        segments: List[List[Tuple[int, float, float]]] = [[ordered[0]]]
        for prev, curr in zip(ordered[:-1], ordered[1:]):
            frame_gap = curr[0] - prev[0]
            step = float(np.hypot(curr[1] - prev[1], curr[2] - prev[2]))
            if frame_gap != 1 or step > self.max_step_px:
                segments.append([curr])
            else:
                segments[-1].append(curr)
        return [
            segment for segment in segments
            if len(segment) >= self.min_segment_length
        ]


class DashboardQAgent:
    """Defines the minimum dashboard assets expected from a tracking run."""

    REQUIRED_ASSETS = [
        "tracking_video.avi",
        "migration_detailed.csv",
        "migration_summary.csv",
        "plot_trajectories.png",
        "plot_velocity_histogram.png",
        "pipeline_architecture_manifest.json",
    ]

    def missing_assets(self, output_dir: str) -> List[str]:
        return [
            name for name in self.REQUIRED_ASSETS
            if not os.path.exists(os.path.join(output_dir, name))
        ]


class LicenceCuratorQAgent:
    """Selects only open-licensed, sufficiently large candidates."""

    def select(
        self,
        cell_line: str,
        candidates: Iterable[ResearchCandidate],
        min_images: int,
    ) -> TrainingClassPlan:
        plan = TrainingClassPlan(
            class_label=cell_line.strip().replace(os.sep, "_"),
            cell_line=cell_line,
            min_required_images=min_images,
            candidates=list(candidates),
        )
        for candidate in plan.candidates:
            if not candidate.downloadable:
                continue
            if candidate.approx_image_count < min_images:
                plan.notes.append(
                    f"{candidate.dataset_id} matched but has only "
                    f"~{candidate.approx_image_count} images; need {min_images}.")
                continue
            plan.selected_dataset_id = candidate.dataset_id
            plan.selected_dataset_name = candidate.name
            plan.selected_dataset_homepage = candidate.homepage
            plan.selected_dataset_licence = candidate.licence
            plan.approx_available_images = candidate.approx_image_count
            plan.status = "ready"
            plan.notes.append("selected by licence and condition curator")
            return plan

        plan.status = "needs_user_data"
        plan.notes.append(
            "No open-licensed public candidate met the requested cell line, "
            "condition, and minimum image count. Add a local labelled folder "
            "or register an open dataset before automatic training.")
        return plan


LicenseCuratorQAgent = LicenceCuratorQAgent


class MorphologyTrainingQAgent:
    """
    Orchestrates pre-tracking morphology data preparation.

    It writes a JSON plan in every output folder. If every class has a ready
    open dataset, it can also download the images into class folders suitable
    for CellClassifierTrainer.
    """

    def __init__(self,
                 researcher: Optional[WebsiteResearchQAgent] = None,
                 curator: Optional[LicenceCuratorQAgent] = None,
                 project_root: Optional[str] = None):
        self.researcher = researcher or WebsiteResearchQAgent()
        self.curator = curator or LicenceCuratorQAgent()
        self.project_root = project_root
        self.resolver = CellLineResolverQAgent()
        self.condition_matcher = ConditionMatcherQAgent()
        self.detector_ensemble = DetectorEnsembleQAgent()
        self.tracking_curator = TrackingCuratorQAgent()

    @property
    def cache_root(self) -> str:
        return default_model_cache_root(self.project_root)

    def plan(
        self,
        cell_lines: Iterable[str],
        condition_query: str = "light microscope phase contrast brightfield DIC",
        min_images_per_cell_line: int = 200,
    ) -> MorphologyTrainingPlan:
        class_plans = []
        resolved_lines = self.resolver.resolve(cell_lines)
        condition_profile = self.condition_matcher.match(condition_query)
        detector_selection = self.detector_ensemble.select(condition_profile)
        tracking_selection = self.tracking_curator.select(condition_profile)
        for raw in resolved_lines:
            cell_line = raw.strip()
            if not cell_line:
                continue
            candidates = self.researcher.research(cell_line, condition_query)
            class_plans.append(
                self.curator.select(cell_line, candidates,
                                    min_images_per_cell_line)
            )
        return MorphologyTrainingPlan(
            generated_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            condition_query=condition_query,
            min_images_per_cell_line=min_images_per_cell_line,
            class_plans=class_plans,
            architecture=[
                "CellLineResolverQAgent: normalizes requested cell-line names",
                "ConditionMatcherQAgent: classifies modality and density before tracking",
                "DetectorEnsembleQAgent: selected "
                + ", ".join(detector_selection.selected),
                "TrackingCuratorQAgent: selected "
                + ", ".join(tracking_selection.selected),
                "NoCellBaselineCuratorQAgent: blocks empty baseline frames from seeding tracks",
                "VideoMorphologyTrainingQAgent: trains same-video morphology prototypes before final tracking",
                "StaticArtifactCuratorQAgent: rejects unchanged chamber, timestamp, scale-bar, and wall texture",
                "CellBirthCuratorQAgent: requires repeated observations before creating a new cell identity",
                "PerCellVisualAgentQAgent: one visual agent owns one cell and refuses long center jumps",
                "FrameMemoryQAgent: carries per-cell center, velocity, and appearance memory into the next frame",
                "BottomRegionCoverageQAgent: audits lower-frame cells so bottom migration is not lost",
                "WallArtifactCuratorQAgent: removes chamber-wall artifact bands before counting cells",
                "MicroscopyInsetExtractionQAgent: crops microscope inset before detection in presentation videos",
                "IdentityJumpRepairQAgent: splits trajectories with impossible center jumps before metrics",
                "WebsiteResearchQAgent: finds public microscopy dataset candidates",
                "LicenceCuratorQAgent: blocks non-open or undersized sources",
                "UserDataTrainingQAgent: validates local class folders when the user provides training images",
                "MorphologyTrainingQAgent: creates/downloads class folders before tracking in model_cache, not RESULT",
                "ClassifierTrainer: trains morphology model from prepared folders",
                "ResultBundleCurator: stores only run videos, dashboards, metrics, and QC in RESULT",
            ],
        )

    def default_output_dir(self, namespace: str = "qagent_morphology_pretraining") -> str:
        return os.path.join(self.cache_root, namespace)

    def write_plan(self, plan: MorphologyTrainingPlan, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "morphology_qagent_plan.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(plan.to_dict(), f, indent=2, sort_keys=True)
        return path

    def prepare_training_data(
        self,
        plan: MorphologyTrainingPlan,
        target_root: str,
        max_samples_per_class: Optional[int] = None,
        url_opener=None,
    ) -> str:
        selections: List[Tuple[str, Dataset]] = []
        by_id = {dataset.id: dataset for dataset in catalogue()}
        for class_plan in plan.class_plans:
            if class_plan.status != "ready" or not class_plan.selected_dataset_id:
                continue
            dataset = by_id[class_plan.selected_dataset_id]
            selections.append((class_plan.class_label, dataset))
        if len(selections) != len(plan.class_plans):
            missing = [p.cell_line for p in plan.class_plans if p.status != "ready"]
            raise RuntimeError(
                "Cannot auto-download training data for: " + ", ".join(missing))
        per_class = max_samples_per_class or plan.min_images_per_cell_line
        return build_phenotype_folders(
            selections,
            target_root,
            max_samples_per_class=per_class,
            url_opener=url_opener,
        )


def parse_cell_lines(text: str) -> List[str]:
    return [part.strip() for part in (text or "").split(",") if part.strip()]
