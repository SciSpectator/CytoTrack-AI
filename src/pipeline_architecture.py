"""
Quality-first pipeline architecture for CytoTrack AI.

This module turns the enhancement architecture into executable contracts:
backend registries, pre-tracking run plans, cache locations, and run
manifests. Heavy third-party models remain optional dependencies; the code
records which backends are available and which are planned/fallback-only.
"""

from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class BackendSpec:
    name: str
    role: str
    package: Optional[str]
    priority: int
    planned: bool = True

    @property
    def available(self) -> bool:
        if not self.package:
            return True
        return importlib.util.find_spec(self.package) is not None

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["available"] = self.available
        return data


SEGMENTATION_BACKENDS: List[BackendSpec] = [
    BackendSpec(
        "cellpose-sam",
        "primary generalist cell/nucleus instance segmentation",
        "cellpose",
        100,
    ),
    BackendSpec(
        "stardist",
        "nuclei and compact/star-convex cell instance segmentation",
        "stardist",
        80,
    ),
    BackendSpec(
        "deepcell-mesmer",
        "whole-cell and nuclear segmentation for tissue/multiplex images",
        "deepcell",
        70,
    ),
    BackendSpec(
        "omnipose",
        "morphology-diverse and elongated cell segmentation",
        "omnipose",
        60,
    ),
    BackendSpec(
        "micro-sam",
        "SAM-based microscopy annotation, correction, and fine-tuning",
        "micro_sam",
        55,
    ),
    BackendSpec(
        "monai-vista2d",
        "MONAI/NVIDIA-compatible microscopy cell instance segmentation",
        "monai",
        50,
    ),
    BackendSpec(
        "classical-fallback",
        "local threshold/watershed/blob fallback when AI backends are absent",
        None,
        10,
    ),
]

TRACKING_BACKENDS: List[BackendSpec] = [
    BackendSpec(
        "trackastra",
        "transformer association of segmented cells over time",
        "trackastra",
        100,
    ),
    BackendSpec(
        "ultrack",
        "tracking under segmentation uncertainty for dense 2D/3D data",
        "ultrack",
        95,
    ),
    BackendSpec(
        "btrack",
        "Bayesian multi-object tracking and lineage reconstruction",
        "btrack",
        80,
    ),
    BackendSpec(
        "local-kalman-hungarian",
        "built-in transparent fallback tracker with identity QC hooks",
        None,
        40,
    ),
]

RESTORATION_BACKENDS: List[BackendSpec] = [
    BackendSpec("noise2void", "self-supervised microscopy denoising", "n2v", 80),
    BackendSpec("care-csbdeep", "content-aware fluorescence restoration", "csbdeep", 70),
    BackendSpec("denoiseg", "joint denoising and segmentation with few labels", "denoiseg", 65),
    BackendSpec("deepinterpolation", "self-supervised video/time-series denoising", "deepinterpolation", 60),
]


@dataclass
class PipelineRunPlan:
    generated_utc: str
    quality_mode: str
    cell_lines: List[str] = field(default_factory=list)
    condition_query: str = "light microscope phase contrast brightfield DIC"
    min_training_images_per_cell_line: int = 200
    model_cache_root: str = "model_cache"
    result_policy: str = "RESULT contains run outputs only; model research/training cache stays outside RESULT"
    segmentation_backends: List[Dict[str, object]] = field(default_factory=list)
    tracking_backends: List[Dict[str, object]] = field(default_factory=list)
    restoration_backends: List[Dict[str, object]] = field(default_factory=list)
    qagents: List[str] = field(default_factory=list)
    tracking_constraints: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def default_model_cache_root(project_root: Optional[str] = None) -> str:
    root = project_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "model_cache")


def available_backends(backends: Iterable[BackendSpec]) -> List[Dict[str, object]]:
    return [backend.to_dict() for backend in sorted(
        backends, key=lambda b: b.priority, reverse=True)]


def build_quality_first_run_plan(
    cell_lines: Optional[Iterable[str]] = None,
    condition_query: str = "light microscope phase contrast brightfield DIC",
    min_training_images_per_cell_line: int = 200,
    project_root: Optional[str] = None,
    quality_mode: str = "quality_first",
) -> PipelineRunPlan:
    """
    Create the executable architecture plan used by tracking and QAgents.
    """
    return PipelineRunPlan(
        generated_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        quality_mode=quality_mode,
        cell_lines=[c.strip() for c in (cell_lines or []) if c and c.strip()],
        condition_query=condition_query,
        min_training_images_per_cell_line=int(min_training_images_per_cell_line),
        model_cache_root=default_model_cache_root(project_root),
        segmentation_backends=available_backends(SEGMENTATION_BACKENDS),
        tracking_backends=available_backends(TRACKING_BACKENDS),
        restoration_backends=available_backends(RESTORATION_BACKENDS),
        qagents=[
            "CellLineResolverQAgent",
            "DatasetResearchQAgent",
            "LicenceCuratorQAgent",
            "ConditionMatcherQAgent",
            "MorphologyTrainingQAgent",
            "UserDataTrainingQAgent",
            "DetectorEnsembleQAgent",
            "CellBorderCuratorQAgent",
            "CountStabilityQAgent",
            "DuplicateCellCuratorQAgent",
            "EdgeCellCuratorQAgent",
            "SelfRepairCoordinatorQAgent",
            "TrackingCuratorQAgent",
            "NoCellBaselineCuratorQAgent",
            "VideoMorphologyTrainingQAgent",
            "StaticArtifactCuratorQAgent",
            "CellBirthCuratorQAgent",
            "PerCellVisualAgentQAgent",
            "FrameMemoryQAgent",
            "BottomRegionCoverageQAgent",
            "WholeCellBorderQAgent",
            "WallArtifactCuratorQAgent",
            "MicroscopyInsetExtractionQAgent",
            "IdentityJumpRepairQAgent",
            "VisualAuditQAgent",
            "VisualBorderAgent",
            "VisualTrackingAuditAgent",
            "MissedCellHeatmapAgent",
            "DashboardQAgent",
        ],
        tracking_constraints={
            "cell_line_required_before_tracking": True,
            "pretracking_training_sources": [
                "open_website_resources",
                "user_labelled_data",
                "existing_trained_model",
            ],
            "track_point": "mask_centroid_or_distance_transform_center",
            "forbidden_track_point": "cell_edge_or_raw_box_corner",
            "cell_count_source": "instance_borders_required",
            "identity_policy": "prefer slow frame-by-frame validation over fast but jumpy tracks",
            "reject_impossible_velocity": True,
            "empty_baseline_frames_must_not_seed_tracks": True,
            "same_video_morphology_training_before_tracking": True,
            "new_track_requires_temporal_persistence_frames": 3,
            "one_visual_agent_owns_one_cell": True,
            "frame_memory_required_for_dense_video": True,
            "bottom_region_coverage_audit": True,
            "whole_cell_border_required": True,
            "wall_artifact_curator_required": True,
            "microscopy_inset_extraction_before_tracking": True,
            "split_tracks_on_identity_jump_before_metrics": True,
            "static_chamber_artifacts_rejected_by_temporal_foreground": True,
            "forward_backward_validation": quality_mode == "quality_first",
            "export_suspicious_transitions": True,
        },
    )


def write_run_manifest(plan: PipelineRunPlan, output_dir: str) -> str:
    """
    Save the architecture actually used for a tracking run.

    This file is a run manifest, so it belongs with tracking outputs. General
    research reports and training caches do not.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "pipeline_architecture_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(plan.to_dict(), f, indent=2, sort_keys=True)
    return path
