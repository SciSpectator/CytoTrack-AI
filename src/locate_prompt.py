"""
Prompt helpers for the NVIDIA LocateAnything detector backend.

LocateAnything is a per-frame visual grounding model. It returns boxes for
objects described by text; tracking and motion analysis are still handled by
CellTracker downstream.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np


DEFAULT_CELL_LOCATE_QUERY = (
    "individual visible biological cells in a microscopy time-lapse frame; "
    "round or oval bright gray cell bodies, including dim, partially "
    "overlapping, touching, or clustered cells; return one tight bounding box "
    "per visible cell; the tracked point is the center/centroid of each cell, "
    "not the cell edge or boundary; ignore dark background, halos, noise, "
    "debris, text, scale bars, and colored tracking overlays"
)


@dataclass(frozen=True)
class LocatePromptResult:
    query: str
    method: str
    note: str = ""


@dataclass(frozen=True)
class MicroscopyFrameProfile:
    modality_hint: str
    polarity: str
    contrast: str
    prompt_clause: str


def is_generic_cell_query(query: Optional[str]) -> bool:
    q = (query or "").strip().lower()
    return q in {"", "cell", "cells", "biological cell", "biological cells"}


def deterministic_cell_query(query: Optional[str] = None) -> str:
    """Return a microscopy-specific query unless the caller supplied detail."""
    if is_generic_cell_query(query):
        return DEFAULT_CELL_LOCATE_QUERY
    return str(query).strip()


def analyze_microscopy_frame(image) -> MicroscopyFrameProfile:
    """
    Infer a lightweight visual profile for prompt adaptation.

    This is intentionally based on robust image statistics, not model output:
    the prompt should change before detection when the video switches between
    fluorescent bright-on-dark frames and low-contrast DIC/phase frames.
    """
    if image is None:
        return MicroscopyFrameProfile(
            modality_hint="unknown microscopy",
            polarity="unknown polarity",
            contrast="unknown contrast",
            prompt_clause=(
                "microscopy frame with unknown contrast; locate whole visible "
                "cell bodies and their centers"
            ),
        )

    arr = np.asarray(image)
    if arr.ndim == 3:
        # BGR/RGB ordering does not matter for a mean luminance estimate.
        gray = arr.astype(np.float32).mean(axis=2)
    else:
        gray = arr.astype(np.float32)
    if gray.size == 0:
        return analyze_microscopy_frame(None)

    p1, p10, p50, p90, p99 = np.percentile(gray, [1, 10, 50, 90, 99])
    mean = float(gray.mean())
    std = float(gray.std())
    dynamic = float(p99 - p1)
    bright_tail = float(p99 - p50)
    dark_tail = float(p50 - p1)

    if mean < 55 and bright_tail > max(25.0, dark_tail * 1.5):
        return MicroscopyFrameProfile(
            modality_hint="fluorescence / bright cells on dark background",
            polarity="bright-on-dark",
            contrast="high contrast" if dynamic > 80 else "dim fluorescence",
            prompt_clause=(
                "fluorescence-style frame: cells appear as bright, white, "
                "gray, or glowing regions on a dark background. Detect every "
                "bright cell body or nucleus, including dim cells, saturated "
                "cells, partially cropped border cells, and clustered cells. "
                "Do not confuse black background holes with cells"
            ),
        )

    if mean > 150 and dark_tail > max(20.0, bright_tail * 0.8):
        return MicroscopyFrameProfile(
            modality_hint="brightfield / dark cells on bright background",
            polarity="dark-on-bright",
            contrast="low contrast" if std < 35 else "moderate contrast",
            prompt_clause=(
                "brightfield-style frame: cells appear as darker gray cell "
                "bodies, shadows, outlines, or textured regions on a pale "
                "background. Detect whole cell bodies and centers, not just "
                "the dark rim, edge, or internal granules"
            ),
        )

    return MicroscopyFrameProfile(
        modality_hint="DIC / phase contrast / mixed light microscopy",
        polarity="mixed halos and transparent cell bodies",
        contrast="low contrast" if std < 30 else "moderate contrast",
        prompt_clause=(
            "DIC or phase-contrast light microscopy frame: cells are "
            "transparent low-contrast objects with bright/dark halos, rims, "
            "and internal texture. Detect the entire cell body as one object "
            "and place the tracked point at the center/centroid. Do not create "
            "separate detections for internal speckles, nuclei, edges, halos, "
            "or texture patches inside the same cell"
        ),
    )


def build_locate_question(query: str) -> str:
    """
    Build the actual instruction sent to LocateAnything.

    Keep it concise and visual. Do not ask for reasoning: this model is used as
    a grounding detector, and downstream code parses only returned boxes.
    """
    q = deterministic_cell_query(query)
    return (
        "Locate every instance matching this visual description. "
        "Use a separate tight bounding box for each individual object, even "
        "when objects touch or overlap. Do not merge neighboring instances. "
        "The downstream tracker measures the cell center/centroid from each "
        "box; do not target edges, halos, membranes, or partial boundaries as "
        "the tracked point. "
        "If uncertain, include plausible faint instances rather than missing "
        "them. Description: "
        f"{q}."
    )


def build_adaptive_locate_question(query: str, image) -> str:
    """Build a LocateAnything question tailored to this exact frame format."""
    profile = analyze_microscopy_frame(image)
    base = build_locate_question(query)
    return (
        f"{base}\n\n"
        "Frame-specific microscope format:\n"
        f"- modality: {profile.modality_hint}\n"
        f"- polarity: {profile.polarity}\n"
        f"- contrast: {profile.contrast}\n"
        f"- instruction: {profile.prompt_clause}\n\n"
        "Quality requirement: maximize recall of real cells in this frame. "
        "Every visible cell should have one box and one center. Missing cells "
        "is worse than being slow. Avoid duplicate boxes on the same cell."
    )


def _configure_dspy_lm() -> bool:
    """
    Best-effort DSPy LM setup. Returns True only when DSPy is installed and an
    LM is configured. This remains optional so the detector works offline.
    """
    try:
        import dspy  # type: ignore
    except Exception:
        return False

    try:
        if getattr(dspy.settings, "lm", None) is not None:
            return True
        if os.environ.get("OPENAI_API_KEY"):
            dspy.settings.configure(
                lm=dspy.LM(model=os.environ.get("DSPY_OPENAI_MODEL", "openai/gpt-4o-mini"),
                           max_tokens=300)
            )
            return True
        if os.environ.get("ANTHROPIC_API_KEY"):
            dspy.settings.configure(
                lm=dspy.LM(model=os.environ.get("DSPY_ANTHROPIC_MODEL",
                                                "anthropic/claude-3-5-haiku-latest"),
                           max_tokens=300)
            )
            return True
    except Exception:
        return False
    return False


def optimize_cell_query_with_dspy(
    query: Optional[str] = None,
    context: str = "",
) -> LocatePromptResult:
    """
    Optional DSPy ChainOfThought prompt rewrite.

    The generated query is constrained to a short detector prompt. If DSPy or
    an LM is unavailable, callers get the deterministic microscopy query.
    """
    fallback = deterministic_cell_query(query)
    if not _configure_dspy_lm():
        return LocatePromptResult(fallback, "deterministic",
                                  "DSPy unavailable or no LM configured")

    try:
        import dspy  # type: ignore

        class CellLocatePromptSignature(dspy.Signature):
            """Rewrite a generic cell-detection request into a concise visual grounding prompt."""

            base_query = dspy.InputField()
            image_context = dspy.InputField()
            optimized_query = dspy.OutputField(
                desc=(
                    "One concise visual description for a bounding-box detector. "
                    "Mention individual cells, separate boxes for touching or "
                    "overlapping cells, faint cells, center/centroid tracking "
                    "rather than edge tracking, and ignored non-cell artifacts."
                )
            )

        program = dspy.ChainOfThought(CellLocatePromptSignature)
        pred = program(
            base_query=fallback,
            image_context=context or (
                "Synthetic or real time-lapse microscopy frame with bright "
                "gray cells on a dark noisy background."
            ),
        )
        optimized = str(getattr(pred, "optimized_query", "")).strip()
        if len(optimized) < 20:
            return LocatePromptResult(fallback, "deterministic",
                                      "DSPy returned an empty/short query")
        return LocatePromptResult(optimized, "dspy_chain_of_thought")
    except Exception as e:
        return LocatePromptResult(fallback, "deterministic",
                                  f"DSPy optimization failed: {e}")
