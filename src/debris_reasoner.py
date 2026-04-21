"""
CytoTrack AI — DSPy Debris-vs-Cell Reasoner
============================================
For every ambiguous detection the pipeline can ask this reasoner:
"is this really a cell, or debris?" The reasoner packages up
measurable features of the detection and runs them through a
DSPy program that produces a structured Yes/No answer plus an
explanation.

Two DSPy programs are supported and are chosen at runtime:

  * ``ChainOfThought`` — the usual DSPy CoT signature.
  * ``ReAct``          — ReAct (Reason+Act) style, with three
                         tools so the LM can actively request
                         additional measurements:
                           - ``shape_tool``    : circularity / aspect ratio
                           - ``intensity_tool`` : intensity profile stats
                           - ``neighbor_tool`` : are there similar cells
                                                 nearby?

If DSPy is not installed (or no language-model backend is
configured), the reasoner transparently falls back to a
deterministic rule-based classifier so the rest of the pipeline
always works.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import dspy  # type: ignore
    HAS_DSPY = True
except Exception:
    HAS_DSPY = False


# ============================================================ feature bag
@dataclass
class CellFeatures:
    """Numerical features extracted from a crop. Used by tools and
    ultimately by the DSPy program."""
    area: float
    width: int
    height: int
    aspect_ratio: float
    circularity: float
    fg_fraction: float
    mean_intensity: float
    std_intensity: float
    neighbor_count: int = 0
    neighbor_similarity: float = 0.0

    def as_text(self) -> str:
        return (
            f"area={self.area:.1f}, size={self.width}x{self.height}, "
            f"aspect={self.aspect_ratio:.2f}, circularity={self.circularity:.2f}, "
            f"fg_fraction={self.fg_fraction:.2f}, "
            f"mean={self.mean_intensity:.1f}, std={self.std_intensity:.1f}, "
            f"neighbors={self.neighbor_count}, "
            f"neighbor_sim={self.neighbor_similarity:.2f}"
        )


def _crop(frame: np.ndarray, bbox) -> np.ndarray:
    x, y, w, h = [int(v) for v in bbox]
    pad = 4
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame.shape[1], x + w + pad)
    y2 = min(frame.shape[0], y + h + pad)
    return frame[y1:y2, x1:x2].copy()


def extract_features(frame: np.ndarray, bbox,
                     neighbors: Optional[List] = None) -> CellFeatures:
    """Compute classical features for a candidate detection."""
    crop = _crop(frame, bbox)
    if crop.size == 0 or not HAS_CV2:
        return CellFeatures(0.0, 0, 0, 1.0, 0.0, 0.0, 0.0, 0.0)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    h, w = gray.shape
    mean_int = float(gray.mean())
    std_int = float(gray.std())

    _, mask = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if mean_int > 128:
        mask = cv2.bitwise_not(mask)
    fg_fraction = float(mask.sum()) / (255.0 * mask.size)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))
        perim = float(cv2.arcLength(cnt, True))
        circularity = (4 * np.pi * area / (perim * perim)) if perim > 0 else 0.0
    else:
        area = 0.0
        circularity = 0.0

    aspect = (max(w, h) / max(1, min(w, h))) if min(w, h) > 0 else 1.0

    nc = 0
    nsim = 0.0
    if neighbors:
        bx = bbox[0] + bbox[2] / 2.0
        by = bbox[1] + bbox[3] / 2.0
        radius = max(bbox[2], bbox[3]) * 2.5
        sims = []
        for nb in neighbors:
            nxc = nb[0] + nb[2] / 2.0
            nyc = nb[1] + nb[3] / 2.0
            d = np.hypot(bx - nxc, by - nyc)
            if d < radius and d > 1:
                nc += 1
                size_ratio = min(bbox[2], nb[2]) / max(bbox[2], nb[2] or 1)
                sims.append(size_ratio)
        if sims:
            nsim = float(np.mean(sims))

    return CellFeatures(area=area, width=w, height=h,
                        aspect_ratio=aspect, circularity=circularity,
                        fg_fraction=fg_fraction,
                        mean_intensity=mean_int, std_intensity=std_int,
                        neighbor_count=nc, neighbor_similarity=nsim)


# ============================================================ result type
@dataclass
class DebrisJudgement:
    is_cell: bool
    confidence: float
    reasoning: str
    method: str                     # "cot" | "react" | "heuristic"
    trace: List[str] = field(default_factory=list)   # tool calls for ReAct


# ============================================================ tools (ReAct)
class _Tools:
    """Simple tools usable by dspy.ReAct. Each tool takes and returns
    plain strings so they are easy for an LM to call."""

    def __init__(self, features: CellFeatures, trace_log: List[str]):
        self.f = features
        self.log = trace_log

    def shape_tool(self, _query: str = "") -> str:
        self.log.append("shape_tool")
        return (f"circularity={self.f.circularity:.2f}; "
                f"aspect_ratio={self.f.aspect_ratio:.2f}; "
                f"area={self.f.area:.0f}px²")

    def intensity_tool(self, _query: str = "") -> str:
        self.log.append("intensity_tool")
        return (f"mean_intensity={self.f.mean_intensity:.1f}; "
                f"std_intensity={self.f.std_intensity:.1f}; "
                f"foreground_fraction={self.f.fg_fraction:.2f}")

    def neighbor_tool(self, _query: str = "") -> str:
        self.log.append("neighbor_tool")
        return (f"neighbor_count={self.f.neighbor_count}; "
                f"neighbor_size_similarity={self.f.neighbor_similarity:.2f}")


# ============================================================ main class
class DebrisReasoner:
    """
    Public entry point. Usage:

        reasoner = DebrisReasoner(strategy="cot")
        verdict = reasoner.judge(frame, bbox, neighbors=all_other_bboxes)

    ``strategy`` may be "cot", "react", "auto", or "heuristic".
    ``"auto"`` (the default) tries CoT, then falls back to heuristic.
    """

    def __init__(self, strategy: str = "auto",
                 lm: Optional[object] = None):
        self.strategy = strategy
        self._lm = lm
        self._program_cot = None
        self._program_react = None

        if HAS_DSPY:
            self._configure_lm()

    # ------------------------------------------------------ configure LM
    def _configure_lm(self) -> None:
        """Configure dspy.settings.lm if the caller passed one, or if an
        API key is available in the environment."""
        try:
            if self._lm is not None:
                dspy.settings.configure(lm=self._lm)
                return
            if os.environ.get("ANTHROPIC_API_KEY"):
                # DSPy 2.5+ supports a generic 'anthropic/<model>' target
                lm = dspy.LM(model="anthropic/claude-opus-4-7", max_tokens=400)
                dspy.settings.configure(lm=lm)
            elif os.environ.get("OPENAI_API_KEY"):
                lm = dspy.LM(model="openai/gpt-4o-mini", max_tokens=400)
                dspy.settings.configure(lm=lm)
        except Exception:
            # No LM configured — DSPy code paths will gracefully fail and
            # we'll fall through to the heuristic.
            pass

    # ------------------------------------------------------ public entry
    def judge(self, frame: np.ndarray, bbox,
              neighbors: Optional[List] = None,
              extra_context: str = "") -> DebrisJudgement:
        feats = extract_features(frame, bbox, neighbors=neighbors)

        strategy = self.strategy
        if strategy == "auto":
            strategy = "cot" if HAS_DSPY and self._lm_ready() else "heuristic"

        if strategy == "cot" and HAS_DSPY and self._lm_ready():
            try:
                return self._judge_cot(feats, extra_context)
            except Exception as e:
                return self._heuristic_judge(feats,
                                             reason=f"cot-fallback: {e}")

        if strategy == "react" and HAS_DSPY and self._lm_ready():
            try:
                return self._judge_react(feats, extra_context)
            except Exception as e:
                return self._heuristic_judge(feats,
                                             reason=f"react-fallback: {e}")

        return self._heuristic_judge(feats)

    # ------------------------------------------------------- LM readiness
    @staticmethod
    def _lm_ready() -> bool:
        if not HAS_DSPY:
            return False
        try:
            return getattr(dspy.settings, "lm", None) is not None
        except Exception:
            return False

    # =========================================================== CoT
    def _build_cot(self):
        if self._program_cot is not None:
            return self._program_cot

        class CellOrDebris(dspy.Signature):
            """Given quantitative features of a microscopy detection,
            decide whether it is a live cell or debris/artifact."""
            features: str = dspy.InputField(
                desc="Measured features of the detection.")
            context: str = dspy.InputField(
                desc="Any extra context about the experiment.")
            verdict: str = dspy.OutputField(
                desc='One of: "cell", "debris", "ambiguous".')
            confidence: float = dspy.OutputField(
                desc="Confidence between 0 and 1.")

        self._program_cot = dspy.ChainOfThought(CellOrDebris)
        return self._program_cot

    def _judge_cot(self, feats: CellFeatures,
                   context: str) -> DebrisJudgement:
        prog = self._build_cot()
        result = prog(features=feats.as_text(), context=context or "none")
        verdict = str(getattr(result, "verdict", "ambiguous")).lower()
        conf = float(getattr(result, "confidence", 0.5) or 0.5)
        reasoning = str(getattr(result, "reasoning", "")).strip()[:400]
        return DebrisJudgement(
            is_cell=(verdict == "cell"),
            confidence=conf,
            reasoning=reasoning or verdict,
            method="cot",
        )

    # =========================================================== ReAct
    def _build_react(self, tools: _Tools):
        class CellOrDebrisReAct(dspy.Signature):
            """Decide whether a microscopy detection is a live cell or
            debris. Use the available tools to inspect its shape,
            intensity, and neighborhood before committing to an answer."""
            context: str = dspy.InputField(
                desc="Background information about the detection.")
            verdict: str = dspy.OutputField(
                desc='One of: "cell", "debris", "ambiguous".')
            confidence: float = dspy.OutputField(
                desc="Confidence in [0,1].")

        return dspy.ReAct(
            CellOrDebrisReAct,
            tools=[tools.shape_tool, tools.intensity_tool, tools.neighbor_tool],
            max_iters=4,
        )

    def _judge_react(self, feats: CellFeatures,
                     context: str) -> DebrisJudgement:
        trace: List[str] = []
        tools = _Tools(feats, trace)
        prog = self._build_react(tools)
        result = prog(context=context or "none")
        verdict = str(getattr(result, "verdict", "ambiguous")).lower()
        conf = float(getattr(result, "confidence", 0.5) or 0.5)
        reasoning = str(getattr(result, "reasoning", "")).strip()[:400]
        return DebrisJudgement(
            is_cell=(verdict == "cell"),
            confidence=conf,
            reasoning=reasoning or verdict,
            method="react",
            trace=trace,
        )

    # =========================================================== heuristic
    def _heuristic_judge(self, feats: CellFeatures,
                         reason: str = "no LM available") -> DebrisJudgement:
        score = 0.0
        if feats.circularity > 0.55:
            score += 0.35
        elif feats.circularity > 0.35:
            score += 0.15
        if 0.12 < feats.fg_fraction < 0.85:
            score += 0.2
        if feats.aspect_ratio < 2.0:
            score += 0.15
        if feats.std_intensity > 15:
            score += 0.15
        if feats.area > 30:
            score += 0.1
        if feats.neighbor_similarity > 0.6 and feats.neighbor_count >= 1:
            score += 0.05

        score = float(np.clip(score, 0.0, 1.0))
        is_cell = score >= 0.55

        explanation = (
            f"heuristic score={score:.2f} from {feats.as_text()}; {reason}")
        return DebrisJudgement(
            is_cell=is_cell,
            confidence=score,
            reasoning=explanation,
            method="heuristic",
        )


# =========================================================== convenience
def filter_debris(detector, frame: np.ndarray, detections,
                  reasoner: Optional[DebrisReasoner] = None,
                  circularity_floor: float = 0.35) -> Tuple[list, list]:
    """
    Run the reasoner on every detection whose shape features fall in
    an 'ambiguous' range. Returns (kept_detections, rejected_detections).

    Detections with clearly cell-like features (high circularity,
    reasonable size) are kept without querying the reasoner.
    Detections that are clearly wrong (tiny area, very elongated) are
    dropped without querying the reasoner. Everything in-between is
    submitted to the DebrisReasoner.
    """
    reasoner = reasoner or DebrisReasoner(strategy="auto")
    bboxes = [d.bbox if hasattr(d, "bbox") else d for d in detections]

    kept = []
    rejected = []
    for d, bb in zip(detections, bboxes):
        feats = extract_features(frame, bb, neighbors=bboxes)
        if feats.circularity > 0.75 and 0.2 < feats.fg_fraction < 0.8:
            kept.append(d)
            continue
        if feats.area < 10 or feats.aspect_ratio > 4:
            rejected.append((d, "obvious-debris"))
            continue
        if feats.circularity < circularity_floor and feats.fg_fraction < 0.15:
            rejected.append((d, "low-signal"))
            continue

        judgement = reasoner.judge(frame, bb, neighbors=bboxes)
        if judgement.is_cell:
            kept.append(d)
        else:
            rejected.append((d, judgement.method + ":" + judgement.reasoning[:60]))
    return kept, rejected
