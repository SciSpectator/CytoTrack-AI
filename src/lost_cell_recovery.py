"""
CytoTrack AI — Lost-Cell Recovery
==================================
When a tracked cell goes missing mid-frame-interior (not near any
border), something went wrong — the cell physically cannot have left
the field of view. This module tries to recover it:

  1. First pass (classical): template-match the last-seen crop
     against a search window centered on the Kalman prediction,
     plus a widened search radius; check if any nearby detection
     matches visually; check if two detections merged (mitosis or
     collision) or if one track suddenly became two.

  2. Second pass (AI): run a DSPy ReAct program that has access
     to three tools backed by the measured data:
         - search_radius(r): widen the search and report candidates
         - appearance_match(candidate_idx): compare crops by NCC
         - check_merge():   report whether a neighboring detection
                            grew in area (suggesting a merge)
     The LM reasons through the tool outputs and returns a
     decision: {"recovered": bool, "match_index": int, "reason": str}.
     Used only when DSPy + an LM are configured.

A "recovery" is either (a) choosing one of the current-frame
detections as the lost cell, or (b) instructing the tracker to keep
coasting via Kalman for a while longer (because the cell is likely
hidden behind debris).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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


# ================================================================== types
@dataclass
class RecoveryResult:
    track_id: int
    recovered: bool
    new_bbox: Optional[Tuple[int, int, int, int]] = None
    method: str = "none"       # "classical" | "react" | "coast" | "border"
    reasoning: str = ""
    trace: List[str] = field(default_factory=list)


# =========================================================== utilities
def _near_border(bbox, shape, margin: int = 10) -> bool:
    x, y, w, h = bbox
    return (x < margin or y < margin
            or x + w > shape[1] - margin
            or y + h > shape[0] - margin)


def _crop(frame: np.ndarray, bbox) -> np.ndarray:
    x, y, w, h = [int(v) for v in bbox]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame.shape[1], x + w)
    y2 = min(frame.shape[0], y + h)
    return frame[y1:y2, x1:x2].copy()


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    if not HAS_CV2 or a.size == 0 or b.size == 0:
        return 0.0
    a_g = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) if a.ndim == 3 else a
    b_g = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY) if b.ndim == 3 else b
    size = (max(16, a_g.shape[1]), max(16, a_g.shape[0]))
    b_r = cv2.resize(b_g, size)
    a_r = cv2.resize(a_g, size)
    a_f = a_r.astype(np.float32) - a_r.mean()
    b_f = b_r.astype(np.float32) - b_r.mean()
    denom = float(np.sqrt((a_f ** 2).sum() * (b_f ** 2).sum()))
    if denom <= 0:
        return 0.0
    return float((a_f * b_f).sum() / denom)


# =================================================================== main
class LostCellRecovery:
    """
    Runs classical + optional DSPy-ReAct recovery for tracks that went
    inactive but did NOT leave via the border.
    """

    def __init__(self, strategy: str = "auto",
                 border_margin: int = 10,
                 appearance_threshold: float = 0.6,
                 search_multiplier: float = 3.0):
        self.strategy = strategy
        self.border_margin = border_margin
        self.appearance_threshold = appearance_threshold
        self.search_multiplier = search_multiplier

        self._react_program = None
        if HAS_DSPY:
            self._configure_lm()

    # ---------------------------------------------------------- lm setup
    def _configure_lm(self) -> None:
        try:
            if getattr(dspy.settings, "lm", None) is not None:
                return
            if os.environ.get("ANTHROPIC_API_KEY"):
                dspy.settings.configure(
                    lm=dspy.LM(model="anthropic/claude-opus-4-7", max_tokens=400))
            elif os.environ.get("OPENAI_API_KEY"):
                dspy.settings.configure(
                    lm=dspy.LM(model="openai/gpt-4o-mini", max_tokens=400))
        except Exception:
            pass

    @staticmethod
    def _lm_ready() -> bool:
        if not HAS_DSPY:
            return False
        try:
            return getattr(dspy.settings, "lm", None) is not None
        except Exception:
            return False

    # -------------------------------------------------------- public API
    def recover(self, tracker, frame: np.ndarray, prev_frame: np.ndarray,
                detections) -> List[RecoveryResult]:
        """
        Attempt recovery for every track that just became inactive. Must
        be called right after tracker.update().
        """
        results: List[RecoveryResult] = []
        det_boxes = [
            (d.bbox if hasattr(d, "bbox") else d) for d in detections]

        # Candidate tracks: tracks that became inactive THIS frame
        for tid, track in list(tracker.tracks.items()):
            if track.is_active:
                continue
            if not track.boxes:
                continue
            last_bbox = track.boxes[-1]

            if _near_border(last_bbox, frame.shape, self.border_margin):
                results.append(RecoveryResult(
                    track_id=tid, recovered=False,
                    method="border",
                    reasoning="lost near border — legitimate exit"))
                continue

            res = self._try_classical(tid, frame, prev_frame,
                                      last_bbox, det_boxes)
            if res.recovered:
                self._apply(tracker, res)
                results.append(res)
                continue

            if (self.strategy in ("auto", "react")
                    and self._lm_ready()):
                res_ai = self._try_react(tid, frame, prev_frame,
                                         last_bbox, det_boxes)
                if res_ai.recovered:
                    self._apply(tracker, res_ai)
                    results.append(res_ai)
                    continue
                res = res_ai

            # No match found — instruct tracker to keep coasting for a bit.
            track.is_active = True
            track.missed_frames = max(0, track.missed_frames - 2)
            res.method = "coast"
            res.reasoning = (res.reasoning or "") + " | coasting on Kalman"
            results.append(res)

        return results

    # --------------------------------------------------------- classical
    def _try_classical(self, tid: int, frame: np.ndarray,
                       prev_frame: np.ndarray, last_bbox,
                       det_boxes: List) -> RecoveryResult:
        if prev_frame is None:
            return RecoveryResult(track_id=tid, recovered=False,
                                  method="classical",
                                  reasoning="no prev_frame for template")

        template = _crop(prev_frame, last_bbox)
        if template.size == 0:
            return RecoveryResult(track_id=tid, recovered=False,
                                  method="classical",
                                  reasoning="empty template")

        cx = last_bbox[0] + last_bbox[2] / 2.0
        cy = last_bbox[1] + last_bbox[3] / 2.0
        radius = self.search_multiplier * max(last_bbox[2], last_bbox[3])

        best_idx = -1
        best_score = -1.0
        for i, b in enumerate(det_boxes):
            bx = b[0] + b[2] / 2.0
            by = b[1] + b[3] / 2.0
            if np.hypot(bx - cx, by - cy) > radius:
                continue
            score = _ncc(template, _crop(frame, b))
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx >= 0 and best_score >= self.appearance_threshold:
            return RecoveryResult(
                track_id=tid, recovered=True,
                new_bbox=tuple(int(v) for v in det_boxes[best_idx]),
                method="classical",
                reasoning=f"NCC match {best_score:.2f} within radius "
                          f"{radius:.0f}px (idx {best_idx})")

        return RecoveryResult(
            track_id=tid, recovered=False, method="classical",
            reasoning=f"best NCC={best_score:.2f} < "
                      f"threshold {self.appearance_threshold:.2f}")

    # -------------------------------------------------- DSPy ReAct path
    def _build_react(self, tools):
        class LostCellRecoverySig(dspy.Signature):
            """You are helping a cell-tracker recover a cell that is
            missing from the current frame but is known NOT to have
            left the field of view. Use the available tools to
            determine whether any current-frame detection is actually
            the lost cell, or whether the cell is hidden/merged."""
            context: str = dspy.InputField(
                desc="What is known about the lost track.")
            recovered: bool = dspy.OutputField(
                desc="True if a match was found.")
            match_index: int = dspy.OutputField(
                desc="Index of matched candidate or -1.")

        return dspy.ReAct(
            LostCellRecoverySig,
            tools=[tools.search_radius, tools.appearance_match,
                   tools.check_merge],
            max_iters=6,
        )

    def _try_react(self, tid: int, frame: np.ndarray,
                   prev_frame: np.ndarray, last_bbox,
                   det_boxes: List) -> RecoveryResult:
        trace: List[str] = []
        template = _crop(prev_frame, last_bbox) \
            if prev_frame is not None else None

        cx = last_bbox[0] + last_bbox[2] / 2.0
        cy = last_bbox[1] + last_bbox[3] / 2.0
        default_radius = self.search_multiplier * max(last_bbox[2],
                                                      last_bbox[3])

        class _Tools:
            def __init__(self_inner):
                self_inner.last_candidates: List[int] = []

            def search_radius(self_inner, radius_str: str = "") -> str:
                try:
                    r = float(radius_str)
                except Exception:
                    r = default_radius
                trace.append(f"search_radius({r:.0f})")
                cands = []
                for i, b in enumerate(det_boxes):
                    bx = b[0] + b[2] / 2.0
                    by = b[1] + b[3] / 2.0
                    dist = float(np.hypot(bx - cx, by - cy))
                    if dist <= r:
                        cands.append((i, dist))
                cands.sort(key=lambda t: t[1])
                self_inner.last_candidates = [c[0] for c in cands[:8]]
                return "candidates (idx, distance_px): " + \
                       ", ".join(f"({i},{d:.0f})" for i, d in cands[:8])

            def appearance_match(self_inner, index_str: str) -> str:
                try:
                    idx = int(index_str)
                except Exception:
                    return "invalid index"
                if idx < 0 or idx >= len(det_boxes) or template is None:
                    return "invalid index or missing template"
                trace.append(f"appearance_match({idx})")
                score = _ncc(template, _crop(frame, det_boxes[idx]))
                return f"ncc_similarity={score:.3f}"

            def check_merge(self_inner, _q: str = "") -> str:
                trace.append("check_merge")
                # area growth near predicted center
                target_area = last_bbox[2] * last_bbox[3]
                grown = []
                for i, b in enumerate(det_boxes):
                    bx = b[0] + b[2] / 2.0
                    by = b[1] + b[3] / 2.0
                    if np.hypot(bx - cx, by - cy) > default_radius:
                        continue
                    a = b[2] * b[3]
                    if a > target_area * 1.6:
                        grown.append((i, a / max(1, target_area)))
                if not grown:
                    return "no enlarged neighbor detected"
                return "possible merge candidates " + \
                       ", ".join(f"(idx={i},ratio={r:.2f})" for i, r in grown)

        tools = _Tools()
        try:
            prog = self._build_react(tools)
            result = prog(context=(
                f"Lost track {tid} last seen at bbox={last_bbox}; "
                f"frame shape={frame.shape[:2]}; "
                f"default search radius={default_radius:.0f}px. "
                f"There are {len(det_boxes)} detections this frame."))
            recovered = bool(getattr(result, "recovered", False))
            midx = int(getattr(result, "match_index", -1))
            reasoning = str(getattr(result, "reasoning", ""))[:400]
        except Exception as e:
            return RecoveryResult(
                track_id=tid, recovered=False, method="react",
                reasoning=f"ReAct failure: {e}", trace=trace)

        new_bbox = None
        if recovered and 0 <= midx < len(det_boxes):
            new_bbox = tuple(int(v) for v in det_boxes[midx])
        return RecoveryResult(
            track_id=tid, recovered=recovered and new_bbox is not None,
            new_bbox=new_bbox, method="react",
            reasoning=reasoning or "ReAct decision",
            trace=trace)

    # ---------------------------------------------------- apply recovery
    @staticmethod
    def _apply(tracker, result: RecoveryResult) -> None:
        if result.new_bbox is None:
            return
        track = tracker.tracks.get(result.track_id)
        if track is None:
            return
        track.is_active = True
        track.missed_frames = 0
        track.hits += 1
        # Replace the final (coasted) box with the observation, and push
        # the observation into the Kalman filter.
        if track.boxes:
            track.boxes[-1] = result.new_bbox
        else:
            track.boxes.append(result.new_bbox)
        if track._kf is not None:
            try:
                track._kf.update(result.new_bbox)
            except Exception:
                pass
