"""
CytoTrack AI — Visual-LLM Assistant
====================================
Optional helper that uses a vision-language model to:

  * verify that a tracked box really still contains the cell
    it was following (sanity-check across frames);
  * decide whether an ambiguous detection is a cell or debris;
  * provide a short textual explanation for the log.

Backends are supported and chosen automatically in order:

  1. vLLM  — local GPU inference with a multimodal model such as
             Qwen/Qwen2-VL-7B-Instruct.                              (fast)
  2. OpenAI/Azure OpenAI vision — cloud API when fully configured.   (cloud)
  3. Claude vision — Anthropic API (requires ANTHROPIC_API_KEY).     (cloud)
  4. Heuristic — shape/intensity rules.                              (always)

The classical heuristic backend means the wider CytoTrack pipeline
works even on machines without GPUs or internet.
"""

from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ------------------------------------------------------------------ backends
def _has_vllm() -> bool:
    try:
        import vllm  # noqa: F401
        return True
    except Exception:
        return False


def _has_anthropic() -> bool:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
        return True
    except Exception:
        return False


def _has_openai() -> bool:
    if not os.environ.get("OPENAI_API_KEY"):
        return False
    try:
        import openai  # noqa: F401
        return True
    except Exception:
        return False


def _has_azure_openai() -> bool:
    required = (
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_DEPLOYMENT",
    )
    if not all(os.environ.get(k) for k in required):
        return False
    try:
        import openai  # noqa: F401
        return True
    except Exception:
        return False


# -------------------------------------------------------------- data classes
@dataclass
class VerificationResult:
    is_cell: bool
    confidence: float          # 0..1
    label: str                 # "cell" / "debris" / "ambiguous"
    reasoning: str             # short explanation
    backend: str               # "vllm" / "openai" / "azure-openai" / "claude" / "heuristic"


# ----------------------------------------------------------------- helpers
def _crop_bbox(frame: np.ndarray, bbox) -> np.ndarray:
    x, y, w, h = [int(v) for v in bbox]
    pad = 6
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame.shape[1], x + w + pad)
    y2 = min(frame.shape[0], y + h + pad)
    return frame[y1:y2, x1:x2].copy()


def _encode_png_base64(image_bgr: np.ndarray) -> str:
    if not HAS_CV2:
        raise RuntimeError("cv2 is required to encode image")
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("failed to encode image")
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ================================================================= main API
class VisualLLMHelper:
    """
    Thin wrapper that routes calls to whichever visual-LLM backend is
    available. All public methods return simple Python dataclasses so
    the rest of the pipeline does not need to know which backend was
    used.
    """

    def __init__(self, prefer: str = "auto",
                 vllm_model: str = "Qwen/Qwen2-VL-7B-Instruct",
                 openai_model: str = "gpt-4o-mini",
                 claude_model: str = "claude-opus-4-7"):
        """
        prefer: "auto" | "vllm" | "openai" | "azure-openai" | "claude" | "heuristic"
        """
        self.vllm_model = vllm_model
        self.openai_model = os.environ.get("OPENAI_VISION_MODEL", openai_model)
        self.azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
        self.claude_model = claude_model
        self._vllm = None
        self._openai = None
        self._azure_openai = None
        self._anthropic = None
        self.backend = self._pick_backend(prefer)

    # -------------------------------------------------- backend selection
    def _pick_backend(self, prefer: str) -> str:
        if prefer == "heuristic":
            return "heuristic"
        if prefer == "vllm":
            return "vllm" if _has_vllm() else "heuristic"
        if prefer == "openai":
            return "openai" if _has_openai() else "heuristic"
        if prefer == "azure-openai":
            return "azure-openai" if _has_azure_openai() else "heuristic"
        if prefer == "claude":
            return "claude" if _has_anthropic() else "heuristic"
        # auto
        if _has_vllm():
            return "vllm"
        if _has_azure_openai():
            return "azure-openai"
        if _has_openai():
            return "openai"
        if _has_anthropic():
            return "claude"
        return "heuristic"

    # ------------------------------------------------------- public entry
    def verify_cell(self, frame: np.ndarray, bbox,
                    context: Optional[str] = None) -> VerificationResult:
        """Decide whether a detection is a real cell or debris."""
        crop = _crop_bbox(frame, bbox)
        if crop.size == 0:
            return VerificationResult(False, 0.0, "debris",
                                      "empty crop", self.backend)

        if self.backend == "vllm":
            try:
                return self._vllm_verify(crop, context)
            except Exception as e:
                return self._heuristic_verify(crop, reason=f"vllm-error: {e}")

        if self.backend == "openai":
            try:
                return self._openai_verify(crop, context)
            except Exception as e:
                return self._heuristic_verify(crop, reason=f"openai-error: {e}")

        if self.backend == "azure-openai":
            try:
                return self._azure_openai_verify(crop, context)
            except Exception as e:
                return self._heuristic_verify(crop, reason=f"azure-openai-error: {e}")

        if self.backend == "claude":
            try:
                return self._claude_verify(crop, context)
            except Exception as e:
                return self._heuristic_verify(crop, reason=f"claude-error: {e}")

        return self._heuristic_verify(crop)

    def follow_cell(self, frame: np.ndarray, prev_bbox,
                    candidate_bboxes: List) -> int:
        """
        Given the previous bbox and a list of candidate new-frame
        bboxes, return the index of the candidate that most likely
        contains the same cell. Falls back to nearest-center when no
        visual backend is active.
        """
        if not candidate_bboxes:
            return -1
        prev = _crop_bbox(frame, prev_bbox)
        crops = [_crop_bbox(frame, b) for b in candidate_bboxes]

        if self.backend == "vllm":
            try:
                return self._vllm_follow(prev, crops)
            except Exception:
                pass  # fall through to heuristic

        return self._heuristic_follow(prev, crops, candidate_bboxes, prev_bbox)

    def backend_status(self) -> dict:
        """Return configuration state without exposing secrets."""
        return {
            "selected_backend": self.backend,
            "vllm_available": _has_vllm(),
            "openai_available": _has_openai(),
            "azure_openai_available": _has_azure_openai(),
            "anthropic_available": _has_anthropic(),
            "azure_missing": [
                k for k in (
                    "AZURE_OPENAI_ENDPOINT",
                    "AZURE_OPENAI_API_VERSION",
                    "AZURE_OPENAI_DEPLOYMENT",
                )
                if not os.environ.get(k)
            ],
        }

    # ================================================ heuristic backend
    @staticmethod
    def _heuristic_verify(crop: np.ndarray,
                          reason: str = "classical heuristics") -> VerificationResult:
        if not HAS_CV2 or crop.size == 0:
            return VerificationResult(False, 0.5, "ambiguous", reason, "heuristic")

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        h, w = gray.shape
        if h < 4 or w < 4:
            return VerificationResult(False, 0.3, "debris",
                                      "too small", "heuristic")

        # --- features -----------------------------------------------------
        mean_int = float(gray.mean())
        std_int = float(gray.std())
        _, mask = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if mean_int > 128:
            mask = cv2.bitwise_not(mask)  # make cell foreground
        fg_fraction = mask.sum() / (255.0 * mask.size)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return VerificationResult(False, 0.4, "debris",
                                      "no foreground", "heuristic")
        cnt = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))
        perim = float(cv2.arcLength(cnt, True))
        circularity = (4 * np.pi * area / (perim * perim)) if perim > 0 else 0.0
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = max(bw, bh) / max(1, min(bw, bh))
        elongated = 1.6 <= aspect <= 8.0

        # --- rules --------------------------------------------------------
        score = 0.0
        score += 0.35 if circularity > 0.55 else (0.15 if circularity > 0.35 else 0.0)
        # Migrating melanoma / fibroblast-like cells are often spindle-shaped,
        # so low circularity alone must not make them debris.
        score += 0.25 if elongated else 0.0
        score += 0.25 if 0.12 < fg_fraction < 0.85 else 0.05
        score += 0.2 if std_int > 15 else 0.0
        score += 0.2 if area > 30 else 0.0

        score = float(np.clip(score, 0.0, 1.0))
        is_cell = score >= 0.55
        label = "cell" if is_cell else ("ambiguous" if score >= 0.35 else "debris")
        explanation = (f"circ={circularity:.2f}, aspect={aspect:.2f}, "
                       f"fg={fg_fraction:.2f}, std={std_int:.1f}, "
                       f"area={area:.0f} — {reason}")
        return VerificationResult(is_cell, score, label, explanation, "heuristic")

    @staticmethod
    def _heuristic_follow(prev: np.ndarray, crops, bboxes, prev_bbox) -> int:
        """Simple appearance + center-distance similarity."""
        px = prev_bbox[0] + prev_bbox[2] / 2.0
        py = prev_bbox[1] + prev_bbox[3] / 2.0
        best_idx = 0
        best_score = -np.inf
        for i, b in enumerate(bboxes):
            bx = b[0] + b[2] / 2.0
            by = b[1] + b[3] / 2.0
            dist = float(np.hypot(px - bx, py - by))
            # appearance: intensity mean delta
            delta = 0.0
            if HAS_CV2 and prev.size > 0 and crops[i].size > 0:
                try:
                    a = cv2.resize(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
                                   if prev.ndim == 3 else prev, (16, 16))
                    c = cv2.resize(cv2.cvtColor(crops[i], cv2.COLOR_BGR2GRAY)
                                   if crops[i].ndim == 3 else crops[i], (16, 16))
                    delta = float(np.abs(a.astype(float) - c.astype(float)).mean())
                except Exception:
                    delta = 0.0
            score = -dist - delta * 0.5
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    # ================================================= vLLM backend
    def _ensure_vllm(self):
        if self._vllm is not None:
            return
        from vllm import LLM, SamplingParams  # type: ignore
        self._vllm = {
            "llm": LLM(model=self.vllm_model, trust_remote_code=True,
                       dtype="auto"),
            "sp": SamplingParams(max_tokens=160, temperature=0.0),
        }

    def _vllm_verify(self, crop: np.ndarray,
                     context: Optional[str]) -> VerificationResult:
        self._ensure_vllm()
        prompt = (
            "You are a microscopy assistant. Look at the attached small "
            "image crop from a phase-contrast microscopy time-lapse. "
            "Decide whether it shows a live cell or is debris / "
            "artifact / background. Respond in strict JSON with fields "
            '{"label":"cell"|"debris"|"ambiguous","confidence":0..1,'
            '"reason":"short"}. '
            + (f"Context: {context}" if context else "")
        )
        b64 = _encode_png_base64(crop)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }]

        out = self._vllm["llm"].chat(messages, self._vllm["sp"])
        text = out[0].outputs[0].text.strip()
        return self._parse_json_response(text, backend="vllm")

    def _vllm_follow(self, prev: np.ndarray, crops) -> int:
        self._ensure_vllm()
        prompt = (
            "Image 1 is a reference crop of a cell from frame N. "
            "The remaining images are candidate crops from frame N+1. "
            "Return strict JSON {\"index\": <0-based index of the "
            "candidate that best matches the reference>}."
        )
        content = [{"type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,"
                                           f"{_encode_png_base64(prev)}"}}]
        for c in crops:
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,"
                                                   f"{_encode_png_base64(c)}"}})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]
        out = self._vllm["llm"].chat(messages, self._vllm["sp"])
        text = out[0].outputs[0].text.strip()
        try:
            data = json.loads(self._extract_json_blob(text))
            return int(data.get("index", 0))
        except Exception:
            return 0

    # ================================================= OpenAI backends
    def _openai_prompt(self, context: Optional[str]) -> str:
        return (
            "You are a microscopy visual verifier. Inspect this crop from a "
            "phase-contrast/DIC time-lapse of WM239A melanoma migration. "
            "Classify whether the crop contains one real cell body center "
            "rather than chamber wall, timestamp/scale bar, debris, or "
            "background. Respond only as JSON with fields "
            '{"label":"cell"|"debris"|"ambiguous","confidence":0..1,'
            '"reason":"short"}. '
            + (f"Context: {context}" if context else "")
        )

    def _ensure_openai(self):
        if self._openai is not None:
            return
        from openai import OpenAI  # type: ignore
        self._openai = OpenAI()

    def _ensure_azure_openai(self):
        if self._azure_openai is not None:
            return
        from openai import AzureOpenAI  # type: ignore
        self._azure_openai = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )

    def _openai_verify(self, crop: np.ndarray,
                       context: Optional[str]) -> VerificationResult:
        self._ensure_openai()
        b64 = _encode_png_base64(crop)
        resp = self._openai.chat.completions.create(
            model=self.openai_model,
            temperature=0,
            max_tokens=180,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": self._openai_prompt(context)},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
        )
        text = resp.choices[0].message.content or ""
        return self._parse_json_response(text, backend="openai")

    def _azure_openai_verify(self, crop: np.ndarray,
                             context: Optional[str]) -> VerificationResult:
        self._ensure_azure_openai()
        b64 = _encode_png_base64(crop)
        resp = self._azure_openai.chat.completions.create(
            model=self.azure_deployment,
            temperature=0,
            max_tokens=180,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": self._openai_prompt(context)},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
        )
        text = resp.choices[0].message.content or ""
        return self._parse_json_response(text, backend="azure-openai")

    # ================================================= Claude backend
    def _ensure_anthropic(self):
        if self._anthropic is not None:
            return
        import anthropic  # type: ignore
        self._anthropic = anthropic.Anthropic()

    def _claude_verify(self, crop: np.ndarray,
                       context: Optional[str]) -> VerificationResult:
        self._ensure_anthropic()
        b64 = _encode_png_base64(crop)
        prompt = (
            "You are a microscopy assistant. Look at this small crop "
            "from a microscopy time-lapse and decide whether it is a "
            "live cell or debris/artifact. Respond in strict JSON: "
            '{"label":"cell"|"debris"|"ambiguous","confidence":0..1,'
            '"reason":"short"}. '
            + (f"Context: {context}" if context else "")
        )
        resp = self._anthropic.messages.create(
            model=self.claude_model,
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": "image/png",
                                "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        text = "".join(part.text for part in resp.content
                       if getattr(part, "type", None) == "text")
        return self._parse_json_response(text, backend="claude")

    # ================================================= JSON helpers
    @staticmethod
    def _extract_json_blob(text: str) -> str:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("no JSON object in response")
        return text[start:end + 1]

    def _parse_json_response(self, text: str, backend: str) -> VerificationResult:
        try:
            data = json.loads(self._extract_json_blob(text))
            label = str(data.get("label", "ambiguous")).lower()
            confidence = float(data.get("confidence", 0.5))
            reason = str(data.get("reason", ""))[:200]
            is_cell = (label == "cell")
            return VerificationResult(is_cell, confidence, label,
                                      reason, backend)
        except Exception:
            return VerificationResult(False, 0.5, "ambiguous",
                                      f"parse failure: {text[:80]!r}", backend)
