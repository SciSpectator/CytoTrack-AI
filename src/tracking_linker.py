"""
Learned cell-linking features for frame-to-frame tracking.

The tracker can use these features to replace a purely hand-tuned association
cost with a model trained on public CTC TRA masks. The model predicts whether
an object in frame t and an object in frame t+1 are the same biological cell.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - optional in import-only contexts
    cv2 = None


Box = Tuple[float, float, float, float]

FEATURE_NAMES = [
    "abs_dx",
    "abs_dy",
    "distance",
    "normalized_distance",
    "bbox_iou",
    "abs_log_area_ratio",
    "abs_log_width_ratio",
    "abs_log_height_ratio",
    "appearance_similarity",
    "source_area_log",
    "target_area_log",
]


def center(box: Sequence[float]) -> Tuple[float, float]:
    x, y, w, h = [float(v) for v in box]
    return x + w / 2.0, y + h / 2.0


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def _safe_log_ratio(a: float, b: float) -> float:
    a = max(1e-6, float(a))
    b = max(1e-6, float(b))
    return abs(float(np.log(a / b)))


def build_link_features(
    source_box: Sequence[float],
    target_box: Sequence[float],
    source_area: float | None = None,
    target_area: float | None = None,
    appearance_similarity: float = 0.5,
) -> np.ndarray:
    sx, sy = center(source_box)
    tx, ty = center(target_box)
    sw, sh = max(1.0, float(source_box[2])), max(1.0, float(source_box[3]))
    tw, th = max(1.0, float(target_box[2])), max(1.0, float(target_box[3]))
    sa = float(source_area) if source_area is not None else sw * sh
    ta = float(target_area) if target_area is not None else tw * th
    dx = tx - sx
    dy = ty - sy
    distance = float(np.hypot(dx, dy))
    norm = max(1.0, np.sqrt(max(1.0, sa)))
    return np.asarray([
        abs(dx),
        abs(dy),
        distance,
        distance / norm,
        bbox_iou(source_box, target_box),
        _safe_log_ratio(sa, ta),
        _safe_log_ratio(sw, tw),
        _safe_log_ratio(sh, th),
        float(np.clip(appearance_similarity, 0.0, 1.0)),
        float(np.log1p(max(1.0, sa))),
        float(np.log1p(max(1.0, ta))),
    ], dtype=np.float32)


def predict_link_probability(model, features: Iterable[float]) -> float:
    x = np.asarray(list(features), dtype=np.float32).reshape(1, -1)
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(x)[0, 1])
    if hasattr(model, "decision_function"):
        score = float(model.decision_function(x)[0])
        return float(1.0 / (1.0 + np.exp(-score)))
    return float(model.predict(x)[0])


def crop_appearance(image: np.ndarray, box: Sequence[float],
                    size: int = 16) -> np.ndarray | None:
    if image is None or cv2 is None:
        return None
    x, y, w, h = [int(round(float(v))) for v in box]
    H, W = image.shape[:2]
    pad_x = max(2, int(w * 0.5))
    pad_y = max(2, int(h * 0.5))
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2, y2 = min(W, x + w + pad_x), min(H, y + h + pad_y)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    crop = image[y1:y2, x1:x2]
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    vec = crop.astype(np.float32).reshape(-1)
    vec -= float(vec.mean())
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return None
    return vec / norm


def appearance_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.5
    return float((np.clip(np.dot(a, b), -1.0, 1.0) + 1.0) * 0.5)
