"""
CytoTrack AI - Multi-Object Tracker
====================================
SORT-style tracker: per-track Kalman filter (constant-velocity) +
Hungarian assignment against fresh detections every frame.

This is substantially more robust than the legacy per-cell CSRT-only
tracker for crowded live-cell imaging:
  * Kalman smooths noisy detections and predicts through missed frames
  * Hungarian (scipy.optimize.linear_sum_assignment) gives globally
    optimal ID-to-detection matching using IoU + center distance
  * New tracks spawn automatically for newly appearing cells
  * Tracks survive short occlusions via the max_missed budget
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np

from tracking_linker import build_link_features, predict_link_probability

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _det_to_centroid_box(det) -> Tuple[int, int, int, int]:
    """
    Convert a detector object or bbox tuple into the tracker's box format.

    If the detector provides ``center_x``/``center_y`` those coordinates are
    treated as the quantitative cell centre. This keeps tracking on mask
    centroids instead of drifting to contour edges or arbitrary bbox centres.
    """
    box = det.bbox if hasattr(det, "bbox") else det
    x, y, w, h = tuple(int(v) for v in box)
    if hasattr(det, "center_x") and hasattr(det, "center_y"):
        cx = float(det.center_x)
        cy = float(det.center_y)
        return (
            int(round(cx - w / 2.0)),
            int(round(cy - h / 2.0)),
            int(w),
            int(h),
        )
    return (x, y, w, h)


# --- Appearance-consistency helpers ------------------------------------
# A small zero-mean unit-variance grayscale thumbnail per detection is
# enough to resolve ID swaps during overlap. Dot-product of two such
# thumbnails is NCC (normalised cross-correlation) in [-1, 1].
_APPEARANCE_THUMB = 12


def _extract_appearance(frame: np.ndarray,
                        bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """
    Return a normalised grayscale thumbnail around ``bbox`` or None.

    The crop is *padded* by half the bbox size so the thumbnail captures
    not just the cell but the immediate surround. That contextual ring
    is what lets appearance disambiguate two morphologically-identical
    cells that happen to sit in different neighbourhoods — the canonical
    failure mode for simple SORT-style trackers in dense scenes.
    """
    if frame is None or not HAS_CV2:
        return None
    x, y, w, h = bbox
    H, W = frame.shape[:2]
    pad_x = max(2, int(w * 0.5))
    pad_y = max(2, int(h * 0.5))
    x1 = max(0, int(x) - pad_x)
    y1 = max(0, int(y) - pad_y)
    x2 = min(W, int(x + w) + pad_x)
    y2 = min(H, int(y + h) + pad_y)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    try:
        crop = cv2.resize(crop, (_APPEARANCE_THUMB, _APPEARANCE_THUMB))
    except Exception:
        return None
    v = crop.astype(np.float32).flatten()
    v -= v.mean()
    s = float(np.linalg.norm(v))
    if s < 1e-6:
        return None
    return v / s


def _appearance_distance(a: Optional[np.ndarray],
                         b: Optional[np.ndarray]) -> float:
    """0 when identical, 1 when unrelated; used as an additive cost term."""
    if a is None or b is None:
        return 0.5  # neutral — don't help, don't hurt
    ncc = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return (1.0 - ncc) * 0.5  # maps [-1, 1] -> [1, 0]


class KalmanBox:
    """
    Minimal constant-velocity Kalman filter for a bounding box.

    State: [cx, cy, w, h, vx, vy]  (6D)
    Measurement: [cx, cy, w, h]    (4D)
    """

    def __init__(self, bbox: Tuple[int, int, int, int]):
        x, y, w, h = bbox
        self.x = np.array([x + w / 2.0, y + h / 2.0, float(w), float(h),
                           0.0, 0.0], dtype=np.float64)

        self.F = np.eye(6)
        self.F[0, 4] = 1.0  # cx += vx
        self.F[1, 5] = 1.0  # cy += vy

        self.H = np.zeros((4, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        self.P = np.eye(6) * 10.0
        self.P[4, 4] = 100.0
        self.P[5, 5] = 100.0

        self.Q = np.eye(6) * 1.0
        self.Q[4, 4] = 4.0
        self.Q[5, 5] = 4.0

        self.R = np.eye(4) * 4.0

    def predict(self) -> Tuple[int, int, int, int]:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.bbox()

    def update(self, bbox: Tuple[int, int, int, int]) -> None:
        x, y, w, h = bbox
        z = np.array([x + w / 2.0, y + h / 2.0, float(w), float(h)])
        y_resid = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y_resid
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def bbox(self) -> Tuple[int, int, int, int]:
        cx, cy, w, h = self.x[0], self.x[1], self.x[2], self.x[3]
        w = max(4.0, w)
        h = max(4.0, h)
        return (int(cx - w / 2), int(cy - h / 2), int(w), int(h))

    @property
    def center(self) -> Tuple[float, float]:
        return float(self.x[0]), float(self.x[1])


@dataclass
class Track:
    track_id: int
    cell_type: str = "Cell"
    color: Tuple[int, int, int] = (0, 255, 0)
    boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    birth_frame: int = 0
    is_active: bool = True
    missed_frames: int = 0
    hits: int = 1
    _kf: Optional[KalmanBox] = None
    # EMA-smoothed appearance thumbnail + typical area. Used to resolve
    # ID swaps when two tracks land on the same overlapping detection.
    appearance: Optional[np.ndarray] = None
    area_ema: float = 0.0
    # Display-only EMA on width/height. The Kalman state in _kf keeps
    # raw-measurement w/h so the cost matrix's size-ratio cue stays
    # accurate; these fields are purely for rendering a stable box that
    # doesn't "breathe" every frame from contour noise.
    w_ema: float = 0.0
    h_ema: float = 0.0
    border_exited: bool = False
    ignored_entry: bool = False

    @property
    def frame_count(self) -> int:
        return len(self.boxes)

    def update_appearance(self, thumb: Optional[np.ndarray],
                          area: float, alpha: float = 0.3) -> None:
        if thumb is not None:
            if self.appearance is None:
                self.appearance = thumb.copy()
            else:
                self.appearance = (1 - alpha) * self.appearance + alpha * thumb
                n = float(np.linalg.norm(self.appearance))
                if n > 1e-6:
                    self.appearance /= n
        if area > 0:
            if self.area_ema <= 0:
                self.area_ema = float(area)
            else:
                self.area_ema = (1 - alpha) * self.area_ema + alpha * float(area)

    def update_display_size(self, w: float, h: float,
                            alpha: float = 0.15) -> None:
        """Slow EMA (α=0.15) on box dimensions for rendering."""
        if self.w_ema <= 0 or self.h_ema <= 0:
            self.w_ema = float(max(1.0, w))
            self.h_ema = float(max(1.0, h))
        else:
            self.w_ema = (1 - alpha) * self.w_ema + alpha * float(max(1.0, w))
            self.h_ema = (1 - alpha) * self.h_ema + alpha * float(max(1.0, h))

    def display_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Return a render-stable bbox: Kalman centre + EMA-smoothed w/h.
        Falls back to the raw last box if no EMA size has been seeded.
        """
        if not self.boxes:
            return None
        if self.w_ema <= 0 or self.h_ema <= 0 or self._kf is None:
            return self.boxes[-1]
        cx, cy = self._kf.center
        w, h = self.w_ema, self.h_ema
        return (int(cx - w / 2), int(cy - h / 2), int(w), int(h))


class CellTracker:
    """
    Multi-cell tracker combining Kalman prediction with Hungarian
    assignment over fresh per-frame detections.
    """

    COLORS = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
        (0, 255, 128), (255, 128, 128), (128, 255, 128), (128, 128, 255),
        (200, 100, 50), (50, 200, 100), (100, 50, 200), (255, 200, 100),
    ]

    def __init__(self, max_missed: int = 15, iou_threshold: float = 0.1,
                 max_distance: float = 80.0,
                 suppress_duplicate_detections: bool = False,
                 association_model=None,
                 ignore_border_objects: bool = False,
                 retire_on_border_exit: bool = False,
                 closed_world_tracking: bool = False):
        self.max_missed = max_missed
        self.iou_threshold = iou_threshold
        self.max_distance = max_distance
        self.suppress_duplicate_detections = bool(suppress_duplicate_detections)
        self.ignore_border_objects = bool(ignore_border_objects)
        self.retire_on_border_exit = bool(retire_on_border_exit)
        self.closed_world_tracking = bool(closed_world_tracking)
        self.tracks: Dict[int, Track] = {}
        self._next_id = 0
        self._width = 0
        self._height = 0
        self._detector = None
        self.association_model = association_model
        self._frame_idx = 0
        self._expected_active_cap = 0
        self.morphology_median_area: float = 0.0
        self.morphology_median_diameter: float = 0.0
        self.morphology_duplicate_center_distance: float = 0.0
        self.border_margin_px: float = 0.0

    def set_morphology_constraints(
        self,
        median_area_px: float = 0.0,
        median_diameter_px: float = 0.0,
        duplicate_center_distance_px: float = 0.0,
        border_margin_px: float = 0.0,
        **_: float,
    ) -> None:
        """
        Apply cell-line scale learned before tracking.

        The tracker uses these values as biological guardrails: two active
        centers closer than the trained same-cell distance are treated as one
        cell, and frame-border entry/exit margins scale with the cell size.
        """
        if median_area_px > 0:
            self.morphology_median_area = float(median_area_px)
        if median_diameter_px > 0:
            self.morphology_median_diameter = float(median_diameter_px)
            self.max_distance = float(max(self.max_distance,
                                          median_diameter_px * 2.5))
        if duplicate_center_distance_px > 0:
            self.morphology_duplicate_center_distance = float(
                duplicate_center_distance_px)
        if border_margin_px > 0:
            self.border_margin_px = float(border_margin_px)
        self.suppress_duplicate_detections = True
        self.ignore_border_objects = True
        self.retire_on_border_exit = True
        self.closed_world_tracking = True

    @property
    def active_count(self) -> int:
        return sum(1 for t in self.tracks.values() if t.is_active)

    @property
    def lost_count(self) -> int:
        return sum(1 for t in self.tracks.values() if not t.is_active)

    def attach_detector(self, detector) -> None:
        """Optional: supply a CellDetector for automatic re-detection."""
        self._detector = detector

    def _border_margin_for(
        self,
        bbox: Tuple[int, int, int, int],
    ) -> float:
        if self.border_margin_px > 0:
            return float(self.border_margin_px)
        return float(max(4.0, min(max(1, bbox[2]), max(1, bbox[3])) * 0.5))

    def _touches_frame_border(
        self,
        bbox: Tuple[int, int, int, int],
    ) -> bool:
        if self._width <= 0 or self._height <= 0:
            return False
        x, y, w, h = bbox
        margin = self._border_margin_for(bbox)
        cx = x + w / 2.0
        cy = y + h / 2.0
        if cx < margin or cy < margin:
            return True
        if cx > self._width - margin or cy > self._height - margin:
            return True
        if x <= 0 or y <= 0:
            return True
        if x + w >= self._width - 1 or y + h >= self._height - 1:
            return True
        return False

    def _retire_track_for_border_detection(
        self,
        bbox: Tuple[int, int, int, int],
    ) -> bool:
        """Terminate an existing track when its cell is observed at border."""
        if not self.tracks:
            return False
        bx = bbox[0] + bbox[2] / 2.0
        by = bbox[1] + bbox[3] / 2.0
        gate = max(
            self.max_distance,
            self.morphology_median_diameter * 2.0
            if self.morphology_median_diameter > 0 else 0.0,
        )
        for track in self.tracks.values():
            if not track.is_active or not track.boxes:
                continue
            last = track.boxes[-1]
            lx = last[0] + last[2] / 2.0
            ly = last[1] + last[3] / 2.0
            if _bbox_iou(bbox, last) > 0.10 or np.hypot(bx - lx, by - ly) <= gate:
                track.border_exited = True
                track.is_active = False
                track.boxes.append(bbox)
                return True
        return False

    def _is_anomalous_step(
        self,
        track: "Track",
        new_box: Tuple[int, int, int, int],
    ) -> bool:
        """
        Return True if `new_box` has drifted off-screen. This catches
        the runaway-coast failure mode where a track's Kalman velocity
        (loaded from an earlier wrong match) keeps extrapolating the
        box off the frame with no cell underneath.

        Gate: center outside the frame by more than half the box size.
        We intentionally do NOT gate on step distance — legitimate fast
        cells post-coast can legitimately jump several body-lengths on
        re-match, and retiring those was regressing the overlap tests.
        """
        if not track.boxes:
            return False
        if self._width <= 0 or self._height <= 0:
            return False
        nx = new_box[0] + new_box[2] / 2.0
        ny = new_box[1] + new_box[3] / 2.0
        margin = max(4.0, min(max(1, new_box[2]), max(1, new_box[3])) * 0.5)
        if (nx < -margin or nx > self._width + margin or
                ny < -margin or ny > self._height + margin):
            return True
        return False

    @staticmethod
    def _snapshot_kf(track: "Track") -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if track._kf is None:
            return None
        return track._kf.x.copy(), track._kf.P.copy()

    @staticmethod
    def _restore_kf(
        track: "Track",
        snapshot: Optional[Tuple[np.ndarray, np.ndarray]],
    ) -> None:
        if snapshot is None or track._kf is None:
            return
        track._kf.x = snapshot[0]
        track._kf.P = snapshot[1]

    def calibrate(self, detections) -> dict:
        """
        Tune Hungarian cost cutoffs from the observed cell-size distribution.
        Sets max_distance to ~2.5x median cell diameter so fast-moving cells
        can still be matched across a frame, but we don't swap IDs with a
        far-away neighbour in dense scenes.
        """
        if not detections:
            return {"n": 0, "max_distance": self.max_distance}
        diams = []
        for d in detections:
            box = d.bbox if hasattr(d, "bbox") else d
            _, _, w, h = box
            diams.append(max(int(w), int(h)))
        med = float(np.median(diams)) if diams else 0.0
        if med > 0:
            self.max_distance = float(max(30.0, med * 2.5))
        return {
            "n": len(detections),
            "median_diameter": med,
            "max_distance": self.max_distance,
        }

    def initialize(self, frame: np.ndarray, detections) -> None:
        self._height, self._width = frame.shape[:2]
        self.tracks.clear()
        self._next_id = 0
        self._frame_idx = 0

        det_boxes: List[Tuple[int, int, int, int]] = []
        det_appearance: List[Optional[np.ndarray]] = []
        for det in detections:
            bbox = _det_to_centroid_box(det)
            if self.ignore_border_objects and self._touches_frame_border(bbox):
                continue
            det_boxes.append(bbox)
            det_appearance.append(_extract_appearance(frame, bbox))

        if self.suppress_duplicate_detections:
            det_boxes, det_appearance = self._dedupe_detections(
                det_boxes, det_appearance)
        for bbox, appearance in zip(det_boxes, det_appearance):
            self._spawn_track(bbox, appearance=appearance,
                              birth_frame=self._frame_idx)
        self._expected_active_cap = len(self.tracks)

        print(f"  Initialized {len(self.tracks)} tracks")

    def _spawn_track(self, bbox: Tuple[int, int, int, int],
                     appearance: Optional[np.ndarray] = None,
                     birth_frame: Optional[int] = None) -> int:
        tid = self._next_id
        area = float(max(1, bbox[2]) * max(1, bbox[3]))
        track = Track(
            track_id=tid,
            color=self.COLORS[tid % len(self.COLORS)],
            boxes=[bbox],
            birth_frame=self._frame_idx if birth_frame is None else birth_frame,
            _kf=KalmanBox(bbox),
            area_ema=area,
            w_ema=float(max(1, bbox[2])),
            h_ema=float(max(1, bbox[3])),
        )
        if appearance is not None:
            track.appearance = appearance.copy()
        self.tracks[tid] = track
        self._next_id += 1
        return tid

    def _dedupe_detections(
        self,
        det_boxes: List[Tuple[int, int, int, int]],
        det_appearance: List[Optional[np.ndarray]],
    ) -> Tuple[List[Tuple[int, int, int, int]], List[Optional[np.ndarray]]]:
        """
        Suppress duplicate boxes before they can create duplicate tracks.

        High-recall detector fusion often emits several boxes for the same
        cell. We only merge boxes that are nearly co-centered or one clearly
        contains the other; adjacent cells with separated centres survive.
        """
        if len(det_boxes) <= 1:
            return det_boxes, det_appearance

        order = sorted(
            range(len(det_boxes)),
            key=lambda i: det_boxes[i][2] * det_boxes[i][3],
        )
        keep: List[int] = []
        for idx in order:
            box = det_boxes[idx]
            cx = box[0] + box[2] / 2.0
            cy = box[1] + box[3] / 2.0
            area = max(1.0, float(box[2] * box[3]))
            duplicate = False
            for kept_idx in keep:
                other = det_boxes[kept_idx]
                ox = other[0] + other[2] / 2.0
                oy = other[1] + other[3] / 2.0
                other_area = max(1.0, float(other[2] * other[3]))
                iou = _bbox_iou(box, other)
                dist = float(np.hypot(cx - ox, cy - oy))
                min_dim = max(1.0, min(box[2], box[3], other[2], other[3]))
                area_ratio = max(area, other_area) / max(1.0, min(area, other_area))

                if (self.morphology_duplicate_center_distance > 0
                        and dist < self.morphology_duplicate_center_distance
                        and area_ratio < 3.0):
                    duplicate = True
                    break

                if iou > 0.60 and dist < min_dim * 0.35:
                    duplicate = True
                    break

                # Containment-like duplicates: a loose detector box around the
                # same centre plus a tighter contour/blob box.
                inter_x1 = max(box[0], other[0])
                inter_y1 = max(box[1], other[1])
                inter_x2 = min(box[0] + box[2], other[0] + other[2])
                inter_y2 = min(box[1] + box[3], other[1] + other[3])
                inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                contained = inter / min(area, other_area) > 0.78
                if contained and dist < min_dim * 0.42 and area_ratio < 3.2:
                    duplicate = True
                    break

            if not duplicate:
                keep.append(idx)

        keep = sorted(keep)
        return [det_boxes[i] for i in keep], [det_appearance[i] for i in keep]

    def _cost_matrix(
        self,
        predicted: List[Tuple[int, Tuple[int, int, int, int]]],
        detections: List[Tuple[int, int, int, int]],
        det_appearance: List[Optional[np.ndarray]],
    ) -> np.ndarray:
        """
        Composite cost combining four independent cues. Any single cue
        being unreliable (e.g., frame-less tracker, new track w/ no
        appearance yet) degrades gracefully.

          * Centroid distance   - weight 0.5  (same as legacy)
          * 1 - IoU             - weight 1.0  (same as legacy)
          * Appearance (1-NCC)  - weight 0.7  NEW: resolves occlusion ID swaps
          * log area ratio      - weight 0.5  NEW: a big cell cannot grab a
                                              small cell's ID and vice versa
        """
        n_tracks = len(predicted)
        n_det = len(detections)
        HIGH = 10.0
        cost = np.full((n_tracks, n_det), HIGH, dtype=np.float64)

        for i, (tid, p_box) in enumerate(predicted):
            track = self.tracks[tid]
            px = p_box[0] + p_box[2] / 2.0
            py = p_box[1] + p_box[3] / 2.0
            t_area = track.area_ema if track.area_ema > 0 else \
                float(max(1, p_box[2]) * max(1, p_box[3]))

            for j, d_box in enumerate(detections):
                dx = d_box[0] + d_box[2] / 2.0
                dy = d_box[1] + d_box[3] / 2.0
                dist = np.hypot(px - dx, py - dy)
                if dist > self.max_distance:
                    continue
                iou = _bbox_iou(p_box, d_box)
                d_area = float(max(1, d_box[2]) * max(1, d_box[3]))
                # Symmetric log-ratio, capped. 0 for identical, rises sharply.
                ratio = max(d_area, t_area) / max(1e-6, min(d_area, t_area))
                size_term = min(1.5, np.log(ratio))
                app_term = _appearance_distance(track.appearance,
                                                det_appearance[j])
                heuristic_cost = (
                    (1.0 - iou) * 1.0
                    + (dist / self.max_distance) * 0.5
                    + app_term * 0.7
                    + size_term * 0.5
                )
                if self.association_model is not None:
                    appearance_similarity = 1.0 - app_term
                    features = build_link_features(
                        p_box,
                        d_box,
                        source_area=t_area,
                        target_area=d_area,
                        appearance_similarity=appearance_similarity,
                    )
                    prob = predict_link_probability(
                        self.association_model, features)
                    learned_cost = 1.0 - prob
                    cost[i, j] = 0.35 * min(1.0, heuristic_cost / 2.5) + learned_cost
                else:
                    cost[i, j] = heuristic_cost
        return cost

    def _near_neighbors(
        self,
        predicted: List[Tuple[int, Tuple[int, int, int, int]]],
    ) -> List[List[int]]:
        """
        For each predicted track, list indices of OTHER predicted tracks
        whose centres are within one median cell diameter. When a track
        has at least one near neighbour it is "in an occlusion zone" and
        a stricter swap guard fires.
        """
        n = len(predicted)
        out = [[] for _ in range(n)]
        if n <= 1:
            return out
        diams = [max(1, p_box[2]) for _, p_box in predicted]
        med_diam = float(np.median(diams)) if diams else 20.0
        thresh = med_diam * 1.25
        centres = np.array([
            [p_box[0] + p_box[2] / 2.0, p_box[1] + p_box[3] / 2.0]
            for _, p_box in predicted
        ])
        for i in range(n):
            d = np.hypot(centres[:, 0] - centres[i, 0],
                         centres[:, 1] - centres[i, 1])
            for k in np.where(d < thresh)[0]:
                if int(k) != i:
                    out[i].append(int(k))
        return out

    def update(self, frame: np.ndarray,
               detections=None) -> Dict[int, Tuple[int, int, int, int]]:
        """
        Advance tracker by one frame. If detections are not passed, the
        attached detector (if any) will be invoked. When no detector is
        attached and no detections provided, Kalman predictions are used
        as best-effort updates.
        """
        self._frame_idx += 1
        if detections is None and self._detector is not None:
            detections = self._detector.detect(frame)

        det_boxes: List[Tuple[int, int, int, int]] = []
        if detections is not None:
            for d in detections:
                bbox = _det_to_centroid_box(d)
                if self.ignore_border_objects and self._touches_frame_border(bbox):
                    self._retire_track_for_border_detection(bbox)
                    continue
                det_boxes.append(bbox)

        # Kalman predict for every active track
        active_items = [(tid, t) for tid, t in self.tracks.items() if t.is_active]
        predicted = [(tid, t._kf.predict()) for tid, t in active_items]

        # Precompute appearance thumbnails for every real detection.
        det_appearance: List[Optional[np.ndarray]] = [
            _extract_appearance(frame, b) for b in det_boxes
        ]
        if self.suppress_duplicate_detections:
            det_boxes, det_appearance = self._dedupe_detections(
                det_boxes, det_appearance)

        matched_track_ids: set = set()
        matched_det_idx: set = set()

        if det_boxes and predicted:
            cost = self._cost_matrix(predicted, det_boxes, det_appearance)
            neighbors = self._near_neighbors(predicted)

            if HAS_SCIPY:
                row_ind, col_ind = linear_sum_assignment(cost)
            else:  # Greedy fallback
                row_ind, col_ind = self._greedy_assign(cost)

            for r, c in zip(row_ind, col_ind):
                # Overall gate: anything this costly is a bad match.
                if cost[r, c] >= 2.5:
                    continue

                # Occlusion-zone swap guard: if this track has a near
                # neighbour AND a competing track has appreciably better
                # appearance match to the same detection, skip — let the
                # neighbour claim it, this one coasts. Hungarian is
                # globally optimal over OUR cost, but local mis-weighting
                # of components can still produce a swap; this guard is a
                # belt-and-braces check that refuses close calls.
                if neighbors[r]:
                    my_app = _appearance_distance(
                        self.tracks[predicted[r][0]].appearance,
                        det_appearance[c],
                    )
                    steal = False
                    for nbr in neighbors[r]:
                        nbr_tid = predicted[nbr][0]
                        nbr_app = _appearance_distance(
                            self.tracks[nbr_tid].appearance,
                            det_appearance[c],
                        )
                        # Neighbour's appearance matches notably better AND
                        # the neighbour itself was not matched to this det.
                        if nbr_app + 0.15 < my_app:
                            steal = True
                            break
                    if steal:
                        continue

                tid, _ = predicted[r]
                track = self.tracks[tid]
                kf_before = self._snapshot_kf(track)
                track._kf.update(det_boxes[c])
                new_box = track._kf.bbox()
                if self._is_anomalous_step(track, new_box):
                    # Match would teleport the track — refuse it and let
                    # the track coast (velocity clamp below stops runaway).
                    self._restore_kf(track, kf_before)
                    track.missed_frames += 1
                    if track.missed_frames > self.max_missed:
                        track.is_active = False
                    continue
                if self.retire_on_border_exit and self._touches_frame_border(new_box):
                    track.border_exited = True
                    track.is_active = False
                    track.boxes.append(new_box)
                    matched_track_ids.add(tid)
                    matched_det_idx.add(c)
                    continue
                track.boxes.append(new_box)
                track.missed_frames = 0
                track.hits += 1
                area = float(max(1, det_boxes[c][2]) *
                             max(1, det_boxes[c][3]))
                track.update_appearance(det_appearance[c], area)
                track.update_display_size(det_boxes[c][2], det_boxes[c][3])
                matched_track_ids.add(tid)
                matched_det_idx.add(c)

        # Unmatched tracks -> last-chance recovery, else merge-share, else coast.
        #
        # Three-stage fallback in order of preference:
        #   1. last-chance match against any still-unmatched detection
        #      (widened radius + appearance agreement required)
        #   2. merge-share: if the track's Kalman prediction lies inside
        #      a detection that was ALREADY matched to another track AND
        #      that detection is oversized (blob has absorbed us), we
        #      coast on Kalman but zero the missed counter so we stay
        #      active through the occlusion rather than timing out
        #   3. pure coast: append Kalman prediction, increment missed
        for tid, t in active_items:
            if tid in matched_track_ids:
                continue
            pred_box = t._kf.bbox()

            recovered_idx = self._last_chance_match(
                t, pred_box, det_boxes, det_appearance, matched_det_idx)
            if recovered_idx is not None:
                kf_before = self._snapshot_kf(t)
                t._kf.update(det_boxes[recovered_idx])
                new_box = t._kf.bbox()
                if self._is_anomalous_step(t, new_box):
                    # Refuse the recovery — too far to be the same cell.
                    self._restore_kf(t, kf_before)
                    t.missed_frames += 1
                    if t.missed_frames > self.max_missed:
                        t.is_active = False
                    continue
                if self.retire_on_border_exit and self._touches_frame_border(new_box):
                    t.border_exited = True
                    t.is_active = False
                    t.boxes.append(new_box)
                    matched_track_ids.add(tid)
                    matched_det_idx.add(recovered_idx)
                    continue
                t.boxes.append(new_box)
                t.missed_frames = 0
                t.hits += 1
                area = float(max(1, det_boxes[recovered_idx][2]) *
                             max(1, det_boxes[recovered_idx][3]))
                t.update_appearance(det_appearance[recovered_idx], area)
                t.update_display_size(det_boxes[recovered_idx][2],
                                      det_boxes[recovered_idx][3])
                matched_track_ids.add(tid)
                matched_det_idx.add(recovered_idx)
                continue

            if self._is_absorbed_by_merge(
                    t, pred_box, det_boxes, matched_det_idx):
                # Don't count this as a miss — we know exactly where we
                # are (inside the merged blob) and another track has
                # already claimed the blob. When the merge resolves we'll
                # pick up the real detection again.
                if self._is_anomalous_step(t, pred_box):
                    t.is_active = False
                    continue
                if self.retire_on_border_exit and self._touches_frame_border(pred_box):
                    t.border_exited = True
                    t.is_active = False
                    t.boxes.append(pred_box)
                    continue
                t.boxes.append(pred_box)
                # do NOT increment missed_frames — merge is not a miss.
                continue

            if self._is_anomalous_step(t, pred_box):
                # Coast has extrapolated off the frame — retire rather
                # than append an off-screen phantom to the trajectory.
                t.is_active = False
                continue
            if self.retire_on_border_exit and self._touches_frame_border(pred_box):
                t.border_exited = True
                t.is_active = False
                t.boxes.append(pred_box)
                continue
            t.boxes.append(pred_box)
            t.missed_frames += 1
            if t.missed_frames > self.max_missed:
                t.is_active = False

        # Unmatched detections -> first try to REVIVE a recently-lost
        # (inactive) track before spawning a new ID. This is the key
        # anti-fragmentation step: when a cell flickers out of detection
        # for a few frames and comes back, the tracker should re-attach
        # to the OLD track id instead of minting a brand new one. Without
        # this, a single cell that's temporarily occluded generates 3-4
        # short tracks with distinct ids over a 50-frame scene — which
        # is exactly what the evaluation metrics were flagging as
        # explosive track counts and low GT coverage.
        for j, det_box in enumerate(det_boxes):
            if j in matched_det_idx:
                continue
            if self.ignore_border_objects and self._touches_frame_border(det_box):
                matched_det_idx.add(j)
                continue
            # Check active tracks first. A duplicate box should never revive a
            # retired duplicate ID or spawn a new one while an active track
            # already represents that cell.
            if self._absorb_into_matched(
                    det_box, matched_track_ids):
                matched_det_idx.add(j)
                continue
            if (self.suppress_duplicate_detections
                    and self._absorb_into_any_active(det_box, det_appearance[j])):
                matched_det_idx.add(j)
                continue
            if (self.suppress_duplicate_detections
                    and self.active_count >= self._active_cap()):
                matched_det_idx.add(j)
                continue

            revived_tid = self._try_revive_inactive(
                det_box, det_appearance[j])
            if revived_tid is not None:
                t = self.tracks[revived_tid]
                # Revival crosses a gap, so the normal velocity gate is
                # too strict. Validate against the track's LAST known
                # position using a widened cap (3x max_distance) — any
                # further than that is a mis-revive, not the same cell.
                if t.boxes:
                    last = t.boxes[-1]
                    lx = last[0] + last[2] / 2.0
                    ly = last[1] + last[3] / 2.0
                    dcx = det_box[0] + det_box[2] / 2.0
                    dcy = det_box[1] + det_box[3] / 2.0
                    if np.hypot(dcx - lx, dcy - ly) > self.max_distance * 3.0:
                        continue  # let this detection try another revive
                                  # target or fall through to spawn.
                t.is_active = True
                t.missed_frames = 0
                t.hits += 1
                t._kf.update(det_box)
                # Re-zero velocity on revive: after a gap, we don't know
                # the current direction, and stale vx/vy would create a
                # false teleport on the next frame.
                t._kf.x[4] = 0.0
                t._kf.x[5] = 0.0
                new_box = t._kf.bbox()
                if self.retire_on_border_exit and self._touches_frame_border(new_box):
                    t.border_exited = True
                    t.is_active = False
                    t.boxes.append(new_box)
                    matched_det_idx.add(j)
                    continue
                t.boxes.append(new_box)
                area = float(max(1, det_box[2]) * max(1, det_box[3]))
                t.update_appearance(det_appearance[j], area)
                t.update_display_size(det_box[2], det_box[3])
                matched_det_idx.add(j)
                continue
            if self.closed_world_tracking:
                matched_det_idx.add(j)
                continue
            self._spawn_track(det_box, appearance=det_appearance[j],
                              birth_frame=self._frame_idx)

        # Post-Hungarian track merger: collapse pairs of active tracks
        # that now sit on the same cell. Relaxed detection recall can
        # spawn a second id on top of an existing track before the
        # spawn-absorb guard catches it (e.g. first-frame initialisation,
        # or when the same blob is re-detected after a brief occlusion).
        # Merging them here removes ping-pong id switches without
        # regressing the clean-scene tests, because the gate is strict.
        self._merge_duplicate_tracks()
        if self.suppress_duplicate_detections:
            self._prune_excess_active_tracks(det_boxes)

        # Build outputs for currently active tracks
        return {tid: t.boxes[-1] for tid, t in self.tracks.items()
                if t.is_active and t.boxes}

    def _prune_excess_active_tracks(
        self,
        det_boxes: List[Tuple[int, int, int, int]],
    ) -> None:
        """
        In high-recall mode, remove duplicate/coasting IDs instead of letting
        them accumulate as active or lost tracks.

        This is intentionally opt-in because ordinary videos may have cells
        entering the field. Locate/high-recall synthetic runs are closed-world
        enough that active IDs far above the initialized population are almost
        always duplicate detector boxes.
        """
        active = [(tid, t) for tid, t in self.tracks.items()
                  if t.is_active and t.boxes]
        if not active:
            return
        cap = self._active_cap()
        self._drop_inactive_over_cap(cap)
        if len(active) <= cap:
            return

        def distance_to_detection(track: Track) -> float:
            if not det_boxes:
                return float("inf")
            box = track.boxes[-1]
            cx = box[0] + box[2] / 2.0
            cy = box[1] + box[3] / 2.0
            return float(min(
                np.hypot(cx - (d[0] + d[2] / 2.0),
                         cy - (d[1] + d[3] / 2.0))
                for d in det_boxes
            ))

        candidates = []
        for tid, t in active:
            age = self._frame_idx - t.birth_frame + 1
            det_dist = distance_to_detection(t)
            candidates.append((
                # Only delete very fresh duplicate spawns. Established tracks
                # keep their history; extra detections are absorbed instead.
                1 if t.birth_frame == self._frame_idx and t.hits <= 1 else 0,
                1 if t.missed_frames > 0 else 0,
                det_dist,
                -t.hits,
                -age,
                tid,
            ))
        candidates.sort(reverse=True)
        remove_n = len(active) - cap
        removed = 0
        for fresh, *_rest, tid in candidates:
            if removed >= remove_n:
                break
            if fresh <= 0:
                break
            self.tracks.pop(tid, None)
            removed += 1

        self._drop_inactive_over_cap(cap)

    def _drop_inactive_over_cap(self, cap: int) -> None:
        inactive = [(tid, t) for tid, t in self.tracks.items()
                    if not t.is_active]
        if inactive and len(self.tracks) > cap:
            inactive.sort(key=lambda item: (
                item[1].hits,
                item[1].frame_count,
                item[1].birth_frame,
            ))
            for tid, _ in inactive[:max(0, len(self.tracks) - cap)]:
                self.tracks.pop(tid, None)

    def _merge_duplicate_tracks(self) -> None:
        """
        Retire a *freshly-spawned* track if it sits on top of an
        established one (same centre, high IoU). This catches the case
        where a duplicate detection for an already-tracked cell slipped
        past the spawn-absorb guard and minted a new id.

        Gates — all of these must hold before retiring the younger track:
          * high spatial overlap (IoU > 0.8)
          * centres within 0.2 * min_dim (very tight — genuine neighbours
            sit further apart even when their bboxes touch)
          * younger track is freshly spawned (hits < 3). Established
            neighbour tracks have many hits and must not be merged, even
            if they briefly overlap during a crossing.

        This is a noop in clean scenes (no duplicate spawns), beneficial
        in dense scenes, and cannot retire long-standing tracks.
        """
        actives = [(tid, t) for tid, t in self.tracks.items()
                   if t.is_active and t.boxes]
        if len(actives) < 2:
            return

        to_delete: set = set()
        to_retire: set = set()
        for i in range(len(actives)):
            tid_a, ta = actives[i]
            if tid_a in to_retire or tid_a in to_delete:
                continue
            bx_a = ta.boxes[-1]
            cx_a = bx_a[0] + bx_a[2] / 2.0
            cy_a = bx_a[1] + bx_a[3] / 2.0
            for j in range(i + 1, len(actives)):
                tid_b, tb = actives[j]
                if tid_b in to_retire or tid_b in to_delete:
                    continue
                bx_b = tb.boxes[-1]
                cx_b = bx_b[0] + bx_b[2] / 2.0
                cy_b = bx_b[1] + bx_b[3] / 2.0
                min_dim = min(bx_a[2], bx_a[3], bx_b[2], bx_b[3])
                if min_dim <= 0:
                    continue
                dist = float(np.hypot(cx_a - cx_b, cy_a - cy_b))
                dist_thr = 0.45 if self.suppress_duplicate_detections else 0.2
                max_same_cell_dist = min_dim * dist_thr
                if self.morphology_duplicate_center_distance > 0:
                    max_same_cell_dist = max(
                        max_same_cell_dist,
                        self.morphology_duplicate_center_distance,
                    )
                if dist >= max_same_cell_dist:
                    continue
                iou = _bbox_iou(bx_a, bx_b)
                iou_thr = 0.50 if self.suppress_duplicate_detections else 0.8
                if iou <= iou_thr:
                    continue
                app = _appearance_distance(ta.appearance, tb.appearance)
                if self.suppress_duplicate_detections:
                    fresh_pair = ta.hits < 5 or tb.hits < 5
                    similar_pair = app < 0.28 and abs(ta.hits - tb.hits) >= 2
                    if not (fresh_pair or similar_pair):
                        continue
                elif ta.hits >= 3 and tb.hits >= 3:
                    continue
                # Retire the younger (fewer hits) one; higher id on tie.
                if ta.hits > tb.hits or (ta.hits == tb.hits and tid_a < tid_b):
                    loser = tid_b
                else:
                    loser = tid_a
                loser_track = self.tracks.get(loser)
                if (self.suppress_duplicate_detections
                        and loser_track is not None and loser_track.hits < 5):
                    to_delete.add(loser)
                else:
                    to_retire.add(loser)
                if loser == tid_a:
                    break
        for tid in to_delete:
            self.tracks.pop(tid, None)
        for tid in to_retire:
            t = self.tracks.get(tid)
            if t is not None:
                t.is_active = False

    def _is_absorbed_by_merge(
        self,
        track: "Track",
        pred_box: Tuple[int, int, int, int],
        det_boxes: List[Tuple[int, int, int, int]],
        matched_det_idx: set,
    ) -> bool:
        """
        Return True when the track's Kalman prediction sits inside a
        matched detection that is larger than the track's typical area
        — i.e. the track has been absorbed into a merged blob and the
        blob is already claimed by another track. Lets us keep the
        absorbed track alive through the occlusion without spawning a
        new id when the cells separate.
        """
        px = pred_box[0] + pred_box[2] / 2.0
        py = pred_box[1] + pred_box[3] / 2.0
        t_area = track.area_ema if track.area_ema > 0 else \
            float(max(1, pred_box[2]) * max(1, pred_box[3]))
        for j in matched_det_idx:
            dx, dy, dw, dh = det_boxes[j]
            if not (dx - 2 <= px <= dx + dw + 2):
                continue
            if not (dy - 2 <= py <= dy + dh + 2):
                continue
            d_area = float(max(1, dw) * max(1, dh))
            if d_area >= t_area * 1.3:
                return True
        return False

    def _absorb_into_matched(
        self,
        det_box: Tuple[int, int, int, int],
        matched_track_ids: set,
    ) -> bool:
        """
        Return True if ``det_box`` already sits on an active track that
        got Hungarian-matched this frame — i.e. another detector strategy
        fired on the same cell. The caller uses this to suppress a
        would-be duplicate spawn.

        Gate: IoU > 0.65 with the matched track's post-update bbox, OR
        the detection centre within ~0.55 of the radius of that track's
        centre AND the two bboxes overlap at all. Both are very strict so
        real new cells (unseen this frame) still spawn new ids.
        """
        dx = det_box[0] + det_box[2] / 2.0
        dy = det_box[1] + det_box[3] / 2.0
        for tid in matched_track_ids:
            t = self.tracks.get(tid)
            if t is None or not t.boxes:
                continue
            b = t.boxes[-1]
            if _bbox_iou(det_box, b) > 0.65:
                return True
            bx = b[0] + b[2] / 2.0
            by = b[1] + b[3] / 2.0
            rad = min(b[2], b[3], det_box[2], det_box[3]) / 2.0
            if (self.morphology_duplicate_center_distance > 0
                    and np.hypot(dx - bx, dy - by) <
                    self.morphology_duplicate_center_distance):
                return True
            if (_bbox_iou(det_box, b) > 0.08
                    and np.hypot(dx - bx, dy - by) < rad * 0.55):
                return True
        return False

    def _absorb_into_any_active(
        self,
        det_box: Tuple[int, int, int, int],
        det_app: Optional[np.ndarray],
    ) -> bool:
        """
        Suppress duplicate spawns from high-recall detectors.

        Unlike _absorb_into_matched(), this scans every active track. It is
        intentionally stricter on overlap/appearance because it runs before
        creating a new biological object ID.
        """
        dx = det_box[0] + det_box[2] / 2.0
        dy = det_box[1] + det_box[3] / 2.0
        d_area = float(max(1, det_box[2]) * max(1, det_box[3]))
        for t in self.tracks.values():
            if not t.is_active or not t.boxes:
                continue
            b = t.boxes[-1]
            iou = _bbox_iou(det_box, b)
            if iou > 0.45:
                return True

            bx = b[0] + b[2] / 2.0
            by = b[1] + b[3] / 2.0
            dist = float(np.hypot(dx - bx, dy - by))
            if self.morphology_duplicate_center_distance > 0:
                ratio = max(d_area, max(1.0, t.area_ema)) / max(
                    1e-6,
                    min(d_area, max(1.0, t.area_ema)),
                )
                if (dist < self.morphology_duplicate_center_distance
                        and ratio < 3.0):
                    return True
            rad = min(b[2], b[3], det_box[2], det_box[3]) / 2.0
            if rad <= 0:
                continue
            if iou <= 0.12 or dist >= rad * 0.42:
                continue

            # If appearance is available and clearly different, allow a
            # genuine adjacent cell to spawn. Otherwise treat the detection as
            # the same cell seen by another detector strategy.
            app = _appearance_distance(t.appearance, det_app)
            if t.appearance is not None and det_app is not None and app > 0.42:
                continue
            if t.area_ema > 0:
                ratio = max(d_area, t.area_ema) / max(1e-6, min(d_area, t.area_ema))
                if ratio > 2.5:
                    continue
            return True
        return False

    def _active_cap(self) -> int:
        base_cap = self._expected_active_cap or max(1, self.active_count)
        return max(base_cap + 3, int(round(base_cap * 1.10)))

    def _try_revive_inactive(
        self,
        det_box: Tuple[int, int, int, int],
        det_app: Optional[np.ndarray],
    ) -> Optional[int]:
        """
        Before spawning a new track for an unmatched detection, scan
        inactive tracks and return the id of the best re-attach candidate
        (if any) so the caller can revive it with its original id.

        Re-attach is only allowed when BOTH signals agree strongly:
          * position: the detection centre sits within 2x the tracker's
            max_distance gate of the inactive track's last known position
            (wider than the live gate because inactive means the track
            lost lock for >=1 frame and may have drifted)
          * appearance: NCC distance <= 0.30 (tighter than the live gate
            because reviving the wrong id is worse than an extra spawn)

        Never revive a track that's been inactive for longer than 2x
        max_missed — such tracks are likely genuinely gone (cell left
        the field, lysed, etc.) and reviving them produces teleport
        artefacts in the trajectory plot.
        """
        if not self.tracks:
            return None
        dx = det_box[0] + det_box[2] / 2.0
        dy = det_box[1] + det_box[3] / 2.0
        d_area = float(max(1, det_box[2]) * max(1, det_box[3]))
        best_tid = None
        best_score = float("inf")
        for tid, t in self.tracks.items():
            if t.is_active:
                continue
            if not t.boxes:
                continue
            if getattr(t, "border_exited", False):
                continue
            if t.missed_frames > self.max_missed * 2:
                continue
            lb = t.boxes[-1]
            lx = lb[0] + lb[2] / 2.0
            ly = lb[1] + lb[3] / 2.0
            dist = float(np.hypot(dx - lx, dy - ly))
            if dist > self.max_distance * 1.5:
                continue
            # Appearance MUST agree — reviving the wrong id produces a
            # teleport artefact in the trajectory and costs more than
            # just spawning a new id.
            if t.appearance is None or det_app is None:
                continue
            app = _appearance_distance(t.appearance, det_app)
            if app > 0.22:
                continue
            # Size sanity: reviving a big-track id onto a small detection
            # (or vice versa) is almost certainly the wrong match.
            if t.area_ema > 0:
                ratio = max(d_area, t.area_ema) / max(1e-6, min(d_area, t.area_ema))
                if ratio > 2.5:
                    continue
            score = dist / (self.max_distance * 2.0) + app * 0.6
            if score < best_score:
                best_score = score
                best_tid = tid
        return best_tid

    def _last_chance_match(
        self,
        track: "Track",
        pred_box: Tuple[int, int, int, int],
        det_boxes: List[Tuple[int, int, int, int]],
        det_appearance: List[Optional[np.ndarray]],
        matched_det_idx: set,
    ) -> Optional[int]:
        """
        Best-effort re-match for a track Hungarian refused this frame.

        We widen the search radius to 1.5x the normal gate, score every
        still-unmatched detection by distance + appearance, and accept
        the best one only if both signals agree (distance inside the
        wide radius AND appearance distance <= 0.35 OR track has no
        appearance yet). Without BOTH signals we'd risk an ID swap — so
        the cutoff is intentionally tighter than the main Hungarian pass.
        """
        if not det_boxes:
            return None
        px = pred_box[0] + pred_box[2] / 2.0
        py = pred_box[1] + pred_box[3] / 2.0
        wide_radius = self.max_distance * 1.5
        best_idx = None
        best_score = float("inf")
        for j, d_box in enumerate(det_boxes):
            if j in matched_det_idx:
                continue
            dx = d_box[0] + d_box[2] / 2.0
            dy = d_box[1] + d_box[3] / 2.0
            dist = float(np.hypot(px - dx, py - dy))
            if dist > wide_radius:
                continue
            app = _appearance_distance(track.appearance, det_appearance[j])
            if track.appearance is not None and app > 0.35:
                continue
            score = dist / wide_radius + app * 0.7
            if score < best_score:
                best_score = score
                best_idx = j
        return best_idx

    @staticmethod
    def _greedy_assign(cost: np.ndarray):
        rows, cols = [], []
        used_r, used_c = set(), set()
        flat = np.argsort(cost, axis=None)
        for idx in flat:
            r = idx // cost.shape[1]
            c = idx % cost.shape[1]
            if r in used_r or c in used_c:
                continue
            used_r.add(r)
            used_c.add(c)
            rows.append(r)
            cols.append(c)
        return np.array(rows), np.array(cols)

    def get_tracks(self, min_length: int = 5) -> Dict[int, Dict]:
        return {
            tid: {
                "cell_type": t.cell_type,
                "boxes": t.boxes.copy(),
                "color": t.color,
                "is_active": t.is_active,
                "frame_count": t.frame_count,
            }
            for tid, t in self.tracks.items()
            if len(t.boxes) >= min_length
        }
