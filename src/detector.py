"""
CytoTrack AI - Cell Detector
=============================
Detects cells on microscopy images. Combines several complementary
strategies and fuses the results via IoU-based non-maximum suppression.

Strategies
----------
1. Adaptive thresholding + morphology (handles uneven illumination).
2. Otsu global threshold.
3. Distance-transform watershed (splits touching cells).
4. Laplacian-of-Gaussian blob detector (scale-aware point detection).
5. Hough circles (backup for round cells).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    center_x: float
    center_y: float
    area: float
    confidence: float = 1.0

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


def _contour_to_det(cnt, min_area: float, max_area: float,
                    confidence: float = 1.0,
                    max_aspect: float = 4.0) -> Optional[Detection]:
    area = cv2.contourArea(cnt)
    if area < min_area or area > max_area:
        return None
    x, y, w, h = cv2.boundingRect(cnt)
    if min(w, h) < 1:
        return None
    if max(w, h) / max(1, min(w, h)) > max_aspect:
        return None
    M = cv2.moments(cnt)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx, cy = x + w / 2.0, y + h / 2.0
    return Detection(x=int(x), y=int(y), w=int(w), h=int(h),
                     center_x=float(cx), center_y=float(cy),
                     area=float(area), confidence=confidence)


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


class CellDetector:
    """
    Multi-strategy cell detector. Works on both dark-on-bright and
    bright-on-dark imagery (auto-detects polarity by comparing mean
    intensity to the median).
    """

    # Sensitivity presets. "normal" is the default the regression tests
    # lock in. "high" is for dense microscopy fields where the user sees
    # more cells in the image than the tracker picks up — it relaxes
    # duplicate-suppression so adjacent cells both survive. "max" is
    # aggressive: every plausible blob gets a detection, to be paired
    # with the tracker's spawn-absorb guard.
    SENSITIVITY_PRESETS = {
        "low":    {"nms_iou_thr": 0.2, "nms_center_frac": 0.85,
                   "min_area_factor": 0.30},
        "normal": {"nms_iou_thr": 0.3, "nms_center_frac": 0.75,
                   "min_area_factor": 0.25},
        "high":   {"nms_iou_thr": 0.45, "nms_center_frac": 0.55,
                   "min_area_factor": 0.18},
        "max":    {"nms_iou_thr": 0.6, "nms_center_frac": 0.4,
                   "min_area_factor": 0.12},
        # Deep-learning backend. Uses Cellpose-SAM (2024-25 SOTA ViT
        # segmenter trained on microscopy). Bypasses the classical
        # pipeline entirely; the NMS params below only apply if Cellpose
        # fails to load and we fall back.
        "ai":     {"nms_iou_thr": 0.45, "nms_center_frac": 0.55,
                   "min_area_factor": 0.18},
    }

    def __init__(self, min_area: int = 50, max_area: int = 10000,
                 expected_max_diameter: int = 60,
                 use_blob_detector: bool = True,
                 use_hough_circles: bool = True,
                 sensitivity: str = "normal"):
        self.min_area = min_area
        self.max_area = max_area
        self.expected_max_diameter = expected_max_diameter
        # Adaptive state (set by calibrate()).
        self.median_area: float = 0.0
        self.median_diameter: float = 0.0
        self.merge_area_factor: float = 1.8  # contour this much larger => split
        self._calibrated: bool = False
        # Latency-only toggles (do NOT alter accuracy of the core
        # adaptive-threshold / Otsu / watershed strategies).
        self.use_blob_detector = bool(use_blob_detector)
        self.use_hough_circles = bool(use_hough_circles)
        # Sensitivity preset — tunes NMS and min-area for dense scenes.
        if sensitivity not in self.SENSITIVITY_PRESETS:
            raise ValueError(
                f"unknown sensitivity '{sensitivity}', choose from "
                f"{list(self.SENSITIVITY_PRESETS)}")
        self.sensitivity = sensitivity
        preset = self.SENSITIVITY_PRESETS[sensitivity]
        self._nms_iou_thr: float = preset["nms_iou_thr"]
        self._nms_center_frac: float = preset["nms_center_frac"]
        self._min_area_factor: float = preset["min_area_factor"]
        # Lazy-loaded Cellpose-SAM model (only when sensitivity="ai").
        self._ai_model = None
        self._ai_failed = False

    # ---------------------------------------------------------- public API
    def calibrate(self, image: np.ndarray) -> dict:
        """
        Analyse the first frame to learn typical cell size and tune
        adaptive parameters:
          * expected_max_diameter -> 1.4x median detected diameter
          * min_area / max_area   -> 0.25x / 4x median area
        Also populates median stats so the detector can split merged
        cells whose contour is larger than `merge_area_factor * median`.
        """
        raw = self.detect(image)
        if not raw:
            self._calibrated = True
            return {"n": 0, "median_diameter": 0.0, "median_area": 0.0}

        areas = np.array([d.area for d in raw], dtype=np.float64)
        diams = np.array([max(d.w, d.h) for d in raw], dtype=np.float64)
        med_a = float(np.median(areas))
        med_d = float(np.median(diams))

        self.median_area = med_a
        self.median_diameter = med_d
        # Tune size bounds around the measured median.
        self.expected_max_diameter = int(max(20, med_d * 1.4))
        self.min_area = int(max(8, med_a * self._min_area_factor))
        self.max_area = int(max(self.min_area * 4, med_a * 4.0))
        self._calibrated = True
        return {
            "n": len(raw),
            "median_area": med_a,
            "median_diameter": med_d,
            "min_area": self.min_area,
            "max_area": self.max_area,
            "expected_max_diameter": self.expected_max_diameter,
        }

    def detect(self, image: np.ndarray) -> List[Detection]:
        if image is None:
            return []
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if self.sensitivity == "ai" and not self._ai_failed:
            ai_dets = self._detect_ai(gray)
            if ai_dets is not None:
                return ai_dets
            # Fell through — log once and never retry this session.
            self._ai_failed = True

        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Auto polarity: bright cells on dark background vs dark on bright.
        bright_on_dark = gray.mean() < 128

        # CLAHE for robust contrast handling
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        signal = enhanced if bright_on_dark else (255 - enhanced)

        all_detections: List[Detection] = []
        all_detections.extend(self._adaptive_threshold(signal))
        all_detections.extend(self._otsu(signal))
        all_detections.extend(self._watershed(signal))
        if self.use_blob_detector:
            all_detections.extend(self._blob(signal))
        if self.use_hough_circles:
            all_detections.extend(self._hough(signal))

        # Split merged/connected cells whose area is an outlier.
        if self._calibrated and self.median_area > 0:
            all_detections = self._split_merged(signal, all_detections)

        return self._nms(all_detections)

    # --------------------------------------------------------- strategies
    def _adaptive_threshold(self, gray: np.ndarray) -> List[Detection]:
        detections: List[Detection] = []
        # odd block size scaled to expected cell size
        block = max(11, (self.expected_max_diameter * 2) | 1)
        binary = cv2.adaptiveThreshold(gray, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, block, -5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            det = _contour_to_det(cnt, self.min_area, self.max_area, 0.9)
            if det is not None:
                detections.append(det)
        return detections

    def _otsu(self, gray: np.ndarray) -> List[Detection]:
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        out: List[Detection] = []
        for cnt in contours:
            det = _contour_to_det(cnt, self.min_area, self.max_area, 0.85)
            if det is not None:
                out.append(det)
        return out

    def _watershed(self, gray: np.ndarray) -> List[Detection]:
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        if dist.max() == 0:
            return []
        _, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)

        detections: List[Detection] = []
        for i in range(1, markers.max() + 1):
            mask = np.uint8(markers == i) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                det = _contour_to_det(cnt, self.min_area, self.max_area, 0.95)
                if det is not None:
                    detections.append(det)
        return detections

    def _blob(self, gray: np.ndarray) -> List[Detection]:
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = 255
        params.filterByArea = True
        params.minArea = self.min_area
        params.maxArea = self.max_area
        params.filterByCircularity = True
        params.minCircularity = 0.4
        params.filterByConvexity = True
        params.minConvexity = 0.7
        params.filterByInertia = False
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray)

        detections: List[Detection] = []
        for kp in keypoints:
            r = max(3, int(kp.size / 2))
            x, y = int(kp.pt[0]), int(kp.pt[1])
            detections.append(Detection(
                x=max(0, x - r), y=max(0, y - r),
                w=2 * r, h=2 * r,
                center_x=float(x), center_y=float(y),
                area=float(np.pi * r * r), confidence=0.75,
            ))
        return detections

    def _hough(self, gray: np.ndarray) -> List[Detection]:
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=15,
            param1=80, param2=22,
            minRadius=5,
            maxRadius=max(6, self.expected_max_diameter),
        )
        detections: List[Detection] = []
        if circles is None:
            return detections
        for c in circles[0]:
            x, y, r = int(c[0]), int(c[1]), max(3, int(c[2]))
            area = float(np.pi * r * r)
            if area < self.min_area or area > self.max_area:
                continue
            detections.append(Detection(
                x=max(0, x - r), y=max(0, y - r),
                w=2 * r, h=2 * r,
                center_x=float(x), center_y=float(y),
                area=area, confidence=0.7,
            ))
        return detections

    # ------------------------------------------------- merged-cell splitter
    def _split_merged(self, gray: np.ndarray,
                      dets: List[Detection]) -> List[Detection]:
        """
        For any detection whose area is >= merge_area_factor * median_area,
        run a local distance-transform watershed to split it into its
        constituent cells. If splitting fails, keep the original.
        """
        if not dets or self.median_area <= 0:
            return dets

        threshold = self.merge_area_factor * self.median_area
        out: List[Detection] = []
        H, W = gray.shape[:2]

        for d in dets:
            if d.area < threshold:
                out.append(d)
                continue

            # ROI with small padding
            pad = 4
            x0 = max(0, d.x - pad); y0 = max(0, d.y - pad)
            x1 = min(W, d.x + d.w + pad); y1 = min(H, d.y + d.h + pad)
            roi = gray[y0:y1, x0:x1]
            if roi.size == 0:
                out.append(d); continue

            _, bw = cv2.threshold(roi, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
            dist = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
            if dist.max() <= 0:
                out.append(d); continue

            # Iteratively lower the peak threshold until we find >=2 seeds
            n_labels = 0
            markers = None
            for frac in (0.55, 0.45, 0.35, 0.25):
                _, peaks = cv2.threshold(dist, frac * dist.max(), 255, 0)
                peaks = np.uint8(peaks)
                n_labels, lbl = cv2.connectedComponents(peaks)
                if n_labels - 1 >= 2:
                    markers = lbl
                    break
            if markers is None or n_labels - 1 < 2:
                out.append(d); continue

            # Watershed on the ROI to grow the seeds to cell territories
            markers_ws = markers.copy() + 1
            markers_ws[bw == 0] = 0
            roi_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            markers_ws = cv2.watershed(roi_bgr, markers_ws)

            splits: List[Detection] = []
            for i in range(2, markers_ws.max() + 1):
                mask = np.uint8(markers_ws == i) * 255
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                for cnt in cnts:
                    sub = _contour_to_det(cnt, self.min_area,
                                          self.max_area, 0.9)
                    if sub is None:
                        continue
                    # Shift ROI-local coords back to full-image coords
                    splits.append(Detection(
                        x=sub.x + x0, y=sub.y + y0,
                        w=sub.w, h=sub.h,
                        center_x=sub.center_x + x0,
                        center_y=sub.center_y + y0,
                        area=sub.area, confidence=0.95,
                    ))

            if len(splits) >= 2:
                out.extend(splits)
            else:
                out.append(d)
        return out

    # --------------------------------------------------- Cellpose-SAM (AI)
    def _load_ai_model(self) -> bool:
        """
        Lazy-load Cellpose-SAM (the 2024-25 ViT-based cell segmenter).
        Returns True if the model is ready, False to signal fallback.
        """
        if self._ai_model is not None:
            return True
        try:
            from cellpose.models import CellposeModel  # type: ignore
            import torch  # type: ignore
            use_gpu = bool(torch.cuda.is_available())
            # pretrained_model='cpsam' is the default in cellpose>=4 and
            # corresponds to the Cellpose-SAM checkpoint.
            self._ai_model = CellposeModel(gpu=use_gpu)
            print(f"[detector] Cellpose-SAM loaded (gpu={use_gpu})")
            return True
        except Exception as e:  # pragma: no cover - depends on env
            print(f"[detector] Cellpose load failed ({e}); "
                  f"falling back to classical pipeline")
            self._ai_model = None
            return False

    def _detect_ai(self, gray: np.ndarray) -> Optional[List[Detection]]:
        """
        Run Cellpose-SAM on `gray`. Returns a list of Detections, or
        None if the backend is unavailable (so caller can fall back).
        """
        if not self._load_ai_model():
            return None
        try:
            # Cellpose-SAM auto-estimates diameter when diameter=None.
            # Pass the calibrated median if we have one for a small speedup.
            diam = self.median_diameter if self.median_diameter > 0 else None
            masks, _, _ = self._ai_model.eval(
                gray,
                diameter=diam,
                min_size=max(15, int(self.min_area * 0.6)),
            )
        except Exception as e:  # pragma: no cover - depends on env
            print(f"[detector] Cellpose eval error ({e}); falling back")
            return None

        if masks is None:
            return []

        detections: List[Detection] = []
        labels = np.unique(masks)
        for lbl in labels:
            if lbl == 0:
                continue
            mask = (masks == lbl).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)
            area = float(cv2.contourArea(cnt))
            if area < max(5.0, self.min_area * 0.3):
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if min(w, h) < 1:
                continue
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx, cy = x + w / 2.0, y + h / 2.0
            detections.append(Detection(
                x=int(x), y=int(y), w=int(w), h=int(h),
                center_x=float(cx), center_y=float(cy),
                area=area, confidence=0.99,
            ))
        return detections

    # ----------------------------------------- per-cell feature extraction
    # Feature dimensionality of Cellpose-SAM's `styles` bottleneck.
    CELLPOSE_STYLE_DIM = 256

    def extract_cell_features(
        self,
        image: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
    ) -> Optional[List[np.ndarray]]:
        """
        Return a 256-dim Cellpose-SAM feature vector per bbox, or None
        if the AI backend is unavailable. Each vector is Cellpose's
        `styles` output — the ViT encoder's bottleneck activation —
        computed on a padded crop of that cell. Used as input to
        `classifier.FeatureClassifier` for AI-native phenotype training.
        """
        if not bboxes:
            return []
        if not self._load_ai_model():
            return None
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        H, W = gray.shape[:2]
        crops: List[np.ndarray] = []
        for (x, y, w, h) in bboxes:
            pad = max(4, min(w, h) // 3)
            x0 = max(0, x - pad); y0 = max(0, y - pad)
            x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
            crop = gray[y0:y1, x0:x1]
            if crop.size == 0:
                # 4x4 placeholder so the batch shape stays consistent —
                # corresponding feature will be garbage, caller should
                # discard.
                crop = np.zeros((4, 4), dtype=np.uint8)
            crops.append(crop)
        try:
            diam = self.median_diameter if self.median_diameter > 0 else None
            _, _, styles_list = self._ai_model.eval(
                crops, diameter=diam, min_size=-1,
            )
        except Exception as e:  # pragma: no cover
            print(f"[detector] Cellpose feature extract failed ({e})")
            return None
        return [np.asarray(s, dtype=np.float32) for s in styles_list]

    # --------------------------------------------------------- NMS helper
    def _nms(self, dets: List[Detection],
             iou_thr: Optional[float] = None) -> List[Detection]:
        if not dets:
            return []
        thr = self._nms_iou_thr if iou_thr is None else iou_thr
        center_frac = self._nms_center_frac
        ordered = sorted(dets, key=lambda d: (d.confidence, d.area), reverse=True)
        kept: List[Detection] = []
        for d in ordered:
            ok = True
            for k in kept:
                if _bbox_iou(d.bbox, k.bbox) > thr:
                    ok = False
                    break
                dx = d.center_x - k.center_x
                dy = d.center_y - k.center_y
                min_r = min(d.w, d.h, k.w, k.h) / 2.0
                if np.hypot(dx, dy) < min_r * center_frac:
                    ok = False
                    break
            if ok:
                kept.append(d)
        return kept
