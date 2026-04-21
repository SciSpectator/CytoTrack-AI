#!/usr/bin/env python3
"""
CytoTrack AI v1.0
==================
Formerly "Cell Tracker v5.0". Rebranded and upgraded:

* Modern SORT-style tracker (Kalman + Hungarian assignment, re-detection
  every frame) replaces the legacy per-cell CSRT tracker.
* Detector fuses adaptive-threshold, Otsu, distance-transform watershed,
  blob, and Hough-circle strategies via IoU-based NMS.
* Same friendly pygame GUI with the full workflow: image settings
  preview, manual/auto/fast classification, migration analytics, and
  publication-quality plots.
"""

import os
import sys
from typing import Optional

# opencv-python ships its own Qt5 plugins that conflict with the system
# PyQt5 installation (the `xcb` plugin from cv2 can't find its transitive
# dependencies). Remove cv2's plugin hints before any Qt / cv2 import so
# the system plugin path wins.
os.environ.pop("QT_PLUGIN_PATH", None)
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
for _cand in (
    "/usr/lib/x86_64-linux-gnu/qt5/plugins",
    "/usr/lib64/qt5/plugins",
):
    if os.path.isdir(_cand):
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(_cand, "platforms")
        break

APP_NAME = "CytoTrack AI"
APP_VERSION = "1.0"
APP_FULL = f"{APP_NAME} v{APP_VERSION}"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


class Settings:
    brightness = 0
    contrast = 1.0
    gamma = 1.0
    filter_mode = 0


def main():
    try:
        from desktop_gui import show_splash_screen
        show_splash_screen(duration=2)
    except Exception:
        pass

    try:
        from desktop_gui import FancyGUI

        gui = FancyGUI()

        while gui.running:
            choice = gui.show_main_menu()

            if choice is None or choice == "Exit":
                break
            elif "Track" in choice:
                run_tracking(gui)
            elif "Train Phenotype (Online" in choice or "Online DB" in choice:
                run_training_online(gui)
            elif "Train" in choice:
                run_training(gui)
            elif "Generate Test" in choice:
                run_generate_data(gui)
            elif "Analyze" in choice:
                run_analysis(gui)
            elif "Help" in choice:
                show_help(gui)

        gui.cleanup()

    except Exception:
        import traceback
        traceback.print_exc()


def _cellpose_available() -> bool:
    try:
        import cellpose  # noqa: F401
        return True
    except Exception:
        return False


def _collect_feature_dataset(data_dir: str, progress=None,
                             progress_base: int = 0, progress_span: int = 70):
    """Walk a ``class_name/image.png`` folder tree, extract a Cellpose-SAM
    256-dim style vector per image (treating each image as one labeled
    cell), and return ``(features, labels, classes)``. Raises
    ``RuntimeError`` if Cellpose is unavailable."""
    import cv2
    import numpy as np
    from detector import CellDetector

    image_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    class_dirs = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith(".")
    ])
    if not class_dirs:
        raise ValueError(f"No class subfolders found in {data_dir}")

    # Flat list of (path, label) across all classes, so we can batch
    # across classes (Cellpose is much faster batched than one-by-one).
    samples: list = []
    for cls in class_dirs:
        cdir = os.path.join(data_dir, cls)
        for name in sorted(os.listdir(cdir)):
            if name.lower().endswith(image_exts):
                samples.append((os.path.join(cdir, name), cls))

    if not samples:
        raise ValueError(f"No images found under {data_dir}")

    detector = CellDetector(sensitivity="ai")
    if not detector._load_ai_model():
        raise RuntimeError(
            "Cellpose-SAM model could not be loaded — feature training "
            "requires a working cellpose installation.")

    features: list = []
    labels: list = []
    total = len(samples)
    BATCH = 32
    for i in range(0, total, BATCH):
        batch = samples[i:i + BATCH]
        batch_imgs = []
        for path, _ in batch:
            img = cv2.imread(path)
            if img is None:
                batch_imgs.append(None)
                continue
            batch_imgs.append(img)

        # extract_cell_features wants one image at a time with a bbox
        # list — we reuse it by giving the full-image bbox for each
        # single-cell training image. Keeps the feature source identical
        # to how the tracker will call it at inference time.
        for j, img in enumerate(batch_imgs):
            if img is None:
                continue
            h, w = img.shape[:2]
            feats = detector.extract_cell_features(img, [(0, 0, w, h)])
            if not feats:
                continue
            features.append(feats[0])
            labels.append(batch[j][1])

        if progress is not None:
            done = min(i + BATCH, total)
            pct = progress_base + int(progress_span * done / total)
            progress(pct, f"Extracting features: {done}/{total}")

    if not features:
        raise RuntimeError(
            "No features were extracted — check that training images are "
            "valid and readable.")
    classes = sorted({l for l in labels})
    return features, labels, classes


def _load_any_classifier(model_path: str):
    """Probe a saved model folder and load either a pixel-based
    ``CellClassifierTrainer`` or the Cellpose-SAM feature-based
    ``FeatureClassifier``. Returns ``(classifier, kind)`` where
    ``kind`` is ``"feature"`` or ``"pixel"``."""
    import json as _json
    info_path = os.path.join(model_path, "model_info.json")
    if os.path.isfile(info_path):
        try:
            with open(info_path, "r") as f:
                info = _json.load(f)
            if info.get("type") == "feature_mlp":
                from classifier import FeatureClassifier
                return FeatureClassifier.load_model(model_path), "feature"
        except Exception:
            pass
    from classifier import CellClassifierTrainer
    return CellClassifierTrainer.load_model(model_path), "pixel"


def _classify_one(classifier, kind, detector, image, bbox):
    """Route a single-cell prediction to the right backend. Returns
    ``(label, confidence)`` or ``("Unknown", 0.0)`` on failure."""
    try:
        if kind == "feature":
            feats = detector.extract_cell_features(image, [bbox])
            if not feats:
                return ("Unknown", 0.0)
            return classifier.predict(feats[0])
        x, y, w, h = bbox
        H, W = image.shape[:2]
        pad = 10
        x1 = max(0, x - pad); y1 = max(0, y - pad)
        x2 = min(W, x + w + pad); y2 = min(H, y + h + pad)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return ("Unknown", 0.0)
        return classifier.predict(crop)
    except Exception:
        return ("Unknown", 0.0)


def show_help(gui):
    gui.show_message(
        f"{APP_FULL} - Help",
        "WORKFLOW:\n"
        "1. Click Track Cells\n"
        "2. Select image folder\n"
        "3. Adjust image settings (preview)\n"
        "4. Click Start Tracking\n\n"
        "KEY FEATURES:\n"
        "• Kalman + Hungarian tracker\n"
        "• Multi-strategy detector\n"
        "• Better re-detection across frames\n\n"
        "OUTPUT FILES:\n"
        "• tracking_video.avi\n"
        "• plot_trajectories.png\n"
        "• plot_trajectories_interactive.html\n"
        "• migration_detailed.csv\n"
        "• migration_summary.csv",
        ["OK"],
    )


def run_tracking(gui):
    """Select data, classification mode, image settings, then track."""
    import cv2
    from desktop_gui import show_image_settings_preview, manual_cell_classification
    from detector import CellDetector

    folder = gui.show_folder_dialog("Select Image Folder")
    if not folder:
        return

    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

    def _images_in(d):
        try:
            return sorted([os.path.join(d, f) for f in os.listdir(d)
                           if f.lower().endswith(exts)])
        except OSError:
            return []

    files = _images_in(folder)
    # If the user picked the output root from "Generate Test Data"
    # (e.g. `test_data/`), the actual PNGs live in `frames/` one level
    # down. Auto-descend so the user doesn't have to know the layout.
    if not files:
        frames_sub = os.path.join(folder, "frames")
        if os.path.isdir(frames_sub):
            files = _images_in(frames_sub)
            if files:
                folder = frames_sub

    if not files:
        gui.show_message(
            "Error",
            f"No images found in:\n{folder}\n\n"
            f"Looked for: {', '.join(exts)}\n"
            f"If you used 'Generate Test Data', select the parent folder "
            f"(e.g. 'test_data') or the 'frames' subfolder directly.",
            ["OK"])
        return

    mode = gui.show_message(
        "Classification Mode",
        "How do you want to identify cells?\n\n"
        "• Fast Mode - Assign ONE type to all cells\n"
        "• Manual - YOU classify each cell interactively\n"
        "• Auto-Classify - AI classifies each cell\n"
        "• No Classification - Track without types",
        ["Fast Mode", "Manual", "Auto-Classify", "No Classification"],
    )

    if mode is None:
        return

    classifier = None
    classifier_kind = None
    single_cell_type = None
    manual_types = None
    cell_type_list = []

    if mode == "Fast Mode":
        params = gui.show_input_dialog(
            "Cell Type",
            ["Enter cell type name (e.g., HeLa, MCF7, Fibroblast)"],
            ["Cell"],
        )
        if params and params[0].strip():
            single_cell_type = params[0].strip()
        else:
            return

    elif mode == "Manual":
        params = gui.show_input_dialog(
            "Define Cell Types",
            ["Enter cell types separated by comma\n(e.g., Cancer, Healthy, Debris)"],
            ["TypeA, TypeB, Debris"],
        )
        if params and params[0].strip():
            cell_type_list = [t.strip() for t in params[0].split(",") if t.strip()]
            if not cell_type_list:
                gui.show_message("Error", "Enter at least one cell type!", ["OK"])
                return
        else:
            return

    elif mode == "Auto-Classify":
        model_dir = gui.show_folder_dialog("Select Trained Model Folder")
        if not model_dir:
            return

        model_path = os.path.join(model_dir, "model")
        if not os.path.exists(model_path):
            model_path = model_dir

        if not os.path.exists(os.path.join(model_path, "class_map.json")):
            gui.show_message(
                "Error",
                "Invalid model folder!\n\nMust contain class_map.json",
                ["OK"],
            )
            return

        try:
            classifier, classifier_kind = _load_any_classifier(model_path)
            kind_label = ("Cellpose-SAM feature MLP"
                          if classifier_kind == "feature" else "pixel-based ViT/CNN")
            gui.show_message(
                "Model Loaded",
                f"Backend: {kind_label}\n\n"
                f"Cell types in model:\n{', '.join(classifier.classes)}",
                ["Continue"],
            )
        except Exception as e:
            gui.show_message("Error", f"Failed to load model:\n{e}", ["OK"])
            return

    gui.show_message(
        "Image Settings",
        f"Found {len(files)} images.\n\n"
        "Next: Adjust brightness, contrast, gamma, and filters.\n"
        "Preview first 10 frames before tracking.",
        ["Continue"],
    )

    confirmed = show_image_settings_preview(gui.screen, files, Settings)
    if not confirmed:
        return

    if mode == "Manual":
        gui.show_message(
            "Manual Classification",
            "Now detecting cells in first frame...\n\n"
            "You will classify each detected cell by:\n"
            "• Clicking the type button, or\n"
            "• Pressing number keys 1-8",
            ["Continue"],
        )

        first_frame = cv2.imread(files[0])
        if first_frame is None:
            gui.show_message("Error", "Cannot read first image!", ["OK"])
            return

        detector = CellDetector(min_area=50, max_area=8000,
                                sensitivity="ai")
        detector.calibrate(first_frame)
        detections = detector.detect(first_frame)

        if not detections:
            gui.show_message("Error", "No cells detected in first frame!", ["OK"])
            return

        gui.show_message(
            "Cells Detected",
            f"Found {len(detections)} cells.\n\n"
            "Now classify each cell interactively.",
            ["Start Classification"],
        )

        manual_types = manual_cell_classification(gui.screen, first_frame,
                                                  detections, cell_type_list)

        if manual_types is None:
            gui.show_message("Cancelled", "Classification cancelled.", ["OK"])
            return

        if len(manual_types) != len(detections):
            gui.show_message(
                "Incomplete",
                f"Only {len(manual_types)}/{len(detections)} cells classified.\n"
                "Please classify all cells.",
                ["OK"],
            )
            return

    params = gui.show_input_dialog(
        "Tracking Parameters",
        ["Pixel size (µm/px)", "Time per frame (sec)"],
        ["1.0", "60"],
    )
    if not params:
        return

    try:
        pixel_size = float(params[0].replace(",", "."))
        time_per_frame = float(params[1].replace(",", "."))
    except ValueError:
        gui.show_message("Error", "Invalid numbers!", ["OK"])
        return

    try:
        import cellpose  # noqa: F401
    except Exception:
        gui.show_message(
            "Cellpose Required",
            "AI detection (Cellpose-SAM) is required for tracking but the\n"
            "'cellpose' package is not installed in this environment.\n\n"
            "Install with:  pip install cellpose",
            ["OK"],
        )
        return

    _do_tracking(
        gui, files, pixel_size, time_per_frame,
        classifier=classifier,
        classifier_kind=classifier_kind,
        single_cell_type=single_cell_type,
        manual_types=manual_types,
        sensitivity="ai",
    )


def _do_tracking(gui, files, pixel_size, time_per_frame,
                 classifier=None, classifier_kind: Optional[str] = None,
                 single_cell_type=None, manual_types=None,
                 sensitivity: str = "ai"):
    import cv2
    import numpy as np
    from datetime import datetime

    from detector import CellDetector
    from tracker import CellTracker
    from analyzer import MigrationAnalyzer
    from visualizer import TrajectoryVisualizer
    from image_utils import apply_all_adjustments, FILTER_NAMES
    from debris_reasoner import DebrisReasoner, filter_debris
    from ai_assistant import VisualLLMHelper
    from lost_cell_recovery import LostCellRecovery
    from hardware_profile import detect_hardware

    hw = detect_hardware()
    print("[hardware] " + hw.summary())
    print("[hardware] " + hw.reason)

    progress = gui.show_progress("Tracking", len(files) + 30)

    try:
        progress(2, "Loading first frame...")
        first = cv2.imread(files[0])
        if first is None:
            gui.show_message("Error", "Cannot read image!", ["OK"])
            return

        height, width = first.shape[:2]

        progress(5, "Detecting cells...")
        # Apply hardware-tier latency knobs (accuracy is unchanged — core
        # detection strategies always run; only the optional extras are gated).
        detector = CellDetector(min_area=50, max_area=8000,
                                use_blob_detector=hw.use_blob_detector,
                                use_hough_circles=hw.use_hough_circles,
                                sensitivity=sensitivity)
        print(f"[detector] sensitivity={sensitivity}")
        # Auto-tune detector to the actual cell size on this dataset.
        calib = detector.calibrate(first)
        print(f"Calibration: {calib}")
        raw_detections = detector.detect(first)
        reasoner = DebrisReasoner(strategy="auto")
        visual_llm = VisualLLMHelper(prefer="auto")
        detections, rejected = filter_debris(detector, first, raw_detections,
                                             reasoner=reasoner)
        print(f"Debris reasoner ({reasoner.strategy}): kept "
              f"{len(detections)}/{len(raw_detections)} detections "
              f"(visual-LLM backend: {visual_llm.backend})")
        num_detected = len(detections)

        if num_detected == 0:
            gui.show_message("Error", "No cells detected!", ["OK"])
            return

        cell_types = {}
        if manual_types:
            progress(8, "Using manual classifications...")
            cell_types = manual_types.copy()
        elif single_cell_type:
            progress(8, f"Assigning type: {single_cell_type}...")
            for i in range(len(detections)):
                cell_types[i] = single_cell_type
        elif classifier:
            progress(8, "Classifying cells with AI...")
            for i, det in enumerate(detections):
                ctype, _ = _classify_one(classifier, classifier_kind,
                                         detector, first, det.bbox)
                cell_types[i] = ctype
        else:
            for i in range(len(detections)):
                cell_types[i] = "Cell"

        progress(10, f"Found {num_detected} cells!")

        folder = os.path.dirname(files[0])
        output_dir = os.path.join(
            os.path.dirname(folder),
            f"tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        os.makedirs(output_dir, exist_ok=True)

        video_path = os.path.join(output_dir, "tracking_video.avi")
        has_types = bool(classifier or single_cell_type or manual_types)
        hud_h = 130 if has_types else 100
        video = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*"XVID"),
            float(hw.video_fps), (width, height + hud_h),
        )

        tracker = CellTracker(max_missed=15)
        # Tune Hungarian gating to the cell size so dense scenes don't ID-swap.
        tr_calib = tracker.calibrate(detections)
        print(f"Tracker gating: {tr_calib}")
        tracker.initialize(first, detections)

        recovery = LostCellRecovery(
            strategy="auto" if hw.enable_recovery_llm else "heuristic")
        prev_frame = first.copy()
        recovered_total = 0

        for tid, track in tracker.tracks.items():
            track.cell_type = cell_types.get(
                tid, single_cell_type if single_cell_type else "Cell"
            )

        total = len(files)
        for idx, fpath in enumerate(files):
            progress(12 + idx, f"Frame {idx+1}/{total}")

            frame = cv2.imread(fpath)
            if frame is None:
                continue

            frame_display = apply_all_adjustments(
                frame, Settings.brightness, Settings.contrast,
                Settings.gamma, Settings.filter_mode,
            )

            if idx > 0:
                # Re-detect then assign -> much better than CSRT drift.
                dets_raw = detector.detect(frame)
                dets, _ = filter_debris(detector, frame, dets_raw,
                                        reasoner=reasoner)
                tracker.update(frame, detections=dets)

                # Try to recover any track that went inactive mid-frame
                # (not at the border — that'd be a legitimate exit).
                try:
                    recs = recovery.recover(tracker, frame, prev_frame, dets)
                    for r in recs:
                        if r.recovered:
                            recovered_total += 1
                            print(f"  [recovery/{r.method}] track {r.track_id}: "
                                  f"{r.reasoning[:80]}")
                except Exception as _re:
                    pass

                prev_frame = frame.copy()
                # Classify newly spawned tracks so mid-sequence arrivals get
                # phenotype labels too, not just the frame-1 detections.
                for tid, t in tracker.tracks.items():
                    if hasattr(t, "cell_type") and t.cell_type is not None:
                        continue
                    default_type = single_cell_type if single_cell_type else "Cell"
                    if classifier and t.boxes:
                        ctype, _ = _classify_one(classifier, classifier_kind,
                                                 detector, frame, t.boxes[-1])
                        t.cell_type = ctype if ctype != "Unknown" else default_type
                    else:
                        t.cell_type = default_type

            vis = frame_display.copy()

            for tid, track in tracker.tracks.items():
                if not track.boxes:
                    continue

                color = track.color

                if len(track.boxes) > 1:
                    pts = [(int(bx + bw / 2), int(by + bh / 2))
                           for bx, by, bw, bh in track.boxes[-50:]]
                    for i in range(1, len(pts)):
                        cv2.line(vis, pts[i - 1], pts[i], color, 2)

                if track.is_active and track.boxes:
                    display = track.display_bbox() if hasattr(
                        track, "display_bbox") else track.boxes[-1]
                    x, y, w, h = display if display else track.boxes[-1]
                    cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
                    cv2.circle(vis, (int(x + w / 2), int(y + h / 2)), 4, color, -1)

                    if classifier and getattr(track, "cell_type", None):
                        label = f"{tid}:{track.cell_type}"
                    else:
                        label = f"ID:{tid}"
                    cv2.putText(vis, label, (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
                    cv2.putText(vis, label, (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            hud = np.zeros((hud_h, width, 3), dtype=np.uint8)
            hud[:] = (40, 40, 40)

            mode_str = ""
            if classifier:
                mode_str = " (AI Classification)"
            elif manual_types:
                mode_str = " (Manual Classification)"
            elif single_cell_type:
                mode_str = f" ({single_cell_type})"

            title = f"{APP_FULL}{mode_str}"
            cv2.putText(hud, title, (15, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 150), 2)
            cv2.putText(hud, f"Frame: {idx+1}/{total}", (15, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(hud, f"Detected: {num_detected}", (180, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(hud, f"Active: {tracker.active_count}", (350, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(hud, f"Lost: {tracker.lost_count}", (500, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            filter_name = FILTER_NAMES[Settings.filter_mode].split(":")[1].strip() \
                if ":" in FILTER_NAMES[Settings.filter_mode] else FILTER_NAMES[Settings.filter_mode]
            settings_str = (f"B:{Settings.brightness:+d} C:{Settings.contrast:.1f} "
                            f"G:{Settings.gamma:.1f} F:{filter_name}")
            cv2.putText(hud, settings_str, (15, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

            if has_types:
                counts = {}
                for t in tracker.tracks.values():
                    if t.is_active and hasattr(t, "cell_type"):
                        counts[t.cell_type] = counts.get(t.cell_type, 0) + 1
                type_str = "  ".join([f"{k}:{v}" for k, v in counts.items()])
                cv2.putText(hud, f"Types: {type_str}", (15, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)

            bar_y = 105 if classifier else 80
            prog_w = int((width - 650) * (idx + 1) / total)
            cv2.rectangle(hud, (640, bar_y), (width - 20, bar_y + 15), (60, 60, 60), -1)
            cv2.rectangle(hud, (640, bar_y), (640 + prog_w, bar_y + 15),
                          (100, 200, 150), -1)

            video.write(np.vstack([hud, vis]))

        video.release()

        progress(total + 15, "Analyzing tracks...")
        tracks = tracker.get_tracks(min_length=5)

        detailed_df = None
        summary_df = None

        if tracks:
            analyzer = MigrationAnalyzer(pixel_size, pixel_size, time_per_frame)
            detailed_df, summary_df = analyzer.analyze(tracks)

            type_map = {tid: t.cell_type for tid, t in tracker.tracks.items()
                        if hasattr(t, "cell_type")}

            if type_map:
                summary_df["Cell_Type"] = summary_df["TrackID"].map(
                    lambda x: type_map.get(x, "Cell"))
                detailed_df["Cell_Type"] = detailed_df["TrackID"].map(
                    lambda x: type_map.get(x, "Cell"))

            detailed_df.to_csv(os.path.join(output_dir, "migration_detailed.csv"),
                               index=False)
            summary_df.to_csv(os.path.join(output_dir, "migration_summary.csv"),
                              index=False)

            unique_types = summary_df["Cell_Type"].unique() \
                if "Cell_Type" in summary_df.columns else []
            if len(unique_types) > 1:
                comparison_df = analyzer.compare_types(summary_df)
                if comparison_df is not None:
                    comparison_df.to_csv(
                        os.path.join(output_dir, "statistical_comparison.csv"),
                        index=False)

                type_summary = analyzer.get_type_summary(summary_df)
                if type_summary is not None:
                    type_summary.to_csv(
                        os.path.join(output_dir, "cell_type_summary.csv"),
                        index=False)

        progress(total + 20, "Generating publication plots...")
        if tracks and detailed_df is not None and summary_df is not None:
            visualizer = TrajectoryVisualizer(pixel_size, pixel_size, time_per_frame)
            plot_files = visualizer.generate_all_plots(tracks, detailed_df,
                                                       summary_df, output_dir)
            print(f"Generated {len(plot_files)} publication-quality plots")

        progress(total + 30, "Done!")

        settings_info = (f"Image Settings Used:\n"
                         f"  Brightness: {Settings.brightness}\n"
                         f"  Contrast: {Settings.contrast}\n"
                         f"  Gamma: {Settings.gamma}\n"
                         f"  Filter: {FILTER_NAMES[Settings.filter_mode]}\n")

        with open(os.path.join(output_dir, "settings_used.txt"), "w") as f:
            f.write(settings_info)

        msg = (f"Detected: {num_detected}\n"
               f"Tracks: {len(tracks)}\n\n"
               f"Image Settings Applied:\n"
               f"  Brightness: {Settings.brightness}\n"
               f"  Contrast: {Settings.contrast:.1f}\n"
               f"  Gamma: {Settings.gamma:.1f}\n"
               f"  Filter: {FILTER_NAMES[Settings.filter_mode]}\n\n"
               f"Output: {output_dir}")

        if classifier:
            counts = {}
            for t in tracker.tracks.values():
                if hasattr(t, "cell_type"):
                    counts[t.cell_type] = counts.get(t.cell_type, 0) + 1
            msg += "\n\nCell Types:\n" + "\n".join(
                [f"  {k}: {v}" for k, v in counts.items()])

        gui.show_message("Success!", msg, ["OK"])

    except Exception as e:
        import traceback
        traceback.print_exc()
        gui.show_message("Error", str(e), ["OK"])


def run_training(gui):
    try:
        import torch  # noqa: F401
        from classifier import CellClassifierTrainer
    except ImportError as e:
        gui.show_message(
            "Missing Requirements",
            f"Install PyTorch:\npip install torch torchvision Pillow\n\nError: {e}",
            ["OK"],
        )
        return

    mode = gui.show_message(
        "Train Cell Classifier",
        "Prepare your training data:\n\n"
        "training_data/\n"
        "├── CellTypeA/ (20+ images)\n"
        "├── CellTypeB/ (20+ images)   (optional — train with 1 class too)\n"
        "└── ...\n\n"
        "You can train a fresh model, or continue training an existing "
        "model with NEW phenotypes added to it (incremental training).",
        ["Train new model", "Continue existing model", "Cancel"],
    )

    if mode == "Cancel" or mode is None:
        return

    resume_from: Optional[str] = None
    if mode == "Continue existing model":
        picked = gui.show_folder_dialog(
            "Select existing model folder (the one containing class_map.json)"
        )
        if not picked:
            return
        map_path = os.path.join(picked, "class_map.json")
        if not os.path.isfile(map_path):
            gui.show_message(
                "Invalid model folder",
                f"No class_map.json in:\n{picked}\n\n"
                "Select the 'model/' subfolder that was produced by a "
                "previous training run.",
                ["OK"],
            )
            return
        resume_from = picked

    data_dir = gui.show_folder_dialog("Select Training Data Folder")
    if not data_dir:
        return

    subfolders = [d for d in os.listdir(data_dir)
                  if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith(".")]

    if len(subfolders) < 1:
        gui.show_message(
            "Error",
            "Need at least 1 class folder with images.\n"
            f"Found: {subfolders}",
            ["OK"],
        )
        return

    params = gui.show_input_dialog(
        "Training Settings",
        ["Epochs (5-20)", "Model name"],
        ["10", "my_cell_model"],
    )
    if not params:
        return

    try:
        epochs = int(params[0])
        model_name = params[1]
    except ValueError:
        gui.show_message("Error", "Invalid parameters!", ["OK"])
        return

    output_dir = os.path.join(os.path.dirname(data_dir), model_name)

    # Default to the Cellpose-SAM feature path when cellpose is available:
    # it's much faster, works with very few samples per phenotype, and
    # shares its feature space with the tracker so inference stays
    # consistent with what the tracker sees on the frame.
    use_feature_path = _cellpose_available()
    if resume_from is not None:
        # Make resume_from select the same backend the existing model
        # was trained with, regardless of current toggle.
        info_path = os.path.join(resume_from, "model_info.json")
        if os.path.isfile(info_path):
            try:
                import json as _json
                with open(info_path) as _f:
                    use_feature_path = _json.load(_f).get("type") == "feature_mlp"
            except Exception:
                pass
        elif os.path.isfile(os.path.join(resume_from, "class_map.json")):
            # Legacy pixel-based model — stick with pixel trainer.
            use_feature_path = False

    if use_feature_path:
        try:
            _train_feature_classifier(
                gui, data_dir=data_dir, output_dir=output_dir,
                epochs=max(epochs, 20), resume_from=resume_from,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            gui.show_message("Training Failed", str(e), ["OK"])
        return

    try:
        progress = gui.show_progress("Training AI Model", 100)

        progress(5, "Initializing...")
        trainer = CellClassifierTrainer(output_dir=output_dir,
                                        resume_from=resume_from)

        progress(10, "Loading data...")
        train_loader, val_loader = trainer.prepare_data(data_dir)

        def prog_cb(step, total, msg):
            pct = 15 + int(80 * step / max(1, total))
            progress(pct, msg)

        history = trainer.train(
            train_loader, val_loader,
            epochs=epochs,
            progress_callback=prog_cb,
        )

        progress(100, "Complete!")

        final_acc = history["val_acc"][-1] * 100 if history["val_acc"] else 0.0
        best_acc = max(history["val_acc"]) * 100 if history["val_acc"] else 0.0

        gui.show_message(
            "Training Complete!",
            f"Final accuracy: {final_acc:.1f}%\n"
            f"Best accuracy: {best_acc:.1f}%\n"
            f"Classes ({len(trainer.classes)}): "
            f"{', '.join(trainer.classes)}\n\n"
            f"Model saved to:\n{output_dir}/model/\n\n"
            "Tip: use 'Continue existing model' next time to add more "
            "phenotypes without losing what's already learned.",
            ["OK"],
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        gui.show_message("Training Failed", str(e), ["OK"])


def _train_feature_classifier(gui, data_dir: str, output_dir: str,
                              epochs: int, resume_from: Optional[str]) -> None:
    """Shared feature-based training worker for the local-folder and
    online-DB flows. Saves to ``<output_dir>/model/``."""
    from classifier import FeatureClassifier

    model_dir = os.path.join(output_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    progress = gui.show_progress("Training phenotype classifier (AI features)", 100)
    progress(5, "Scanning training folders...")
    features, labels, classes = _collect_feature_dataset(
        data_dir, progress=progress, progress_base=10, progress_span=60,
    )
    progress(75, f"Extracted {len(features)} feature vectors "
                 f"across {len(classes)} phenotype(s)")

    clf = FeatureClassifier(output_dir=model_dir, resume_from=resume_from)
    history = clf.fit(features, labels, epochs=max(20, epochs))
    progress(100, "Complete!")

    best_acc = float(history.get("val_accuracy", 0.0)) * 100
    gui.show_message(
        "Training Complete!",
        f"Backend: Cellpose-SAM feature MLP\n"
        f"Best val accuracy: {best_acc:.1f}%\n"
        f"Classes ({len(clf.classes)}): {', '.join(clf.classes)}\n\n"
        f"Model saved to:\n{model_dir}/\n\n"
        "Tip: select 'Continue existing model' to add more phenotypes "
        "later without losing what's already learned.",
        ["OK"],
    )


def run_training_online(gui):
    """Search the open-licensed cell-image database and train a classifier
    on what we pulled. Every downloaded file is verified to be on a
    permissive licence before it touches disk."""
    try:
        import torch  # noqa: F401
        from classifier import CellClassifierTrainer
    except ImportError as e:
        gui.show_message(
            "Missing Requirements",
            f"Install PyTorch:\npip install torch torchvision Pillow\n\n"
            f"Error: {e}",
            ["OK"],
        )
        return

    from cell_image_library import search, build_phenotype_folders

    gui.show_message(
        "Open-License Cell Image Database",
        "Search an indexed catalogue of open-licensed cell image datasets "
        "(BBBC, Cell Image Library). Only CC-0 / CC-BY / CC-BY-SA entries "
        "are downloaded, with full attribution kept next to the images.\n\n"
        "You can build a multi-class training set by searching for each "
        "phenotype one at a time.",
        ["Continue"],
    )

    selections = []  # list of (class_label, Dataset)
    while True:
        params = gui.show_input_dialog(
            f"Search Datasets  ({len(selections)} class(es) chosen so far)",
            ["Phenotype / keyword (e.g. HeLa, MCF-7, U2OS, nuclei)"],
            [""],
        )
        if not params:
            break
        query = (params[0] or "").strip()
        if not query:
            break

        hits = search(query)
        if not hits:
            gui.show_message(
                "No matches",
                f"Nothing in the open catalogue matched '{query}'.\n"
                "Try a broader keyword like 'nuclei' or 'breast'.",
                ["OK"],
            )
            continue

        # Present up to 4 results as buttons.
        buttons = [f"{d.id} · {d.name[:40]}" for d in hits[:4]]
        buttons.append("Skip")
        choice = gui.show_message(
            f"Results for '{query}'",
            "\n".join(
                f"{d.id}: {d.name}\n   licence: {d.licence}\n"
                f"   ~{d.approx_image_count} images — {d.homepage}"
                for d in hits[:4]
            ),
            buttons,
        )
        if not choice or choice == "Skip":
            continue
        picked = next((d for d in hits[:4]
                       if choice.startswith(d.id + " ")), None)
        if picked is None:
            continue

        label_params = gui.show_input_dialog(
            "Class label",
            ["Label this phenotype (shown in tracking overlay + CSV)"],
            [picked.phenotype or picked.name],
        )
        if not label_params:
            continue
        label = (label_params[0] or picked.phenotype or picked.id).strip()
        selections.append((label, picked))

        again = gui.show_message(
            "Added",
            f"Added class '{label}' from {picked.id}.\n\n"
            f"Total classes: {len(selections)}.\n"
            "Add another phenotype?",
            ["Add another", "Start training"],
        )
        if again == "Start training":
            break

    if len(selections) < 1:
        gui.show_message(
            "Nothing selected",
            "Pick at least one phenotype to train on.",
            ["OK"],
        )
        return

    # Offer to resume an existing model so new phenotypes can be stacked
    # on top of previous training rather than discarding that knowledge.
    resume_from: Optional[str] = None
    resume_choice = gui.show_message(
        "Continue existing model?",
        "Would you like to add these phenotypes on top of an already-"
        "trained model (incremental training)?",
        ["Train new model", "Continue existing model"],
    )
    if resume_choice == "Continue existing model":
        picked = gui.show_folder_dialog(
            "Select existing model folder (the one containing class_map.json)"
        )
        if picked and os.path.isfile(os.path.join(picked, "class_map.json")):
            resume_from = picked
        elif picked:
            gui.show_message(
                "Invalid model folder",
                "No class_map.json in that folder. Continuing with a "
                "fresh model instead.",
                ["OK"],
            )

    limits = gui.show_input_dialog(
        "Download & training limits",
        ["Images per class (50-300)", "Epochs (5-15)", "Model folder name"],
        ["100", "8", "phenotype_model"],
    )
    if not limits:
        return
    try:
        per_class = max(10, min(1000, int(limits[0])))
        epochs = max(1, min(100, int(limits[1])))
        model_name = limits[2].strip() or "phenotype_model"
    except ValueError:
        gui.show_message("Error", "Invalid numbers!", ["OK"])
        return

    data_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "phenotype_data",
        model_name,
    )
    try:
        progress = gui.show_progress(
            "Downloading open-licensed images…",
            len(selections) + epochs + 5,
        )
        for i, (label, d) in enumerate(selections):
            progress(i, f"{d.id} ({d.licence}) → {label}")
        build_phenotype_folders(selections, data_root,
                                max_samples_per_class=per_class)

        progress(len(selections), "Preparing training set…")
        output_dir = os.path.join(os.path.dirname(data_root), model_name)

        # Prefer Cellpose-SAM feature path when available, and keep
        # incremental-training backends consistent with the loaded model.
        use_feature_path = _cellpose_available()
        if resume_from is not None:
            info_path = os.path.join(resume_from, "model_info.json")
            if os.path.isfile(info_path):
                try:
                    import json as _json
                    with open(info_path) as _f:
                        use_feature_path = _json.load(_f).get(
                            "type") == "feature_mlp"
                except Exception:
                    pass
            elif os.path.isfile(os.path.join(resume_from, "class_map.json")):
                use_feature_path = False

        if use_feature_path:
            _train_feature_classifier(
                gui, data_dir=data_root, output_dir=output_dir,
                epochs=max(epochs, 20), resume_from=resume_from,
            )
            return

        trainer = CellClassifierTrainer(output_dir=output_dir,
                                        resume_from=resume_from)
        train_loader, val_loader = trainer.prepare_data(data_root)

        def prog_cb(step, total, msg):
            frac = step / max(1, total)
            progress(len(selections) + 1 + int(epochs * frac), msg)

        history = trainer.train(
            train_loader, val_loader,
            epochs=epochs,
            progress_callback=prog_cb,
        )
        progress(len(selections) + epochs + 5, "Done.")

        best_acc = max(history["val_acc"]) * 100
        gui.show_message(
            "Training complete",
            f"Best validation accuracy: {best_acc:.1f}%\n\n"
            f"Training data: {data_root}\n"
            f"Model folder:  {output_dir}/model/\n\n"
            "Each class folder contains a manifest.json with the original "
            "licence and attribution for that dataset.",
            ["OK"],
        )

    except PermissionError as pe:
        gui.show_message(
            "Blocked by licence check",
            f"{pe}\n\nNothing was downloaded.",
            ["OK"],
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        gui.show_message("Training Failed", str(e), ["OK"])


def run_analysis(gui):
    import pandas as pd

    csv_path = gui.show_file_dialog("Select CSV File", extensions=[".csv"])
    if not csv_path:
        return

    try:
        df = pd.read_csv(csv_path)
        output_dir = os.path.dirname(csv_path)

        if "TrackID" in df.columns:
            num_tracks = df["TrackID"].nunique()

            if "Avg_Velocity_um_min" in df.columns:
                from visualizer import TrajectoryVisualizer
                visualizer = TrajectoryVisualizer()

                vel_path = os.path.join(output_dir, "reanalysis_velocity.png")
                visualizer.plot_velocity_histogram(df, vel_path)

                if "Total_Distance_um" in df.columns and "Displacement_um" in df.columns:
                    disp_path = os.path.join(output_dir, "reanalysis_displacement.png")
                    visualizer.plot_displacement_vs_distance(df, disp_path)

                if "Cell_Type" in df.columns:
                    type_path = os.path.join(output_dir, "reanalysis_cell_types.png")
                    visualizer.plot_cell_type_distribution(df, type_path)

            msg = f"Tracks: {num_tracks}\n\n"

            if "Avg_Velocity_um_min" in df.columns:
                msg += f"Avg velocity: {df['Avg_Velocity_um_min'].mean():.2f} µm/min\n"

            if "Cell_Type" in df.columns:
                type_counts = df["Cell_Type"].value_counts()
                msg += "\nCell types:\n"
                for ct, count in type_counts.items():
                    msg += f"  {ct}: {count}\n"

            msg += f"\nPlots saved to: {output_dir}"

            gui.show_message("Analysis Complete", msg, ["OK"])
        else:
            gui.show_message("Info",
                             f"Loaded {len(df)} rows, {len(df.columns)} columns",
                             ["OK"])

    except Exception as e:
        gui.show_message("Error", f"Failed:\n{e}", ["OK"])


def run_generate_data(gui):
    params = gui.show_input_dialog(
        "Generate Test Data",
        ["Number of cells", "Number of frames", "Output folder"],
        ["100", "50", "test_data"],
    )
    if not params:
        return

    try:
        num_cells = int(params[0])
        num_frames = int(params[1])
        output = params[2]

        progress = gui.show_progress("Generating...", num_frames + 5)
        progress(2, f"Creating {num_cells} cells...")

        from synthetic_data import SyntheticDataGenerator
        import cv2
        import glob

        gen = SyntheticDataGenerator(1024, 768, num_cells, num_frames, seed=42)
        gen.generate_cells()

        # Resolve output to an absolute path anchored on the project dir
        # (not cwd) so behavior is identical when launched from the
        # desktop menu vs a terminal.
        if not os.path.isabs(output):
            output = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  output)
        out_dir = os.path.join(output, "frames")
        os.makedirs(out_dir, exist_ok=True)

        # Clear stale frames from earlier runs so the folder always
        # reflects the current generation. Without this, reducing
        # num_frames leaves the tail of a previous longer run behind,
        # and the folder's mtime stays frozen so file managers show it
        # as old — which looks like "nothing was saved".
        for stale in glob.glob(os.path.join(out_dir, "frame_*.png")):
            try:
                os.remove(stale)
            except OSError:
                pass

        written = 0
        for i in range(num_frames):
            progress(5 + i, f"Frame {i+1}/{num_frames}")
            img, _ = gen.generate_frame(i)
            path = os.path.join(out_dir, f"frame_{i:05d}.png")
            ok = cv2.imwrite(path, img)
            if not ok or not os.path.exists(path):
                raise IOError(f"cv2.imwrite failed for {path} — check that "
                              f"the folder is writable and has free disk space.")
            written += 1

        # Verify count matches and surface the absolute path so the user
        # can find the output in their file manager without guessing.
        if written != num_frames:
            raise RuntimeError(
                f"Only {written}/{num_frames} frames written to {out_dir}")

        gui.show_message(
            "Done!",
            f"Generated {num_frames} frames\nwith {num_cells} cells\n\n"
            f"Location:\n{out_dir}",
            ["OK"],
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        gui.show_message("Error",
                         f"Failed to save test data:\n{e}",
                         ["OK"])


if __name__ == "__main__":
    main()
