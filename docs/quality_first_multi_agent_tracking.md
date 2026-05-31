# Quality-First Multi-Agent Tracking Pipeline

This document sketches a quality-first architecture for CytoTrack AI that can use optional visual grounding backends, including NVIDIA LocateAnything when explicitly enabled, and surrounds detector outputs with visual curator agents. The goal is not maximum throughput; it is reproducible cell tracking output that survives visual review, identity handoffs, morphology checks, and result packaging.

## Pipeline Overview

1. **Detector agents** propose cell candidates for each frame.
   - Optional visual grounding detector agents receive microscopy frames plus prompts such as `cell`, `round cell`, `elongated cell`, `mitotic cell`, and user-specified phenotypes.
   - Classical detector agents run in parallel where available: thresholding, blob detection, watershed, contour filtering, and existing trained classifiers.
   - Each detector emits boxes, masks or centroids, confidence, prompt/source metadata, and frame-local evidence crops.
   - Detector outputs are merged only after non-maximum suppression and disagreement logging, so low-confidence or conflicting detections stay reviewable instead of being silently discarded.

2. **Centroid curator** converts detections into physically plausible cell centers.
   - Normalizes boxes, masks, and point detections into centroids with estimated radius, area, eccentricity, and intensity descriptors.
   - Rejects impossible jumps, border artifacts, obvious debris, and duplicate detections over the same cell.
   - Flags ambiguous clusters for visual QA instead of forcing a single center when the image evidence is unclear.

3. **Identity curator** links curated centroids into tracks.
   - Applies motion continuity, appearance similarity, morphology stability, and division/merge rules frame by frame.
   - Maintains explicit identity uncertainty when cells overlap, disappear, split, or reappear.
   - Prefers short identity gaps with audit notes over confident but visually unsupported ID switches.

4. **Frame-by-frame visual QA curator** reviews the overlay result.
   - Renders every frame with detections, track IDs, trajectories, classifier labels, and uncertainty markers.
   - Compares the overlay against the raw frame and asks whether each visible cell is detected, each marker is centered, each ID is stable, and each trajectory is biologically plausible.
   - Produces a per-frame QA ledger: pass, corrected, needs-human-review, or excluded-with-reason.

5. **Morphology/classifier training curator** improves phenotype labels.
   - Builds phenotype training sets from user-labeled crops and open-licensed public datasets such as Broad Bioimage Benchmark Collection, Cell Image Library, Human Protein Atlas, and Cell Tracking Challenge sources when their licenses permit the intended use.
   - Preserves dataset attribution, license text, source URLs, preprocessing steps, class mappings, and train/validation/test splits.
   - Rejects training runs that mix incompatible imaging modalities or unlabeled weak matches without an explicit review flag.
   - Reports model confidence, confusion matrix, per-class examples, and failure cases back into the tracking QA loop.

6. **Result-bundle curator** enforces the `RESULT/` folder contract.
   - No run is complete until all required artifacts exist, have consistent frame counts and track IDs, and include provenance.
   - The curator validates file names, schemas, checksums, settings, model versions, source movie metadata, QA summaries, and license manifests before publishing the bundle.

## Agent Contracts

| Agent | Input | Output | Quality Gate |
| --- | --- | --- | --- |
| LocateAnything detector | Raw frame, prompt set, optional visual examples | Candidate boxes/points with confidences and crops | Must retain prompt/source metadata and low-confidence review candidates |
| Classical detector | Raw/preprocessed frame | Candidate masks, boxes, centroids | Must explain filtering thresholds and rejected components |
| Centroid curator | All detector candidates for one frame | Deduplicated centroid table | Must flag duplicates, off-cell centers, and ambiguous clusters |
| Identity curator | Curated centroids across frames | Track table with ID confidence | Must log every ID handoff, gap, merge, split, and correction |
| Visual QA curator | Raw frames plus overlay render | Per-frame QA ledger and corrections | Must review each frame, not only summary statistics |
| Training curator | Public/open datasets and user labels | Versioned classifier and dataset manifest | Must enforce license, attribution, split, and modality checks |
| Result-bundle curator | Tracking outputs and QA records | Validated `RESULT/` bundle | Must fail closed when required artifacts are missing or inconsistent |

## `RESULT/` Folder Structure

Each completed run should publish one immutable result folder:

```text
RESULT/
  manifest.json
  settings_used.txt
  provenance/
    input_movie_metadata.json
    model_versions.json
    dataset_licenses.json
    qa_decisions.jsonl
  frames/
    raw_index.csv
    overlays/
      frame_000000.png
      frame_000001.png
  detections/
    detector_candidates.csv
    curated_centroids.csv
    rejected_candidates.csv
  tracks/
    migration_detailed.csv
    migration_summary.csv
    identity_events.csv
  classifier/
    classifier_report.json
    confusion_matrix.png
    class_examples/
  plots/
    plot_trajectories.png
    plot_velocity_histogram.png
    plot_displacement_distance.png
    plot_directionality.png
    plot_msd.png
    plot_rose.png
  media/
    tracking_video.avi
  review/
    frame_qa_summary.csv
    needs_human_review.csv
```

The result-bundle curator should treat this structure as a contract. Missing files, mismatched IDs, changed settings, incomplete QA records, or absent license manifests block publication.

## Why Slower Frame-by-Frame Checks Are Acceptable

Cell tracking errors compound. A single missed detection, off-center centroid, or identity swap can corrupt downstream velocity, displacement, persistence, phenotype comparison, and statistical summaries. Faster batch-level checks can prove that a pipeline ran, but they cannot prove that every visible cell was followed correctly through occlusion, contact, division, debris, or focus changes.

Frame-by-frame visual QA is acceptable because CytoTrack AI's priority is quality over latency. Most microscopy tracking jobs are offline analysis tasks where users wait for trustworthy result bundles, not interactive real-time control loops. The slower curator pass creates an audit trail, enables targeted human review, and prevents polished CSV files from hiding visually obvious failures.

## Operating Principles

- Prefer uncertainty over invented certainty.
- Keep every correction traceable to frame, cell ID, agent, model version, and visual evidence.
- Separate detector recall from final acceptance: detector agents may over-propose, but curators decide what enters the final tracks.
- Never train or evaluate on public data without license, attribution, and split records.
- Publish only validated `RESULT/` bundles.
