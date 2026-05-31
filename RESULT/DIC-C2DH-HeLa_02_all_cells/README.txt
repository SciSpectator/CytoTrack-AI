CytoTrack AI small example result bundle
========================================

This folder contains at most 15 labelled cells from one real CTC movie.
All velocity, displacement, CDE, MSD, and trajectory metrics track the true cell center/centroid.
Cell edges, contours, and bounding boxes are visual context only, not tracked points.
Files:
- tracking_video.mp4 / tracking_video.avi: overlay video
- dashboard.html: interactive local dashboard
- migration_detailed.csv: per-frame migration metrics
- migration_summary.csv: per-track migration metrics
- gt_tracks.csv: selected mask-derived boxes
- frame_identity_qc.csv: per-frame centroid identity checks
- identity_quality_report.csv: per-track jump/identity checks
- plot_*.png / plot_interactive.html: publication plots
