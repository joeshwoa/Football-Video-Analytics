# Football Video Analytics: Detection, Tracking, Team Assignment (YOLO + Tracking + Temporal Smoothing)

End-to-end computer vision pipeline that processes football (soccer) match videos to detect and track **players**, **goalkeepers**, **referees**, and the **ball**, then assigns **Team A / Team B** in a match-agnostic way (unsupervised) and exports analytics-ready trajectories.

## Demo
- Final annotated video output: `final_result.mp4`
- Output table: `tracks_final.parquet` / `tracks_final.csv` (per-frame detections/tracks with class + team)

> The pipeline is designed to be used as a foundation for higher-level analytics: team movement analysis, player trajectories, ball trajectory, possession heuristics, heatmaps, and tactical insights.

---

## Features
### 1) Object detection (4 classes)
- `player`
- `goalkeeper`
- `referee`
- `ball`

Model: **Ultralytics YOLOv8m** fine-tuned on a merged dataset (Roboflow-exported + Kaggle datasets in YOLO format).

### 2) Multi-object tracking
- Uses Ultralytics tracking API with a tracker config (ByteTrack / BoT-SORT).
- Produces stable `track_id` values across frames.

### 3) Temporal smoothing (“track memory”)
Stabilizes detections across frames:
- **Class smoothing** per track using confidence-weighted voting (reduces flicker and misclassifications).
- Optional post-filters to remove false positives (e.g., pitch lines/grass artifacts).

### 4) Team assignment (generalizes across matches/teams)
Because team kits change by match, team identity is not hard-coded as detection classes.
Instead:
- For each tracked person, extract **jersey/torso appearance features** across frames.
- Remove grass pixels and aggregate features per `track_id`.
- Cluster into **Team A / Team B** using unsupervised learning.
- Referees are handled separately (including recovery when misclassified as players).

### 5) Gap filling / interpolation
Bridges short tracking gaps to create dense per-frame trajectories:
- If a track disappears for `≤ N` frames, interpolate bounding boxes across the missing frames.
- Adds `is_interpolated` flag to distinguish real detections vs filled frames.

### 6) Structured outputs for analytics
Exports per-frame records with:
- `frame`, `track_id`
- `bbox (x1, y1, x2, y2)`
- `conf`
- `class_name` (`player/gk/ref/ball`)
- `team_id`, `team_name` (`team_A/team_B/referee/unknown`)
- `is_interpolated`

---

## Repository / Notebook Structure
This project was developed on Kaggle (GPU). Suggested structure:

- `01_merge_datasets_and_train_yolo.ipynb`
  - merges multiple YOLO-format datasets into a single unified dataset
  - trains YOLOv8m on classes: player/goalkeeper/referee/ball
  - exports `best.pt`
- `02_infer_track_smooth_team.ipynb`
  - video inference + tracking
  - temporal smoothing
  - team assignment (unsupervised)
  - gap filling
  - exports final video + final tables

---

## Training Summary (example run)
- Train images: ~2,677
- Val images: ~251
- Class counts (sample):
  - player: ~47k
  - goalkeeper: ~1.7k
  - referee: ~5.6k
  - ball: ~2.0k

Evaluation (example):
- mAP50: ~0.875
- mAP50-95: ~0.611

---

## How to Run (Kaggle)
### 1) Train
1. Open `01_merge_datasets_and_train_yolo.ipynb` (GPU enabled).
2. Add datasets in Kaggle Inputs (YOLO format).
3. Run merge → train.
4. Export weights (`best.pt`) and zip outputs for download.

### 2) Inference + Tracking + Team Assignment
1. Open `02_infer_track_smooth_team.ipynb`
2. Add your `best.pt` model as input + add an `.mp4` match video input
3. Run:
   - tracking → smoothing → (optional) post-filter → team clustering → gap filling
4. Outputs saved to `/kaggle/working/`:
   - `final_result.mp4`
   - `tracks_final.parquet`, `tracks_final.csv`

---

## Notes on Generalization
- Detection is trained on generic classes; team identity is inferred **per match** using unsupervised jersey appearance clustering.
- This avoids hard-coding Team A / Team B into the detector, which would not generalize across different matches/uniforms.

---

## Limitations / Next Improvements
- Team clustering can be difficult if kits are similar or lighting is extreme; improvements:
  - stronger appearance embeddings (ReID model)
  - more robust clustering (GMM / spectral clustering)
- Referee recovery uses heuristics; can be improved by:
  - a small dedicated referee classifier (track-level)
- Ball tracking can benefit from:
  - higher inference resolution (`imgsz=1280`)
  - specialized ball dataset or segmentation approach
- For full tactical analytics:
  - camera calibration + homography to map pixel coordinates to pitch coordinates

---

## Tech Stack
- Python, OpenCV
- Ultralytics YOLO (YOLOv8)
- Tracking: ByteTrack / BoT-SORT
- Pandas / Parquet outputs
- Unsupervised clustering: scikit-learn (KMeans)

---

## License / Data
Model training used multiple public datasets with varying licenses. If publishing the model publicly, verify dataset licensing compatibility.
