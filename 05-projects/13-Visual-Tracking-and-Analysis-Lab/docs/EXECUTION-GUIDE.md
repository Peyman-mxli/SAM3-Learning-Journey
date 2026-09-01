# Execution Guide

## 1. Recommended environment

Use Google Colab with a GPU runtime because SAM 3 inference requires substantially more resources than the lightweight database/dashboard layer.

## 2. Install standard dependencies

```bash
pip install -r requirements.txt
```

## 3. Prepare SAM 3

Use the course SAM 3 installation and checkpoint already validated in the learning journey.

Project 13 does not download or redistribute model weights.

## 4. Run the first tracking session

From the Project 13 directory:

```bash
python src/pipeline.py \
  data/input/tracking_test_01.mp4 \
  --db data/project13.sqlite3 \
  --confidence 0.35 \
  --notes "Project 13 technical test 01"
```

Expected behavior:

```text
Video
  ↓
YOLO
  ↓
confidence filter
  ↓
ByteTrack
  ↓
SQLite session + observations
```

## 5. Start the dashboard

```bash
streamlit run app.py
```

The dashboard reads Project 13's SQLite database and displays:

- session history
- observation counts
- unique trackers
- average confidence
- tracker persistence
- class distribution

## 6. Run SAM 3 segmentation

Use `src/segmenter.py` inside the same environment where the local SAM 3 checkpoint is available.

The adapter deliberately does not claim segmentation success until it receives real model output.

## 7. Export predictions

Export frame-level predictions from SQLite to CSV before ground-truth evaluation.

The prediction schema should include at minimum:

```text
frame_index,class_name,x1,y1,x2,y2
```

## 8. Ground-truth evaluation

Create a human-reviewed CSV using the same schema, then run:

```bash
python evaluation/evaluate_detection_csv.py \
  results/predictions.csv \
  evaluation/ground_truth.csv
```

## 9. Segmentation metrics

When ground-truth masks are available, use:

- `mask_iou`
- `dice_score`

from `src/metrics.py`.

## 10. Evidence rule

Only generated Project 13 artifacts should be placed in `results/` and called Project 13 validation evidence.

Project 06 can be cited as previous experimentation, but its metrics must stay labeled as Project 06 metrics.


## 11. Run the complete controlled validation suite

Create five reproducible test videos from the verified source:

```bash
python evaluation/create_condition_variants.py data/input/tracking_test_01.mp4
```

Process all five videos as independent Project 13 sessions:

```bash
python evaluation/run_validation_suite.py
```

Compare robustness against the baseline:

```bash
python evaluation/analyze_validation_conditions.py
```

The suite covers baseline, low light, partial occlusion, motion blur, and
reduced scale.

## 12. Tracking identity ground truth

Use `evaluation/tracking_ground_truth_template.csv` to create reviewed temporal
identity annotations, then run `evaluation/evaluate_tracking_csv.py`.

## 13. Segmentation ground truth

Create human-reviewed binary masks and a manifest based on
`evaluation/segmentation_manifest_template.csv`, then run
`evaluation/evaluate_segmentation_masks.py`.

## 14. Final completion audit

See `docs/FINAL-COMPLETION-AUDIT.md` for the requirement-by-requirement
completion map and evidence boundary.
