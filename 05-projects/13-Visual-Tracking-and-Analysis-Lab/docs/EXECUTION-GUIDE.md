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
