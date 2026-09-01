# Project 13 Evaluation

Project 13 keeps **model output**, **human-reviewed ground truth**, and **evaluation metrics** separate.

## Evaluation target

The definitive persistent run is `lab13_0d46777217`.

Verified execution evidence already exists:

- 299 observations
- 10 unique tracker IDs
- average confidence 0.7059505883865931
- 21 observations with SAM 3 mask area
- average SAM 3 mask area 10047.095238095239 px

These are execution measurements, not accuracy claims.

## Ground-truth workflow

### 1. Extract a deterministic review set

For the 75-frame test video, the default extracts every third frame, producing 25 review images.

```bash
python evaluation/prepare_ground_truth.py \
  data/input/tracking_test_01.mp4 \
  --output-dir evaluation/ground_truth_frames \
  --every 3
```

This creates:

- `manifest.csv`
- `ground_truth_template.csv`
- 25 JPEG review frames

### 2. Bootstrap review candidates

Use the persistent observation export only as a starting aid:

```bash
python evaluation/bootstrap_review_candidates.py \
  /content/drive/MyDrive/Project13-Results/lab13_0d46777217_observations.csv \
  evaluation/ground_truth_frames/manifest.csv \
  --output evaluation/ground_truth_frames/review_candidates.csv
```

**Important:** model candidates are not ground truth. Each selected frame must be checked by a human. False positives must be removed, missed objects added, coordinates corrected when necessary, and `reviewed` changed to `1`.

### 3. Final human-reviewed file

Save the reviewed annotations as:

```text
evaluation/ground_truth.csv
```

Required columns:

```text
frame_index,class_name,x1,y1,x2,y2,track_id,reviewed,notes
```

### 4. Detection metrics

A standard-library evaluator is provided because the Colab run encountered a NumPy/Pandas ABI mismatch.

```bash
python evaluation/evaluate_detection_stdlib.py \
  /content/drive/MyDrive/Project13-Results/lab13_0d46777217_observations.csv \
  evaluation/ground_truth.csv \
  --iou 0.50 \
  --output /content/drive/MyDrive/Project13-Results/detection_evaluation.json
```

It reports:

- TP / FP / FN
- Precision
- Recall
- F1
- mean matched bounding-box IoU
- per-class metrics
- a detection confusion matrix

The evaluator refuses to score a ground-truth file containing rows that are not explicitly marked reviewed.

## Segmentation metrics

`src/metrics.py` implements mask IoU and Dice. These require human-reviewed ground-truth masks corresponding to SAM 3 outputs. Do not report mask IoU or Dice from box annotations.

## Tracking metrics

Tracking accuracy requires human-reviewed temporal identities. Candidate metrics include ID switches, fragmentation, missed tracks, and trajectory continuity. The current verified value of 10 tracker IDs is an execution statistic, not a tracking-accuracy score.

## Evidence rule

No Precision, Recall, F1, IoU, Dice, confusion-matrix interpretation, ID-switch count, or fragmentation result belongs in the portfolio until its corresponding human-reviewed ground truth exists.


## Visual review pack

Render the 94 model candidates over the 25 extracted review frames:

```bash
python evaluation/render_review_candidates.py \
  evaluation/ground_truth_frames \
  evaluation/ground_truth_frames/review_candidates.csv \
  --manifest evaluation/ground_truth_frames/manifest.csv \
  --output-dir evaluation/review_pack
```

This creates one annotated JPEG per review frame plus `evaluation/review_pack/index.html`.

The overlays are review aids only. They must not be treated as ground truth automatically.
