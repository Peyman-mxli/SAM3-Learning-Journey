# Evaluation

Project 13 separates **implementation** from **evidence**.

## Detection evaluation

Use `evaluate_detection_csv.py` with two CSV files:

1. Project 13 predictions.
2. Human-reviewed ground truth.

Required columns:

```text
frame_index
class_name
x1
y1
x2
y2
```

Example:

```bash
python evaluation/evaluate_detection_csv.py \
  results/predictions.csv \
  evaluation/ground_truth.csv \
  --iou 0.50 \
  --output results/detection_evaluation.json
```

The script reports:

- true positives
- false positives
- false negatives
- Precision
- Recall
- F1
- mean IoU for matched boxes

## Segmentation evaluation

`src/metrics.py` includes:

- mask IoU
- Dice score

Project-specific segmentation metrics must only be generated after SAM 3 masks and corresponding ground-truth masks are available.

## Tracking evaluation

For the MVP, preserve and inspect:

- unique tracker IDs
- tracker duration
- trajectory length
- fragmentation
- ID switches
- missed tracks

Do not invent ID-switch or fragmentation counts without human-reviewed temporal ground truth.
