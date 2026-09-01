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


## Mouse-based Colab reviewer

For human ground-truth review, install the lightweight annotation widget:

```bash
pip install -q -r evaluation/requirements-reviewer.txt
```

In Google Colab, enable the custom widget manager and launch the reviewer:

```python
from google.colab import output
output.enable_custom_widget_manager()

from evaluation.colab_ground_truth_reviewer import launch

reviewer = launch(
    frames_dir="evaluation/ground_truth_frames",
    candidates_csv="evaluation/ground_truth_frames/review_candidates.csv",
    manifest_csv="evaluation/ground_truth_frames/manifest.csv",
    output_csv="evaluation/ground_truth.csv",
    backup_csv="/content/drive/MyDrive/Project13-Results/ground_truth.csv",
)
```

The reviewer supports direct mouse drawing, moving, resizing, relabeling, and deleting of boxes. It autosaves reviewed annotations and writes a persistent Drive backup.

The underlying annotation widget supports mouse creation/editing of bounding boxes and programmatic model-assisted starting boxes.


### Reviewed-frame scope

Only predictions from human-reviewed frame indices are scored. Predictions from the other video frames are excluded from FP counts because those frames do not have reviewed ground truth.


## Controlled 5-video validation suite

Project 13 now includes a deterministic robustness suite that expands the single
recorded source into five controlled test videos:

- baseline
- low light
- partial occlusion
- motion blur
- reduced scale

Generate and run the suite:

```bash
python evaluation/create_condition_variants.py data/input/tracking_test_01.mp4
python evaluation/run_validation_suite.py
python evaluation/analyze_validation_conditions.py
```

The analysis compares observation count, tracker count, average confidence,
average track length, and SAM-mask coverage against the baseline.

These are **robustness/sensitivity indicators**, not ground-truth tracking
accuracy.

## Tracking identity evaluation

Human-reviewed temporal identities can be evaluated with:

```bash
python evaluation/evaluate_tracking_csv.py \
  results/predictions_tracking.csv \
  evaluation/tracking_ground_truth.csv \
  --output results/tracking_evaluation.json
```

Template:

`evaluation/tracking_ground_truth_template.csv`

The evaluator reports matched detections, missed ground-truth objects,
unmatched predictions, ID switches, ID-switch rate, and fragmented tracks.

## Segmentation evaluation

Human-reviewed binary masks can be evaluated with:

```bash
python evaluation/evaluate_segmentation_masks.py \
  evaluation/segmentation_manifest.csv \
  --output results/segmentation_evaluation.csv
```

Template:

`evaluation/segmentation_manifest_template.csv`

The evaluator calculates per-sample and mean **mask IoU** and **Dice**.

See the complete proposal audit in:

`docs/FINAL-COMPLETION-AUDIT.md`
