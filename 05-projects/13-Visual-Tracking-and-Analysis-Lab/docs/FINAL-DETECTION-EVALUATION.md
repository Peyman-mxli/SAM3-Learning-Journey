# Final Detection Evaluation — Project 13

## Scope

This evaluation uses the definitive persistent Project 13 run:

```text
lab13_0d46777217
```

Only the **25 human-reviewed frames** are included in scoring.

- Reviewed frames: **25**
- Predictions on reviewed frames: **94**
- Human-reviewed ground-truth objects: **92**
- IoU match threshold: **0.50**

Predictions from the remaining unreviewed video frames are excluded from false-positive counts.

## Overall results

| Metric | Result |
|---|---:|
| True Positives | **79** |
| False Positives | **15** |
| False Negatives | **13** |
| Precision | **0.8404** |
| Recall | **0.8587** |
| F1 | **0.8495** |
| Mean matched box IoU | **0.9945** |

## Per-class results

### Person

| Metric | Result |
|---|---:|
| TP | 65 |
| FP | 1 |
| FN | 4 |
| Precision | **0.9848** |
| Recall | **0.9420** |
| F1 | **0.9630** |

### Bus

| Metric | Result |
|---|---:|
| TP | 14 |
| FP | 0 |
| FN | 9 |
| Precision | **1.0000** |
| Recall | **0.6087** |
| F1 | **0.7568** |

### Car

| Metric | Result |
|---|---:|
| TP | 0 |
| FP | 14 |
| FN | 0 |
| Precision | **0.0000** |
| Recall | **0.0000** |
| F1 | **0.0000** |

The reviewed ground truth contained no true `car` objects in the evaluation subset. The 14 `car` predictions therefore count as false positives. During manual review, several large vehicle detections were corrected to `bus`.

## Detection confusion matrix

Rows are ground truth / background. Columns are predicted object / background.

```text
                         Predicted object   Background
Ground-truth object            79              13
Background                     15               0
```

This corresponds to:

- 79 matched detections
- 13 missed ground-truth objects
- 15 unmatched predictions

## Interpretation

The detector performs very strongly on `person` in this reviewed sample, with high precision and recall.

The `bus` class has perfect precision but lower recall, meaning that when the model predicts a bus it is correct in this sample, but it misses several buses.

The main class-confusion issue observed during review is `bus` versus `car`.

## Important boundary

These results are detection metrics for the 25 reviewed frames only.

They do not yet constitute:

- mask IoU
- Dice score
- tracking ID-switch accuracy
- tracking fragmentation accuracy

Those require corresponding human-reviewed segmentation masks or temporal tracking ground truth.
