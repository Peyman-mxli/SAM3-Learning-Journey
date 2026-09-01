# Verified Run 01 — YOLO + ByteTrack + SAM 3

## Session

- **Session ID:** `lab13_1a429b993d`
- **Input:** `data/input/tracking_test_01.mp4`
- **Pipeline:** YOLO + Supervision + ByteTrack + SAM 3
- **Status:** Verified successful execution
- **Created at:** `2026-09-01T17:52:23.698987+00:00`

## Input video

- Frames: 75
- FPS: 15.0
- Resolution: 640 × 360
- Duration: 5.0 seconds

## Tracking results

- Total observations: **299**
- Unique tracker IDs: **10**
- Average confidence: **0.7059505884**

## SAM 3 results

SAM 3 was executed with the text prompt:

```text
person
```

The Project 13 pipeline associated SAM 3 segmentation outputs with tracked detections using bounding-box IoU matching.

Verified results:

- Observations with SAM 3 mask area: **21**
- Average SAM 3 mask area: **10047.095238095239 pixels**

Example verified mask-area observations:

| Frame | Tracker ID | Class | Confidence | Mask Area |
|---:|---:|---|---:|---:|
| 10 | 2 | person | 0.872754 | 13910 |
| 10 | 1 | person | 0.852898 | 10717 |
| 10 | 7 | person | 0.766690 | 5414 |
| 10 | 6 | person | 0.422235 | 5414 |
| 20 | 2 | person | 0.866052 | 12032 |
| 20 | 1 | person | 0.861385 | 10707 |
| 20 | 7 | person | 0.794226 | 6831 |
| 30 | 2 | person | 0.869998 | 14152 |
| 30 | 1 | person | 0.847406 | 10710 |
| 30 | 8 | person | 0.573026 | 5201 |

## What this verifies

This run demonstrates that Project 13 can:

- process a recorded video;
- detect objects with YOLO;
- convert detections to Supervision format;
- track objects through ByteTrack;
- preserve temporal tracker IDs;
- execute SAM 3 segmentation;
- associate SAM 3 masks with tracked detections;
- calculate mask area;
- persist mask-area evidence in SQLite;
- preserve confidence, geometry, timestamps, and class information.

## Remaining evaluation work

This run verifies execution, persistence, tracking, and SAM 3 mask measurement.

The following still require human-reviewed ground truth:

- Precision
- Recall
- F1
- detection confusion matrix
- mask IoU
- Dice score
- ID-switch counts
- tracker-fragmentation counts

These metrics must not be claimed until ground-truth annotations are available.
