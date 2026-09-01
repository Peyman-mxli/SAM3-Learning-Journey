# Verified Persistent Run — Project 13

## Session

- **Session ID:** `lab13_0d46777217`
- **Pipeline:** YOLO + ByteTrack + SAM 3
- **Input:** `data/input/tracking_test_01.mp4`
- **Status:** Verified successful persistent execution
- **Persistence:** SQLite database and exported evidence saved to Google Drive

## Verified metrics

- Total observations: **299**
- Unique tracker IDs: **10**
- Average confidence: **0.7059505883865931**
- SAM 3 mask observations: **21**
- Average SAM 3 mask area: **10047.095238095239 pixels**

## Persistent artifacts

Generated in:

```text
/content/drive/MyDrive/Project13-Results/
```

Files:

| Artifact | Size |
|---|---:|
| `lab13_0d46777217_observations.csv` | 53,469 bytes |
| `lab13_0d46777217_tracker_summary.csv` | 598 bytes |
| `lab13_0d46777217_run_summary.json` | 379 bytes |
| `project13_sam3.sqlite3` | 57,344 bytes |

## What this run verifies

This persistent run confirms that Project 13 can:

- process the real test video;
- detect objects with YOLO;
- track temporal identities with ByteTrack;
- run SAM 3 text-prompt segmentation for `person`;
- associate segmentation masks with tracked detections;
- calculate and persist mask area;
- store structured observations in SQLite;
- export observation-level CSV evidence;
- export tracker-level summary evidence;
- generate a machine-readable JSON run summary;
- preserve the database and reports outside the ephemeral Colab runtime.

## Reproducibility note

The first successful SAM 3 run used session `lab13_1a429b993d`.
After the Colab runtime disconnected, the pipeline was rerun with persistent Google Drive storage.

The definitive persistent session is:

```text
lab13_0d46777217
```

Both successful executions produced the same verified aggregate results:

- 299 observations
- 10 unique tracker IDs
- average confidence 0.7059505883865931
- 21 observations with SAM 3 mask area
- average SAM 3 mask area 10047.095238095239 pixels

## Evaluation boundary

These are verified execution and measurement results. Precision, Recall, F1, confusion matrices, mask IoU, Dice, ID switches, and fragmentation must still be calculated from human-reviewed ground truth before being claimed as evaluation metrics.
