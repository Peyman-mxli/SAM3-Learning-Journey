# Trackers — Multi-Object Tracking for Supervision Detections

`trackers` is Roboflow's model-agnostic multi-object tracking package. It accepts `supervision.Detections` and assigns persistent IDs across video frames using algorithms such as SORT, ByteTrack, OC-SORT, BoT-SORT, and C-BIoU.

## Resource Summary

| Item | Details |
|---|---|
| Package | <https://pypi.org/project/trackers/> |
| Documentation | <https://trackers.roboflow.com/> |
| Source | <https://github.com/roboflow/trackers> |
| Python | 3.10 or newer |
| License | Apache 2.0 |
| Course association | Session 05 · `02_b_tracking_objetos` |

## Why Tracking Is Needed

A detector processes frames independently. Without tracking, the same person or vehicle has no persistent identity.

```text
Frame 1 detections ─┐
Frame 2 detections ─┼─ Association + motion model → tracker IDs
Frame 3 detections ─┘
```

Tracker IDs make it possible to calculate trajectories, line crossings, dwell time, per-object mask area, speed estimates, and historical records.

## Important Migration

The course notebooks may use:

```python
tracker = sv.ByteTrack()
tracked = tracker.update_with_detections(detections)
```

Current Supervision documentation marks `sv.ByteTrack` deprecated beginning in Supervision 0.28.0 and directs users to the separate package:

```python
from trackers import ByteTrackTracker

tracker = ByteTrackTracker()
tracked = tracker.update(detections)
```

The method name changes from `update_with_detections()` to `update()`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install trackers supervision
```

Google Colab:

```python
!pip install -q trackers supervision ultralytics
```

Verify:

```bash
python -c "from trackers import ByteTrackTracker; print(ByteTrackTracker)"
```

## Standard Video Pattern

```python
import cv2
import supervision as sv
from trackers import ByteTrackTracker
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
tracker = ByteTrackTracker()

capture = cv2.VideoCapture("vehicles.mp4")
while capture.isOpened():
    ok, frame = capture.read()
    if not ok:
        break

    result = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)
    tracked = tracker.update(detections)

capture.release()
```

Create the tracker once before the frame loop. Recreating it for each frame erases its history.

## ByteTrack Concept

ByteTrack associates high-confidence detections first and then considers lower-confidence candidates to recover objects that might otherwise disappear during partial occlusion. Low-confidence candidates are not automatically accepted as new reliable objects; they assist association.

## Available Algorithms

| Algorithm | General characteristic |
|---|---|
| SORT | Kalman filter and Hungarian matching baseline |
| ByteTrack | Two-stage association using high and low confidence detections |
| OC-SORT | Observation-centric recovery and motion handling |
| BoT-SORT | Appearance/motion strategy with camera-motion support |
| C-BIoU | Association designed around buffered IoU concepts |

Algorithm choice should be evaluated on the target video rather than selected only by popularity.

## Tracking Inputs

Tracking quality depends on:

- Detection coordinates
- Detection confidence
- Frame rate
- Object motion
- Occlusion duration
- Camera movement
- Detector consistency
- Tracker parameters

A tracker cannot reliably recover objects the detector never observes for long periods.

## Labels and Traces

```python
labels = [
    f"#{tracker_id} {class_name}"
    for tracker_id, class_name in zip(
        tracked.tracker_id,
        tracked.data["class_name"],
    )
]

box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
trace_annotator = sv.TraceAnnotator(color_lookup=sv.ColorLookup.TRACK)
label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK)
```

Tracker-based color lookup helps keep each identity visually consistent.

## Tracking Is Not Identification

A `tracker_id` is temporary and local to one tracking session. It does not prove a person's identity and should not be treated as biometric identification.

## Evaluation

| Metric | Purpose |
|---|---|
| HOTA | Balances detection and association quality |
| IDF1 | Measures identity consistency |
| MOTA | Combines misses, false positives, and identity switches |
| ID switches | Counts incorrect identity changes |
| Track fragmentation | Counts broken trajectories |

Visual inspection is useful, but a professional evaluation uses labeled sequences and tracking metrics.

## Common Problems

| Problem | Cause | Resolution |
|---|---|---|
| Every frame gets new IDs | Tracker is recreated inside loop | Instantiate once |
| IDs switch during overlap | Occlusion or weak association | Tune tracker and improve detection consistency |
| Objects never activate | Activation threshold is too high | Adjust threshold using validation data |
| False tracks appear | Detector threshold is too low | Improve detection filtering and tracker settings |
| Trace has no history | Missing tracker IDs or recreated annotator | Preserve tracker and trace state |
| `sv.ByteTrack` warning | Deprecated Supervision wrapper | Migrate to `ByteTrackTracker` |

## Professional Practices

- Record tracker name, package version, thresholds, frame rate, and detector settings.
- Reset tracking state between unrelated videos.
- Do not compare tracker IDs across separate runs.
- Use representative labeled sequences for tuning.
- Preserve source timestamps when they matter.
- Separate tracking accuracy from detection accuracy.

## Official References

- PyPI package: <https://pypi.org/project/trackers/>
- Documentation: <https://trackers.roboflow.com/>
- Source repository: <https://github.com/roboflow/trackers>
- Supervision migration notice: <https://supervision.roboflow.com/develop/trackers/>
- ByteTrack paper: <https://arxiv.org/abs/2110.06864>
