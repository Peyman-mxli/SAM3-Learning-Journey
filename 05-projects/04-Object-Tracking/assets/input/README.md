# Input Assets

This directory contains the input files used by the **Object Tracking with YOLO, Supervision, and ByteTrack** project.

---

## Input Video

The main input file for this project is:

```text
vehicles.mp4
```

The expected project path is:

```text
assets/input/vehicles.mp4
```

---

## Purpose

The video is used to test the complete object tracking pipeline:

```text
Input Video
     ↓
YOLO Detection
     ↓
sv.Detections
     ↓
Car Filtering
     ↓
ByteTrack
     ↓
Tracker IDs
     ↓
Tracking Analytics
     ↓
Annotations
     ↓
Output Video
```

---

## Tracked Object Class

This project is configured to track:

```text
car
```

Using the COCO class ID:

```python
TARGET_CLASS_ID = 2
```

This means YOLO may detect multiple object classes, but only detections with:

```python
class_id == 2
```

are passed to ByteTrack.

---

## Video Requirements

The input video should:

- Use a format supported by OpenCV
- Contain visible moving vehicles
- Include multiple consecutive frames
- Have a valid frame rate
- Preferably contain several cars moving through the scene

The project uses:

```text
MP4
```

as the primary video format.

---

## File Name

The Python project expects the exact path:

```python
INPUT_VIDEO = "assets/input/vehicles.mp4"
```

Therefore, the recommended file name is:

```text
vehicles.mp4
```

---

## Why Video Is Required

Object detection can operate on a single image.

Object tracking requires a sequence of frames.

For example:

```text
Frame 1 → car #1
Frame 2 → car #1
Frame 3 → car #1
Frame 4 → car #1
```

ByteTrack uses information across consecutive frames to maintain the identity of the detected object.

---

## Input and Output Separation

Original input files belong in:

```text
assets/input/
```

Generated tracking results belong in:

```text
assets/output/
```

This keeps source files separate from generated results.

---

## Directory Structure

```text
assets/
│
├── README.md
│
├── input/
│   ├── README.md
│   └── vehicles.mp4
│
└── output/
```

After running the project, the output directory will contain:

```text
tracked_vehicles.mp4
```

---

## Important

Do not overwrite the original:

```text
vehicles.mp4
```

with the processed tracking video.

The original video should remain unchanged so the tracking pipeline can be tested repeatedly.

---

## Project

This input belongs to:

```text
05-projects/04-Object-Tracking/
```

and is processed by:

```text
object_tracking_pipeline.py
```

---

## Author

**Peyman Miyandashti**

SAM3 Computer Vision Learning Journey

- GitHub: [Peyman-mxli](https://github.com/Peyman-mxli)
- LinkedIn: [Peyman Miyandashti](https://www.linkedin.com/in/peyman-mxli/)
