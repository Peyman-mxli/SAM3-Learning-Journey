# Object Tracking — Input Assets

This directory contains the input media files used by the practical exercises in the **Object Tracking** section of the SAM3 Computer Vision Learning Journey.

The files stored here serve as the original source material for testing and demonstrating object-tracking workflows.

---

## Main Input Video

The main video used by the practical exercise is:

```text
tracking_demo.mp4
```

The video is approximately **10 seconds long** and runs at **30 FPS**.

It contains multiple moving synthetic objects designed specifically for learning and testing object-tracking concepts.

---

## Purpose

The input video is used to demonstrate how objects can be followed across consecutive video frames.

The practical exercise uses the video to explore:

- Frame-by-frame video processing
- Object detections
- `sv.Detections`
- ByteTrack
- Persistent tracking IDs
- `tracker_id`
- Tracking annotations
- Object trajectories
- Basic tracking analytics

---

## Input Directory Structure

```text
input/
├── README.md
└── tracking_demo.mp4
```

---

## Video Information

```text
Filename: tracking_demo.mp4
Duration: 10 seconds
Frame Rate: 30 FPS
Total Frames: 300
Resolution: 960 × 540
Format: MP4
```

---

## Practical Workflow

The video acts as the starting point of the tracking pipeline:

```text
tracking_demo.mp4
        ↓
Read Video
        ↓
Extract Frames
        ↓
Object Detections
        ↓
sv.Detections
        ↓
ByteTrack
        ↓
tracker_id
        ↓
Tracking Annotations
        ↓
Object Trajectories
        ↓
Tracking Analytics
        ↓
Output Video
```

---

## Synthetic Video

`tracking_demo.mp4` is a synthetic demonstration video created specifically for this practical exercise.

It contains several moving objects following different paths through the scene.

Some objects move in opposite directions and cross paths, allowing the practical exercise to demonstrate how a tracking system attempts to maintain object identities between consecutive frames.

---

## Important Note

The objects in this video are synthetic and are not intended to represent standard real-world YOLO classes.

For this reason, the initial practical exercise focuses primarily on:

```text
Detections
    ↓
Supervision
    ↓
ByteTrack
    ↓
tracker_id
```

rather than relying on YOLO to recognize the synthetic objects.

A later real-world video exercise can extend the pipeline to:

```text
Video
  ↓
YOLO
  ↓
sv.Detections
  ↓
ByteTrack
  ↓
tracker_id
  ↓
Tracking Annotations
```

---

## File Usage

The practical Python script will read the video from:

```text
assets/input/tracking_demo.mp4
```

The original input video should remain unchanged.

Processed and annotated videos will be stored separately inside:

```text
assets/output/
```

This keeps the original source video separate from generated results.

---

## Related Practical Script

The input video will be processed by:

```text
object_tracking_practical.py
```

located in the main `practical/` directory.

---

## Related Course Material

This input asset supports the **Object Tracking** practical exercises and complements the concept notes and class recording.

[Watch the Object Tracking class recording on YouTube](https://youtu.be/UXN0l33NqF4)

---

## Repository

[View the SAM3 Learning Journey on GitHub](https://github.com/Peyman-mxli/SAM3-Learning-Journey)

---

## Author

**Peyman Miyandashti**

- [GitHub](https://github.com/Peyman-mxli)
- [LinkedIn](https://www.linkedin.com/in/peyman-mxli/)
