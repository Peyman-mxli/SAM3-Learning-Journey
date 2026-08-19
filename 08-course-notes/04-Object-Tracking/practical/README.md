# Object Tracking — Practical Exercises

This directory contains the practical work associated with the **Object Tracking** section of the SAM3 Computer Vision Learning Journey.

The practical exercises extend object detection from individual images into video-based workflows where objects can be detected, assigned tracking IDs, and followed across consecutive frames.

---

## Practical Focus

The practical work focuses on combining:

- Ultralytics YOLO
- Supervision
- ByteTrack
- OpenCV
- Video processing
- Detection filtering
- Tracking IDs
- Tracking annotations
- Object trajectories
- Basic tracking analytics

The main progression is:

```text
Video
  ↓
Frames
  ↓
YOLO Detection
  ↓
sv.Detections
  ↓
Detection Filtering
  ↓
ByteTrack
  ↓
tracker_id
  ↓
Tracking Annotations
  ↓
Tracking Analytics
  ↓
Processed Video
