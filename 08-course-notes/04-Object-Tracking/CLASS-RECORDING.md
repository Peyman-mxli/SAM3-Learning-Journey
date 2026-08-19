# Object Tracking — Class Recording

This file documents the class recording associated with the **Object Tracking** section of my SAM3 Computer Vision Learning Journey.

---

## Class Recording

The class recording is available on YouTube:

[Watch the Object Tracking class recording on YouTube](https://youtu.be/UXN0l33NqF4)

The recording supports the course material documented in this section and can be used together with the concept notes, practical exercises, code examples, and project implementations.

---

## Lesson

**Topic:** Object Tracking

This section documents the transition from processing independent object detections to tracking objects across consecutive video frames while maintaining their identities.

The lesson connects:

- YOLO
- Supervision
- ByteTrack
- `sv.Detections`
- `tracker_id`
- Detection filtering
- Video processing
- Tracking annotations
- Object trajectories
- Tracking analytics

---

## Main Workflow

The object-tracking workflow can be represented as:

```text
Input Video
    ↓
Video Frames
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
Tracking Annotation
    ↓
Object Trajectories
    ↓
Tracking Analytics
    ↓
Processed Video
