# Object Tracking — Practical Exercises

This directory contains the practical work associated with the **Object Tracking** section of the SAM3 Computer Vision Learning Journey.

The practical exercises extend object detection from individual images to video-based workflows, where objects are detected, assigned persistent tracking IDs, and followed across consecutive frames.

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

---

## Learning Objectives

Through this practical exercise, we will learn how to:

- Load and process a video with OpenCV
- Read a video frame by frame
- Run YOLO object detection on each frame
- Convert YOLO predictions into `sv.Detections`
- Filter detections using confidence thresholds
- Pass detections to ByteTrack
- Assign persistent `tracker_id` values
- Display class names and tracking IDs
- Draw bounding boxes around tracked objects
- Visualize object trajectories
- Count and analyze tracked objects
- Export the processed frames as a new video

---

## Main Workflow

The complete tracking pipeline is:

```text
Input Video
    ↓
Read Video Frames
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
Object Trajectories
    ↓
Tracking Analytics
    ↓
Processed Video
