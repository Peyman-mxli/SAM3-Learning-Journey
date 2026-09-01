# Source Code

## `pipeline.py`

Recorded-video processing entry point.

Current implemented path:

```text
Video
  ↓
YOLO
  ↓
Supervision Detections
  ↓
ByteTrack
  ↓
Persistent Tracker IDs
  ↓
SQLite observations
```

The SAM 3 integration is kept as an explicit adapter boundary so segmentation evidence is not falsely claimed before a Project 13 SAM 3 run is performed.

## `database.py`

Creates and manages:

- sessions
- observations
- historical summaries

The persistence layer stores frame-level tracking evidence for later analytics and dashboard exploration.
