# Session 11 — Class Recording

## SAM 3 Video Segmentation

This file contains the class recording for **Session 11 — SAM 3 Video Segmentation** from my SAM3 Computer Vision Learning Journey.

---

## Class Recording

**YouTube:**

[SAM 3 en Video — Segmentación y Tracking](https://youtu.be/_EuNGCYS35k)

---

## Topics Covered

The class covers:

- Frame-by-frame video segmentation
- YOLOv8 object detection
- ByteTrack persistent object IDs
- SAM 3 bounding-box prompting
- Attribute transfer from YOLO detections to SAM detections
- Mask, label, and trace visualization with Supervision
- Polygon-zone filtering before segmentation
- Dynamic mask opacity based on confidence
- Mask-area analysis through time
- `SAM3VideoSemanticPredictor`
- Direct semantic video prompts
- Comparison of YOLO + SAM 3 and SAM 3 with text
- Streaming video inference
- OpenCV video output

---

## Main Workflows

### Option A — YOLO + ByteTrack + SAM 3

```text
Video Frame
     ↓
YOLO
     ↓
ByteTrack
     ↓
Bounding-Box Prompts
     ↓
SAM 3
     ↓
Masks + Labels + Traces
```

### Option B — SAM 3 with Text

```text
Video
  ↓
Text Prompts
  ↓
SAM3VideoSemanticPredictor
  ↓
Detection + Segmentation + Tracking
```

---

## Related Files

Main session documentation:

[README.md](./README.md)

Original class notebook:

[05_b_segmentacion_sam_video.ipynb](./05_b_segmentacion_sam_video.ipynb)

Practical documentation:

[practical/README.md](./practical/README.md)

---

## Learning Journey

This recording documents the progression from static-image segmentation and prompting toward complete temporal video-segmentation workflows using object detection, tracking, precise masks, and semantic text prompts.
