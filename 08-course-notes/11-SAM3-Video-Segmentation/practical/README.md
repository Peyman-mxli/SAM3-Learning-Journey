# Session 11 Practical — SAM 3 Video Segmentation

This practical section converts the original Session 11 notebook into a structured and reproducible video-segmentation workflow.

The source notebook is:

[05_b_segmentacion_sam_video.ipynb](../05_b_segmentacion_sam_video.ipynb)

---

## Video-Segmentation Pipelines

### Pipeline A — YOLO + ByteTrack + SAM 3

```text
Input Video
    ↓
YOLOv8 Detection
    ↓
ByteTrack IDs
    ↓
SAM 3 Masks
    ↓
Mask + Label + Trace Annotation
    ↓
Annotated Video
```

### Pipeline B — SAM 3 with Text Prompts

```text
Input Video
    ↓
Text Prompts
    ↓
SAM3VideoSemanticPredictor
    ↓
Semantic Masks
    ↓
Annotated Video
```

---

## Experiments

The notebook contains the following experiments:

1. Full YOLO + ByteTrack + SAM 3 video segmentation
2. Segmentation restricted to a polygon zone
3. Dynamic mask opacity based on detection confidence
4. Mask-area analysis for tracked objects
5. Direct semantic video segmentation with text prompts
6. Tracker-ID filtering challenge

---

## Video Processing Workflow

Each frame passes through a complete computer vision pipeline:

```text
Video Frame
     ↓
Object Detection
     ↓
Persistent Tracking
     ↓
Bounding-Box Prompts
     ↓
Pixel-Level Segmentation
     ↓
Attribute Transfer
     ↓
Visualization
     ↓
Output Video
```

The YOLO detections provide class IDs and confidence scores. ByteTrack adds persistent tracker IDs, while SAM 3 produces precise segmentation masks.

---

## Defined Files

The notebook uses the following paths:

```text
assets/
│
├── vehicles.mp4
├── vehicles_sam.mp4
├── vehicles_sam_zona.mp4
├── vehicles_sam_opacidad.mp4
├── vehicles_sam_areas.mp4
└── vehicles_texto.mp4
```

---

## Main Technologies

- Python
- SAM 3
- Ultralytics
- YOLOv8
- `SAM3VideoSemanticPredictor`
- Supervision
- ByteTrack
- OpenCV
- NumPy
- Matplotlib
- PyTorch

---

## Related Documentation

Main lesson documentation:

[README.md](../README.md)

Class recording:

[CLASS-RECORDING.md](../CLASS-RECORDING.md)

Original notebook:

[05_b_segmentacion_sam_video.ipynb](../05_b_segmentacion_sam_video.ipynb)
