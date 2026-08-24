# Session 07 — Class Recording

## Advanced MaskAnnotator and SAM2

This file contains the class recording for **Session 07 — Advanced MaskAnnotator and SAM2** from my SAM3 Computer Vision Learning Journey.

---

## Class Recording

**YouTube:**

[MaskAnnotator avanzado y SAM2](https://youtu.be/GNwQl-hy8Yw)

---

## Topics Covered

The class covers:

- Advanced segmentation-mask visualization
- `sv.MaskAnnotator`
- Mask opacity configuration
- Comparison of bounding boxes and segmentation masks
- Combining `MaskAnnotator` and `BoxAnnotator`
- Filtering detections before SAM segmentation
- Person-only segmentation
- YOLOv8 + SAM integration
- Reusing the segmentation pipeline with different images
- Introduction to SAM2
- Temporal segmentation concepts
- Memory-based segmentation
- Mask propagation across video frames

---

## Session Workflow

```text
Input Image
     ↓
YOLOv8
     ↓
Object Detection
     ↓
sv.Detections
     ↓
Optional Class Filtering
     ↓
Bounding-Box Prompts
     ↓
SAM
     ↓
Segmentation Masks
     ↓
MaskAnnotator
     ↓
Visualization
```

The session then introduces the conceptual transition toward video:

```text
Static Image Segmentation
          ↓
Initial Object Mask
          ↓
Temporal Memory
          ↓
Future Video Frames
          ↓
Mask Propagation
```

---

## Related Files

Main session documentation:

[README.md](./README.md)

Original class notebook:

[03_b_sam_mask_annotator.ipynb](./03_b_sam_mask_annotator.ipynb)

Practical implementation:

[practical/](./practical/)

---

## Learning Journey

This recording documents the progression from basic SAM segmentation introduced in Session 06 toward more advanced mask visualization, filtering, reusable segmentation pipelines, and temporal segmentation concepts.
