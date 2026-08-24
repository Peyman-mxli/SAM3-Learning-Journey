# Practical — Advanced MaskAnnotator and SAM2

This folder contains the practical implementation for **Session 07 — Advanced MaskAnnotator and SAM2**.

The practical builds on the segmentation workflow from Session 06 and focuses on improving how segmentation masks are visualized, filtered, compared, and reused.

---

## Practical Objective

The objective is to build a reusable segmentation-visualization workflow that demonstrates:

- YOLOv8 object detection
- SAM 3 segmentation
- `sv.MaskAnnotator`
- Mask opacity control
- Bounding-box and mask composition
- Class filtering before segmentation
- Reusing the same pipeline with multiple images
- Preparing the conceptual transition toward temporal segmentation

---

## Main Workflow

```text
Input Image
     ↓
YOLOv8
     ↓
sv.Detections
     ↓
Optional Class Filtering
     ↓
Bounding-Box Prompts
     ↓
SAM 3
     ↓
Segmentation Masks
     ↓
MaskAnnotator
     ↓
BoxAnnotator
     ↓
Visualization / Analysis
```

---

## Practical Steps

The implementation should demonstrate:

1. Load the input image.
2. Load YOLOv8.
3. Load SAM 3.
4. Run YOLO object detection.
5. Convert detections into `sv.Detections`.
6. Use YOLO bounding boxes as SAM 3 prompts.
7. Generate segmentation masks.
8. Apply `MaskAnnotator`.
9. Compare bounding boxes with masks.
10. Experiment with different mask opacity values.
11. Filter detections to the `person` class before SAM.
12. Generate person-only segmentation masks.
13. Run the same pipeline on a second image.
14. Save visual outputs.
15. Document the transition toward temporal segmentation.

---

## MaskAnnotator

The main visualization component is:

```python
sv.MaskAnnotator()
```

Example:

```python
mask_annotator = sv.MaskAnnotator(
    opacity=0.6
)
```

The annotator overlays segmentation masks on the image:

```python
annotated = mask_annotator.annotate(
    scene=image.copy(),
    detections=sam_detections
)
```

---

## Mask Opacity Experiment

The practical compares:

```text
opacity = 0.2
opacity = 0.5
opacity = 0.9
```

This demonstrates the balance between:

```text
Mask Visibility
      ↕
Original Image Visibility
```

Low opacity preserves more of the original image.

High opacity emphasizes the segmentation regions.

---

## Bounding Box vs. Segmentation

The practical compares:

```text
YOLO Bounding Boxes
```

with:

```text
SAM 3 Pixel-Level Masks
```

Conceptually:

```text
Bounding Box
     ↓
Approximate rectangular region

Segmentation Mask
     ↓
Precise pixel-level object region
```

Both visualizations are useful, but they provide different kinds of spatial information.

---

## Filtering Before SAM

The practical filters YOLO detections before sending prompts to SAM.

Example:

```python
persons = detections[
    detections.class_id == 0
]
```

This creates the workflow:

```text
YOLO Detections
      ↓
Person Filter
      ↓
Person Bounding Boxes
      ↓
SAM 3
      ↓
Person Segmentation Masks
```

Filtering before segmentation avoids running SAM on objects that are not needed.

---

## Multiple Input Images

The same pipeline is reused on more than one image.

Initial images:

```text
bus.jpg
zidane.jpg
```

The processing logic remains unchanged:

```text
Input Image
     ↓
YOLO
     ↓
Supervision
     ↓
SAM
     ↓
MaskAnnotator
```

This demonstrates that the pipeline is reusable and not tied to one specific image.

---

## Expected Outputs

The practical should produce visual evidence such as:

```text
assets/output/
├── bbox_vs_mask.png
├── opacity_comparison.png
├── person_only_segmentation.png
└── second_image_segmentation.png
```

These filenames may be adjusted if needed during implementation.

---

## Practical Structure

```text
practical/
├── README.md
├── advanced_mask_annotator.py
│
└── assets/
    ├── README.md
    ├── input/
    │   └── README.md
    └── output/
        └── README.md
```

After validation, the input and generated output files will also be stored in their respective asset directories.

---

## Technologies Used

- Python
- Ultralytics
- YOLOv8
- SAM 3
- Supervision
- OpenCV
- NumPy
- Matplotlib
- Google Colab

---

## SAM 3 Model

The large SAM 3 checkpoint should remain outside the GitHub repository.

The validated Colab environment can use:

```text
/content/drive/MyDrive/SAM3-Models/sam3.pt
```

The practical should verify that the checkpoint exists before attempting SAM 3 inference.

---

## Transition to SAM2

The final concept introduced in this lesson is temporal segmentation.

The current practical focuses on image segmentation:

```text
Image
  ↓
SAM
  ↓
Mask
```

The next conceptual step is:

```text
Video
  ↓
Initial Object Mask
  ↓
Temporal Memory
  ↓
Future Frames
  ↓
Mask Propagation
```

This introduces the reason SAM2 is important for video workflows.

---

## Learning Outcome

After completing this practical, I will understand how to move beyond simply generating segmentation masks and instead build a configurable visualization pipeline around them.

The practical also demonstrates how filtering, segmentation, and visualization can be combined into a reusable Computer Vision workflow and prepares the foundation for temporal segmentation in video.
