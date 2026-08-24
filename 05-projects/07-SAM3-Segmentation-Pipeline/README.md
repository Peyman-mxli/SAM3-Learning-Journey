# Project 07 — SAM 3 Segmentation Pipeline

This project implements a complete **multi-object detection and pixel-level segmentation pipeline** using **YOLOv8**, **Segment Anything Model 3 (SAM 3)**, **Supervision**, **OpenCV**, and **NumPy**.

The project builds on the concepts studied in **Session 06 — Segmentation with SAM** and combines them into a reusable Computer Vision application.

Unlike the smaller examples in `04-examples/06-Segmentation-with-SAM/`, this project integrates detection, confidence filtering, segmentation, mask analysis, visualization, object extraction, and structured result export into a single validated pipeline.

---

# Project Objective

The objective is to build a reusable system capable of:

1. Loading a custom input image.
2. Detecting multiple objects using YOLOv8.
3. Converting YOLO predictions into Supervision detections.
4. Filtering detections using a confidence threshold.
5. Extracting bounding boxes from accepted detections.
6. Using YOLO bounding boxes as prompts for SAM 3.
7. Generating pixel-level segmentation masks.
8. Analyzing each segmentation mask.
9. Comparing mask area with bounding-box area.
10. Extracting individual segmented objects.
11. Exporting individual segmentation masks.
12. Creating a complete annotated segmentation visualization.
13. Saving structured detection and segmentation information to JSON.
14. Preserving representative visual evidence inside the repository.

---

# Pipeline Architecture

The complete project follows this workflow:

```text
Custom Input Image
        ↓
      YOLOv8
        ↓
 Object Detection
        ↓
Confidence Filtering
        ↓
 Accepted Objects
        ↓
 Bounding Boxes
        ↓
 SAM 3 Prompts
        ↓
Pixel-Level Masks
        ↓
Supervision Detections
        ↓
┌───────────────────────────────┐
│                               │
↓                               ↓
Mask Analysis              Visualization
│                               │
↓                               ↓
Area Measurements         Annotated Image
│
↓
Object Extraction
│
↓
Mask Export
│
↓
Structured Results
│
↓
JSON Export
```

---

# Why YOLO + SAM 3?

YOLOv8 and SAM 3 perform different but complementary tasks.

## YOLOv8

YOLOv8 identifies:

```text
What is the object?
Where is the object?
```

Its primary spatial representation is a bounding box:

```text
[x1, y1, x2, y2]
```

A bounding box provides an approximate rectangular region containing the object.

---

## SAM 3

SAM 3 receives a spatial prompt and identifies the pixels belonging to the object.

Conceptually:

```text
YOLO Bounding Box
       ↓
SAM 3 Prompt
       ↓
Pixel-Level Mask
```

Instead of representing an object only as a rectangle, SAM 3 produces a boolean mask describing its actual shape.

```text
True  → pixel belongs to the object
False → pixel belongs to the background
```

This makes it possible to move from approximate object localization to precise pixel-level segmentation.

---

# Project Structure

```text
07-SAM3-Segmentation-Pipeline/
│
├── README.md
├── requirements.txt
├── sam3_segmentation_pipeline.py
│
└── assets/
    │
    ├── README.md
    │
    ├── input/
    │   ├── README.md
    │   └── mexicali_bus_scene.png
    │
    └── output/
        │
        ├── README.md
        ├── annotated_segmentation.png
        ├── segmentation_results.json
        │
        ├── extracted_objects/
        │   ├── README.md
        │   ├── object_00.png
        │   ├── object_01.png
        │   ├── object_02.png
        │   ├── object_04.png
        │   ├── object_05.png
        │   └── object_17.png
        │
        └── masks/
            ├── README.md
            ├── object_00_mask.png
            ├── object_01_mask.png
            ├── object_02_mask.png
            ├── object_04_mask.png
            ├── object_05_mask.png
            └── object_17_mask.png
```

The complete execution generated 25 masks and 25 extracted objects.

A representative subset is preserved in the repository to provide visual evidence without unnecessarily storing every generated image.

---

# Input

The project uses a custom validation scene stored at:

```text
assets/input/mexicali_bus_scene.png
```

The image contains multiple objects at different positions and scales, including pedestrians, vehicles, and smaller objects.

Validated image resolution:

```text
1536 × 1024
```

This provides a more demanding test than the original `bus.jpg` image used during the smaller Session 06 exercises.

The custom scene allows the complete Project 07 pipeline to be evaluated independently.

---

# Stage 1 — Image Loading

OpenCV loads the custom source image from the input directory.

```python
image = cv2.imread(str(INPUT_IMAGE))
```

The pipeline verifies that the image exists and can be read before continuing.

Conceptually:

```text
mexicali_bus_scene.png
        ↓
      OpenCV
        ↓
   Image Array
```

---

# Stage 2 — YOLOv8 Detection

YOLOv8 performs the initial multi-object detection.

```text
Image
  ↓
YOLOv8
  ↓
Bounding Boxes
  ↓
Classes
  ↓
Confidence Scores
```

The predictions are converted into:

```python
sv.Detections
```

using Supervision.

This provides a consistent representation for later filtering, segmentation, annotation, and analysis.

During the validated Project 07 run:

```text
Raw YOLO detections: 25
```

---

# Stage 3 — Confidence Filtering

Not every model prediction should automatically become a SAM 3 prompt.

The project therefore applies a configurable confidence threshold:

```text
0.25
```

Conceptually:

```text
All YOLO Detections
        ↓
Confidence ≥ 0.25
        ↓
Accepted Detections
```

Validated result:

```text
Raw YOLO detections: 25
Accepted detections: 25
```

All 25 detections passed the configured threshold during the validated run.

---

# Stage 4 — Bounding-Box Prompts

The accepted YOLO bounding boxes are extracted from:

```python
detections.xyxy
```

Each bounding box becomes a spatial prompt for SAM 3.

Conceptually:

```text
Accepted YOLO Detection
          ↓
[x1, y1, x2, y2]
          ↓
    SAM 3 Prompt
```

This stage connects object detection with pixel-level segmentation.

During validation:

```text
25 accepted detections
        ↓
25 bounding-box prompts
```

---

# Stage 5 — SAM 3 Segmentation

SAM 3 processes the same input image using the accepted YOLO bounding boxes.

```text
Image
  +
YOLO Bounding Boxes
        ↓
      SAM 3
        ↓
Segmentation Masks
```

Validated result:

```text
SAM masks generated: 25
Mask array shape: (25, 1024, 1536)
```

The mask dimensions correspond to the original image dimensions.

Most importantly:

```text
25 accepted YOLO detections
            ↓
25 SAM 3 segmentation masks
```

Every accepted detection produced a corresponding segmentation mask during the validated run.

---

# Stage 6 — Mask Analysis

Each segmentation mask is analyzed independently.

Measurements include:

- Object pixel count
- Total image pixels
- Image coverage
- Bounding-box area
- Segmentation-mask area
- Mask-to-box percentage

For a boolean mask:

```python
mask_area = int(mask.sum())
```

Every `True` pixel represents one pixel assigned to the segmented object.

The total image size is:

```text
1024 × 1536
=
1,572,864 pixels
```

This makes it possible to calculate the percentage of the complete image occupied by each segmented object.

---

# Mask Area vs. Bounding-Box Area

Bounding-box area is calculated as:

```text
width × height
```

or:

```text
(x2 - x1) × (y2 - y1)
```

The relationship between segmentation area and bounding-box area is calculated as:

```text
      Mask Area
──────────────────── × 100
Bounding Box Area
```

This measurement demonstrates an important difference between detection and segmentation.

A bounding box represents a rectangular region and therefore usually includes background pixels.

A segmentation mask attempts to preserve only pixels belonging to the actual object.

---

# Stage 7 — Object Extraction

The project extracts each segmented object from the original image.

The core operation is:

```python
object_image[~mask] = 0
```

Conceptually:

```text
Original Image
      +
SAM 3 Mask
      ↓
Pixel Selection
      ↓
Background Removed
      ↓
Extracted Object
```

Pixels outside the segmentation mask are set to black.

Pixels belonging to the segmented object remain unchanged.

The complete validated run generated:

```text
25 extracted objects
```

---

# Stage 8 — Mask Export

Every segmentation mask generated during execution is exported as an individual image.

Conceptually:

```text
Boolean Mask
     ↓
0 / 255 Image
     ↓
PNG Mask
```

In the exported images:

```text
White pixels → segmented object
Black pixels → background
```

The complete execution generated:

```text
25 mask images
```

A representative subset is stored in the repository.

---

# Stage 9 — Complete Visualization

The project creates a final visualization combining:

- SAM 3 segmentation masks
- YOLO bounding boxes
- Object classes
- Confidence scores
- Object indices

The result is stored at:

```text
assets/output/annotated_segmentation.png
```

<p align="center">
  <img src="assets/output/annotated_segmentation.png"
       alt="Project 07 SAM 3 annotated segmentation result"
       width="900">
</p>

The visualization provides a single overview of the complete inference result.

---

# Stage 10 — Structured JSON Export

The project saves structured information about every analyzed object.

The output file is:

```text
assets/output/segmentation_results.json
```

The JSON stores information including:

```text
Project name
Input image
Image width
Image height
Confidence threshold
Raw YOLO detection count
Accepted detection count
SAM mask count
Analyzed object count
Annotated output filename
```

For every object it also stores:

```text
Object index
Class ID
Class name
Confidence
Bounding-box coordinates
Bounding-box area
Mask area
Image coverage percentage
Mask-to-box percentage
Mask filename
Extracted-object filename
```

This makes the segmentation results machine-readable and allows later analysis without repeating model inference.

---

# Final Validation Results

Project 07 was successfully executed and validated using the complete YOLOv8 + SAM 3 pipeline.

## Validation Configuration

```text
Input image: mexicali_bus_scene.png
Image resolution: 1536 × 1024
YOLO model: YOLOv8n
Confidence threshold: 0.25
SAM model: SAM 3
SAM checkpoint: sam3.pt
```

## Detection and Segmentation Results

```text
Raw YOLO detections:       25
Accepted detections:       25
SAM 3 masks generated:     25
Objects analyzed:          25
```

The validated workflow therefore produced:

```text
25 YOLO detections
        ↓
25 accepted detections
        ↓
25 bounding-box prompts
        ↓
25 SAM 3 masks
        ↓
25 analyzed objects
        ↓
25 masks exported
        ↓
25 objects extracted
```

This confirms a complete one-to-one relationship between the accepted detections and generated segmentation masks during the validation run.

---

# Detected Object Classes

The custom scene produced detections from multiple object categories, including:

- Person
- Bus
- Car
- Traffic light
- Backpack
- Handbag

This makes the scene useful for evaluating segmentation across:

- Different object classes
- Large and small objects
- Foreground and background objects
- Different image positions
- Different object shapes
- Different object scales

---

# Bus Segmentation Result

The bus produced one of the strongest and most visually useful results in the validation scene.

```text
Class: bus
YOLO confidence: 0.9149
Mask area: 395,899 pixels
Image coverage: 25.17%
Mask / bounding-box area: 75.64%
```

Conceptually:

```text
YOLO Detection
      ↓
Bus Bounding Box
      ↓
SAM 3 Prompt
      ↓
Pixel-Level Bus Mask
      ↓
Extracted Bus
```

The result demonstrates why segmentation provides more precise spatial information than object detection alone.

A bounding box contains both object and background pixels, while the SAM 3 mask follows the actual visible object region much more closely.

---

# Representative Extracted Objects

The complete execution generated 25 extracted objects.

A representative subset is stored inside:

```text
assets/output/extracted_objects/
```

The selected outputs are:

```text
object_00.png
object_01.png
object_02.png
object_04.png
object_05.png
object_17.png
```

These examples provide visual evidence across different object positions, classes, and scales.

---

## Object 00

<p align="center">
  <img src="assets/output/extracted_objects/object_00.png"
       alt="Project 07 extracted object 00"
       width="650">
</p>

---

## Object 01

<p align="center">
  <img src="assets/output/extracted_objects/object_01.png"
       alt="Project 07 extracted object 01"
       width="650">
</p>

---

## Object 02 — Bus

<p align="center">
  <img src="assets/output/extracted_objects/object_02.png"
       alt="Project 07 extracted bus"
       width="650">
</p>

---

## Object 04

<p align="center">
  <img src="assets/output/extracted_objects/object_04.png"
       alt="Project 07 extracted object 04"
       width="650">
</p>

---

## Object 05

<p align="center">
  <img src="assets/output/extracted_objects/object_05.png"
       alt="Project 07 extracted object 05"
       width="650">
</p>

---

## Object 17

<p align="center">
  <img src="assets/output/extracted_objects/object_17.png"
       alt="Project 07 extracted object 17"
       width="650">
</p>

---

# Representative SAM 3 Masks

The corresponding segmentation masks are stored inside:

```text
assets/output/masks/
```

The selected masks are:

```text
object_00_mask.png
object_01_mask.png
object_02_mask.png
object_04_mask.png
object_05_mask.png
object_17_mask.png
```

---

## Object 00 Mask

<p align="center">
  <img src="assets/output/masks/object_00_mask.png"
       alt="Project 07 SAM 3 mask object 00"
       width="650">
</p>

---

## Object 01 Mask

<p align="center">
  <img src="assets/output/masks/object_01_mask.png"
       alt="Project 07 SAM 3 mask object 01"
       width="650">
</p>

---

## Object 02 Mask — Bus

<p align="center">
  <img src="assets/output/masks/object_02_mask.png"
       alt="Project 07 SAM 3 bus segmentation mask"
       width="650">
</p>

---

## Object 04 Mask

<p align="center">
  <img src="assets/output/masks/object_04_mask.png"
       alt="Project 07 SAM 3 mask object 04"
       width="650">
</p>

---

## Object 05 Mask

<p align="center">
  <img src="assets/output/masks/object_05_mask.png"
       alt="Project 07 SAM 3 mask object 05"
       width="650">
</p>

---

## Object 17 Mask

<p align="center">
  <img src="assets/output/masks/object_17_mask.png"
       alt="Project 07 SAM 3 mask object 17"
       width="650">
</p>

---

# Representative Mask-to-Object Relationship

Each stored mask corresponds directly to an extracted object.

```text
object_00_mask.png → object_00.png
object_01_mask.png → object_01.png
object_02_mask.png → object_02.png
object_04_mask.png → object_04.png
object_05_mask.png → object_05.png
object_17_mask.png → object_17.png
```

The relationship is:

```text
Original Image
      +
SAM 3 Mask
      ↓
Pixel Selection
      ↓
Extracted Object
```

This provides direct visual evidence that the exported segmentation masks are being used to isolate objects from the source image.

---

# Output Organization

Generated files are separated from the source media.

```text
assets/
│
├── input/
│   └── mexicali_bus_scene.png
│
└── output/
    ├── annotated_segmentation.png
    ├── segmentation_results.json
    ├── extracted_objects/
    └── masks/
```

This prevents generated files from overwriting the original input.

It also improves:

- Reproducibility
- Organization
- Debugging
- Documentation
- Result inspection
- Result comparison

---

# SAM 3 Model

The SAM 3 checkpoint is intentionally **not stored inside this GitHub repository** because the model file is very large.

The validated Google Colab environment stores the checkpoint externally in Google Drive:

```text
/content/drive/MyDrive/SAM3-Models/sam3.pt
```

Approximate model size:

```text
3.21 GB
```

The project verifies that the checkpoint exists before attempting segmentation.

---

# Technologies

The project uses:

- Python
- Ultralytics
- YOLOv8
- SAM 3
- Supervision
- OpenCV
- NumPy
- JSON
- pathlib
- Google Colab
- NVIDIA T4 GPU

---

# Validation

The project was validated end-to-end rather than only checking individual code sections.

## 1. Input Validation

Validated:

```text
Input image exists                 ✅
Image loads correctly              ✅
Image dimensions are valid         ✅
```

---

## 2. Detection Validation

Validated:

```text
YOLOv8 model loads                 ✅
Detections are generated           ✅
Confidence filtering works         ✅
Bounding boxes are available       ✅
```

Result:

```text
25 raw detections
25 accepted detections
```

---

## 3. SAM 3 Validation

Validated:

```text
SAM 3 checkpoint exists            ✅
SAM 3 loads successfully           ✅
Bounding-box prompts accepted      ✅
Segmentation masks generated       ✅
```

Result:

```text
25 prompts
25 segmentation masks
```

---

## 4. Mask Validation

Validated:

```text
Masks are available                ✅
Mask dimensions match image        ✅
Object pixels can be counted       ✅
Mask areas can be calculated       ✅
Image coverage can be calculated   ✅
Mask/box ratio can be calculated   ✅
```

Mask array:

```text
(25, 1024, 1536)
```

---

## 5. Export Validation

Validated:

```text
Annotated image generated          ✅
Individual masks generated         ✅
Extracted objects generated        ✅
JSON output generated              ✅
```

Complete execution produced:

```text
1 annotated visualization
1 structured JSON file
25 mask images
25 extracted object images
```

---

## 6. Result Validation

The key relationship was verified:

```text
Accepted YOLO detections
          =
Generated SAM 3 masks
          =
Analyzed objects
```

Validated values:

```text
25 = 25 = 25
```

This confirms that every accepted YOLO detection was successfully passed through the SAM 3 segmentation stage during this execution.

---

# Relationship to Session 06

This project is based directly on the concepts studied in:

```text
08-course-notes/
└── 06-Segmentation-with-SAM/
```

and the reusable examples inside:

```text
04-examples/
└── 06-Segmentation-with-SAM/
```

The learning progression is:

```text
Session 06 Course Material
          ↓
Six Small Examples
          ↓
Validated Practical
          ↓
Project 07
          ↓
Custom Multi-Object Scene
          ↓
Reusable Segmentation Pipeline
```

Project 07 therefore represents the integration stage of the Session 06 learning process.

It is not simply a copy of the Session 06 practical.

The project introduces:

- A different custom input scene
- A complete reusable pipeline
- Confidence filtering
- Multi-object analysis
- Individual mask export
- Individual object extraction
- Structured result storage
- Representative GitHub evidence
- End-to-end validation

---

# Learning Outcomes

After completing this project, I can:

- Connect an object detector with a segmentation model.
- Use YOLO bounding boxes as SAM 3 prompts.
- Filter detections before segmentation.
- Generate multiple pixel-level segmentation masks.
- Work directly with boolean NumPy masks.
- Measure segmented object geometry.
- Calculate image coverage.
- Compare segmentation masks with bounding boxes.
- Extract individual objects from images.
- Export segmentation masks as images.
- Generate complete annotated visualizations.
- Export structured segmentation metadata.
- Organize a reusable Computer Vision application.
- Validate a multi-stage Computer Vision pipeline.
- Preserve representative visual evidence for reproducibility and documentation.

---

# Final Outcome

Project 07 successfully demonstrates an end-to-end multi-object segmentation system:

```text
mexicali_bus_scene.png
        ↓
      YOLOv8
        ↓
25 Raw Detections
        ↓
Confidence Filtering
        ↓
25 Accepted Detections
        ↓
25 Bounding-Box Prompts
        ↓
      SAM 3
        ↓
25 Pixel-Level Masks
        ↓
Mask Analysis
        ↓
25 Mask Exports
        ↓
25 Object Extractions
        ↓
Annotated Visualization
        ↓
Structured JSON Results
```

The project moves beyond the isolated Session 06 examples by integrating detection, filtering, segmentation, geometric analysis, visualization, object extraction, mask export, and structured result storage into one reproducible pipeline.

---

# Project Status

```text
Project 07 — SAM 3 Segmentation Pipeline

Project structure       ✅ Completed
Input handling          ✅ Completed
YOLO detection          ✅ Completed
Confidence filtering    ✅ Completed
SAM 3 integration       ✅ Completed
Mask analysis           ✅ Completed
Object extraction       ✅ Completed
Mask export             ✅ Completed
Visualization           ✅ Completed
JSON export             ✅ Completed
Final validation        ✅ Completed
Documentation           ✅ Completed
```

**Project 07 — SAM 3 Segmentation Pipeline is complete and validated.** ✅

---

## Author

**Peyman Miyandashti**

SAM3 Computer Vision Learning Journey

Python · YOLOv8 · SAM 3 · Supervision · OpenCV · NumPy · Computer Vision
