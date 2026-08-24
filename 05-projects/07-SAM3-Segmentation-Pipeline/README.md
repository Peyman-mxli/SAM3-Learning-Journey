# Project 07 — SAM 3 Segmentation Pipeline

This project implements a complete **object detection and pixel-level segmentation pipeline** using **YOLOv8**, **Segment Anything Model 3 (SAM 3)**, **Supervision**, **OpenCV**, and **NumPy**.

The project builds on the concepts studied in **Session 06 — Segmentation with SAM** and combines them into a reusable Computer Vision application.

Unlike the smaller examples in `04-examples/06-Segmentation-with-SAM/`, this project integrates detection, segmentation, analysis, visualization, object extraction, and structured result export into a single pipeline.

---

## Project Objective

The objective is to build a reusable system capable of:

1. Loading an input image.
2. Detecting multiple objects using YOLOv8.
3. Converting YOLO predictions into Supervision detections.
4. Filtering detections using a confidence threshold.
5. Extracting bounding boxes from the detected objects.
6. Using those bounding boxes as prompts for SAM 3.
7. Generating pixel-level segmentation masks.
8. Analyzing each segmentation mask.
9. Comparing mask area with bounding-box area.
10. Extracting individual segmented objects.
11. Creating a complete annotated segmentation visualization.
12. Saving structured detection and segmentation information to JSON.
13. Preserving generated results inside a dedicated output directory.

---

# Pipeline Architecture

The complete project follows this workflow:

```text
Input Image
     ↓
YOLOv8
     ↓
Object Detection
     ↓
Confidence Filtering
     ↓
Bounding Boxes
     ↓
SAM 3 Prompts
     ↓
Pixel-Level Segmentation Masks
     ↓
Supervision Detections
     ↓
┌───────────────────────────────┐
│                               │
↓                               ↓
Mask Analysis             Visualization
│                               │
↓                               ↓
Area Measurements         Annotated Image
│
↓
Object Extraction
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
    │   └── README.md
    │
    └── output/
        └── README.md
```

After execution, the output directory will also contain generated project results.

Example:

```text
assets/output/
│
├── annotated_segmentation.png
├── segmentation_results.json
├── masks/
│   ├── object_00_mask.png
│   ├── object_01_mask.png
│   └── ...
│
└── extracted_objects/
    ├── object_00.png
    ├── object_01.png
    └── ...
```

Generated output files are created by the pipeline and should not replace the original input media.

---

# Input

The project accepts an image stored inside:

```text
assets/input/
```

The initial validation image can be:

```text
bus.jpg
```

This image is useful because it contains multiple detectable objects and allows the complete multi-object segmentation pipeline to be evaluated.

---

# Stage 1 — Image Loading

OpenCV loads the source image from the input directory.

```python
image = cv2.imread(str(INPUT_IMAGE))
```

The pipeline verifies that the image exists before continuing.

Conceptually:

```text
Input File
    ↓
OpenCV
    ↓
Image Array
```

---

# Stage 2 — YOLOv8 Detection

YOLOv8 performs the initial object detection.

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

This provides a consistent representation for later processing.

---

# Stage 3 — Confidence Filtering

Not every model prediction should automatically become a SAM 3 prompt.

The project therefore applies a configurable confidence threshold.

Conceptually:

```text
All YOLO Detections
        ↓
Confidence Threshold
        ↓
Accepted Detections
```

This reduces low-confidence detections before segmentation.

The threshold is configured inside the project rather than being hard-coded throughout the pipeline.

---

# Stage 4 — Bounding-Box Prompts

The accepted YOLO bounding boxes are extracted from:

```python
detections.xyxy
```

Each bounding box becomes a SAM 3 prompt.

Example:

```text
Object 0
[x1, y1, x2, y2]

Object 1
[x1, y1, x2, y2]

Object 2
[x1, y1, x2, y2]
```

These prompts connect the object detector with the segmentation model.

---

# Stage 5 — SAM 3 Segmentation

SAM 3 processes the input image using the YOLO bounding boxes.

```text
Image
  +
Bounding Boxes
      ↓
    SAM 3
      ↓
Segmentation Masks
```

Ideally, each accepted YOLO detection produces a corresponding SAM 3 mask.

The project verifies that masks are returned before continuing with analysis.

---

# Stage 6 — Mask Analysis

Each segmentation mask is analyzed independently.

Measurements include:

- Mask dimensions
- Object pixel count
- Total image pixels
- Image coverage
- Bounding-box area
- Mask area
- Mask-to-box percentage

For a boolean mask:

```python
mask_area = int(mask.sum())
```

because every `True` pixel represents one segmented object pixel.

---

# Mask Area vs Bounding-Box Area

Bounding-box area is calculated as:

```text
width × height
```

or:

```text
(x2 - x1) × (y2 - y1)
```

Mask coverage inside the bounding box can then be calculated using:

```text
Mask Area
────────────── × 100
Bounding Box Area
```

This measurement helps demonstrate the difference between rectangular detection and pixel-level segmentation.

A bounding box may contain substantial background, while the segmentation mask attempts to represent only the object itself.

---

# Stage 7 — Object Extraction

The project extracts every segmented object from the original image.

The core operation is:

```python
object_image[~mask] = 0
```

Conceptually:

```text
Original Image
      ↓
Segmentation Mask
      ↓
Remove Background
      ↓
Extracted Object
```

Each detected object is saved separately.

This provides direct visual evidence of the segmentation result.

---

# Stage 8 — Mask Export

Each segmentation mask is also exported as an image.

This allows the masks to be inspected independently from the original image.

Example:

```text
object_00_mask.png
object_01_mask.png
object_02_mask.png
```

These images provide a simple visual representation of the pixel-level predictions generated by SAM 3.

---

# Stage 9 — Complete Visualization

The project creates a final visualization containing the detected and segmented objects.

The visualization may include:

- Segmentation masks
- Bounding boxes
- Class labels
- Confidence scores
- Object indices

The objective is to create a single output image that summarizes the complete inference result.

```text
Original Image
      ↓
Masks
      ↓
Bounding Boxes
      ↓
Labels
      ↓
Final Annotated Segmentation
```

---

# Stage 10 — Structured JSON Export

The project saves structured information about every processed object.

The JSON output contains information such as:

```json
{
    "input_image": "bus.jpg",
    "objects": [
        {
            "object_index": 0,
            "class_id": 5,
            "class_name": "bus",
            "confidence": 0.87,
            "bounding_box": [
                22.87,
                231.27,
                805.00,
                756.84
            ],
            "mask_area_pixels": 265686,
            "bounding_box_area_pixels": 411059.31,
            "mask_to_box_percentage": 64.63
        }
    ]
}
```

The exact values depend on the input image and model inference.

---

# Output Organization

Generated files are separated from source media.

```text
assets/
│
├── input/
│   └── Original media
│
└── output/
    └── Generated results
```

This prevents generated files from overwriting the original input.

It also improves:

- Reproducibility
- Organization
- Debugging
- Documentation
- Result comparison

---

# SAM 3 Model

The SAM 3 checkpoint is intentionally **not stored inside this GitHub repository** because the model file is very large.

The validated course environment stores the checkpoint externally in Google Drive:

```text
/content/drive/MyDrive/SAM3-Models/sam3.pt
```

Approximate model size:

```text
3.21 GB
```

The project will allow the model path to be configured rather than requiring the checkpoint to exist inside the repository.

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

# Expected Outputs

A successful project execution should demonstrate:

- YOLOv8 object detection
- Confidence filtering
- Multiple detected objects
- SAM 3 bounding-box prompting
- Multiple segmentation masks
- Boolean mask analysis
- Mask-area calculations
- Bounding-box-area calculations
- Mask-to-box comparisons
- Individual mask exports
- Individual object extractions
- Complete segmentation visualization
- Structured JSON results

---

# Validation Plan

The project will be validated step by step.

## 1. Input Validation

Verify:

```text
Input image exists
Image loads correctly
Image dimensions are valid
```

## 2. Detection Validation

Verify:

```text
YOLO model loads
Detections are generated
Confidence filtering works
Bounding boxes are available
```

## 3. SAM 3 Validation

Verify:

```text
SAM 3 checkpoint exists
SAM 3 loads successfully
Bounding-box prompts are accepted
Segmentation masks are generated
```

## 4. Mask Validation

Verify:

```text
Masks are boolean arrays
Mask dimensions match the input image
Object pixels can be counted
Mask areas can be calculated
```

## 5. Export Validation

Verify:

```text
Annotated image is generated
Individual masks are generated
Extracted objects are generated
JSON output is generated
```

## 6. Result Validation

Verify that:

```text
Number of accepted detections
        ≈
Number of generated masks
```

and that the generated files correctly correspond to the detected objects.

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
Reusable Segmentation Pipeline
```

The project therefore represents the integration stage of the Session 06 learning process.

---

# Learning Outcomes

After completing this project, I will be able to:

- Connect an object detector with a segmentation model.
- Use YOLO bounding boxes as SAM 3 prompts.
- Generate multiple pixel-level segmentation masks.
- Work directly with boolean NumPy masks.
- Measure segmented object geometry.
- Compare segmentation masks with bounding boxes.
- Extract individual objects from images.
- Generate reusable Computer Vision outputs.
- Export structured segmentation metadata.
- Organize a complete segmentation application.
- Validate a multi-stage Computer Vision pipeline.

---

# Project Status

```text
Project 07 — SAM 3 Segmentation Pipeline

Project structure       🔄 In Progress
Input handling          ⏳ Pending
YOLO detection          ⏳ Pending
Confidence filtering    ⏳ Pending
SAM 3 integration       ⏳ Pending
Mask analysis           ⏳ Pending
Object extraction       ⏳ Pending
Mask export             ⏳ Pending
Visualization           ⏳ Pending
JSON export             ⏳ Pending
Final validation        ⏳ Pending
Documentation           🔄 In Progress
```

The status will be updated as each component is implemented and validated.

---

## Author

**Peyman Miyandashti**

SAM3 Computer Vision Learning Journey

Python · YOLOv8 · SAM 3 · Supervision · OpenCV · NumPy · Computer Vision
