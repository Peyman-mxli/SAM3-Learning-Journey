# Project 08 Results

## Overview

Project 08 — Advanced Mask Analysis Pipeline was executed successfully using two input images.

The pipeline completed the full workflow:

```text
Input Image
     ↓
YOLOv8 Detection
     ↓
Detection Filtering
     ↓
SAM 3 Segmentation
     ↓
Mask Analysis
     ↓
MaskAnnotator
     ↓
BoxAnnotator
     ↓
Annotated Images
     +
JSON Results
     +
CSV Results
```

No runtime errors occurred during the validated execution.

---

## Execution Summary

| Metric | Result |
|---|---:|
| Images processed | 2 |
| Total objects analyzed | 9 |
| Annotated images generated | 2 |
| JSON result files generated | 2 |
| CSV result files generated | 2 |

---

## Image 1 — bus.jpg

The `bus.jpg` image was processed successfully.

### Detection Results

| Metric | Result |
|---|---:|
| YOLOv8 detections | 6 |
| Detections after filtering | 6 |
| SAM 3 masks generated | 6 |
| Objects analyzed | 6 |

The YOLOv8 model detected:

- 4 persons
- 1 bus
- 1 stop sign

Every YOLOv8 detection was passed to SAM 3 as a bounding-box prompt, and SAM 3 generated one segmentation mask for each detected object.

### Generated Files

```text
data/output/bus_analyzed.png
results/json/bus_analysis.json
results/csv/bus_analysis.csv
```

---

## Image 2 — zidane.jpg

The `zidane.jpg` image was processed successfully.

### Detection Results

| Metric | Result |
|---|---:|
| YOLOv8 detections | 3 |
| Detections after filtering | 3 |
| SAM 3 masks generated | 3 |
| Objects analyzed | 3 |

The YOLOv8 model detected:

- 2 persons
- 1 tie

All three detections were segmented successfully by SAM 3.

### Generated Files

```text
data/output/zidane_analyzed.png
results/json/zidane_analysis.json
results/csv/zidane_analysis.csv
```

---

## Multi-Image Validation

The same reusable pipeline processed both input images without modifying the implementation.

```text
bus.jpg ───────┐
               ↓
       Advanced Mask Analysis Pipeline
               ↓
       Annotated Image + JSON + CSV


zidane.jpg ────┐
               ↓
       Advanced Mask Analysis Pipeline
               ↓
       Annotated Image + JSON + CSV
```

This confirms that the implementation is reusable across multiple images.

---

## Generated Outputs

### Visual Outputs

The pipeline created annotated images containing:

```text
Original Image
      +
SAM 3 Segmentation Masks
      +
YOLO Bounding Boxes
```

Files:

```text
data/output/bus_analyzed.png
data/output/zidane_analyzed.png
```

---

### JSON Outputs

Structured object-level analysis was saved as JSON.

Files:

```text
results/json/bus_analysis.json
results/json/zidane_analysis.json
```

Each JSON file contains:

- Image name
- Object count
- Class ID
- Class name
- Detection confidence
- Bounding box coordinates
- Bounding-box area
- Mask area
- Occupancy ratio

---

### CSV Outputs

Tabular analytical results were saved as CSV.

Files:

```text
results/csv/bus_analysis.csv
results/csv/zidane_analysis.csv
```

Each row represents one detected and segmented object.

Columns include:

```text
image
object_id
class_id
class_name
confidence
x1
y1
x2
y2
box_area
mask_area
occupancy_ratio
```

---

## Mask Analysis

For every segmented object, the pipeline calculated:

```text
Bounding-Box Area
Mask Area
Occupancy Ratio
```

The occupancy ratio is defined as:

```text
Occupancy Ratio = Mask Area / Bounding-Box Area
```

This measurement compares the rectangular YOLOv8 detection region with the pixel-level SAM 3 segmentation mask.

---

## Validation Checklist

- [x] Input images discovered automatically
- [x] YOLOv8 detection completed
- [x] Detection filtering stage executed
- [x] SAM 3 segmentation completed
- [x] Segmentation masks generated for every detection
- [x] Mask area calculated
- [x] Bounding-box area calculated
- [x] Occupancy ratio calculated
- [x] MaskAnnotator visualization created
- [x] BoxAnnotator visualization created
- [x] Annotated images saved
- [x] JSON results saved
- [x] CSV results saved
- [x] Multiple images processed successfully

---

## Final Result

Project 08 — Advanced Mask Analysis Pipeline successfully combined object detection, instance segmentation, visualization, and quantitative mask analysis into a reusable computer-vision workflow.

The validated execution processed **2 images** and analyzed **9 objects**, producing visual evidence and structured analytical outputs for every detected and segmented object.
