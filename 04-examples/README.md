# SAM3 — Code Examples

This directory contains clean, runnable Python examples based on the practical concepts covered throughout my **SAM3 Computer Vision Learning Journey**.

Unlike the detailed course notes, these files focus on small, reusable code examples that can be executed and studied independently.

The examples progressively build from basic AI-assisted programming and object detection toward filtering, tracking, spatial analysis, occupancy monitoring, object counting, and pixel-level segmentation with SAM 3.

---

## Available Examples

### 00 — Agentic AI Programming

[`00-Agentic-AI-Programming/`](./00-Agentic-AI-Programming/)

Examples covering:

- AI-assisted code generation
- Prompt engineering for code
- OpenCV image processing
- AI-assisted debugging
- Error analysis and refinement

---

### 01 — Introduction to Supervision

[`01-Introduction-to-Supervision/`](./01-Introduction-to-Supervision/)

Examples covering:

- OpenCV image loading
- YOLOv8 object detection
- `sv.Detections`
- Bounding boxes
- Confidence scores
- Confidence thresholds
- Supervision annotations
- YOLO model comparison
- Custom-image detection
- JSON prediction export

---

### 02 — Annotation and Visualization

[`02-Annotation-and-Visualization/`](./02-Annotation-and-Visualization/)

Examples covering:

- `BoxAnnotator`
- `RoundBoxAnnotator`
- `HaloAnnotator`
- `BlurAnnotator`
- `BoxCornerAnnotator`
- `LabelAnnotator`
- `DotAnnotator`
- `EllipseAnnotator`
- Bounding box and label composition
- Annotation colors and color palettes
- Bounding box thickness
- Label text scale
- Annotation layer order
- Custom multi-annotator visualization pipelines

The examples demonstrate how the same YOLO detection results can be presented using different visualization techniques and how multiple Supervision Annotators can be composed as visual layers.

---

### 03 — Filtering and Manipulating Detections

[`03-Filtering-and-Manipulating-Detections/`](./03-Filtering-and-Manipulating-Detections/)

Examples covering:

- Confidence filtering
- Boolean masks
- Class filtering
- Multiple filtering conditions
- Class exclusion
- Bounding-box area
- Size filtering
- Detection merging
- Non-Maximum Suppression
- Confidence sorting
- Top-N detection selection
- Bounding-box center calculations
- Spatial filtering

These examples demonstrate how raw YOLO predictions can be transformed into application-specific detections using **Supervision** and **NumPy**.

The filtering workflow can be represented as:

```text
Raw Detections
      ↓
Confidence Filtering
      ↓
Class Filtering
      ↓
Size Filtering
      ↓
NMS
      ↓
Top-N Selection
      ↓
Spatial Filtering
      ↓
Application-Specific Detections
```

---

### 04 — Object Tracking

[`04-Object-Tracking/`](./04-Object-Tracking/)

Examples covering:

- ByteTrack
- Persistent tracker IDs
- Tracking objects across video frames
- Tracker ID visualization
- Class names with tracker IDs
- Detection filtering before tracking
- Frame-based tracking analysis
- Unique object counting
- Object visibility duration
- Complete object-tracking pipelines

Object tracking extends detection by maintaining the identity of objects across multiple frames.

The basic workflow is:

```text
Video Frame
     ↓
YOLO
     ↓
Detections
     ↓
ByteTrack
     ↓
Persistent Tracker IDs
     ↓
Annotation
     ↓
Tracked Video
```

Instead of treating every detection independently, tracking allows the system to understand that:

```text
Frame 100 → ID 4
Frame 101 → ID 4
Frame 102 → ID 4
Frame 103 → ID 4
```

represent the same object moving through time.

---

### 05 — Zones and Counting

[`05-Zones-and-Counting/`](./05-Zones-and-Counting/)

Examples covering:

- Polygon coordinates
- `PolygonZone`
- `PolygonZoneAnnotator`
- Polygon occupancy
- `zone.trigger()`
- `zone.current_count`
- Boolean zone masks
- Spatial detection filtering
- Tracking with PolygonZone
- `LineZone`
- `LineZoneAnnotator`
- Directional crossing detection
- `line_zone.in_count`
- `line_zone.out_count`
- Tracking with LineZone
- PolygonZone and LineZone composition
- Occupancy vs flow
- Complete spatial video analytics pipelines

This lesson extends object tracking with spatial reasoning.

The two main questions are:

```text
PolygonZone:
How many objects are inside this area RIGHT NOW?

LineZone:
How many objects have CROSSED this line IN TOTAL?
```

The complete workflow becomes:

```text
Video
  ↓
YOLOv8
  ↓
Detections
  ↓
ByteTrack
  ↓
Persistent Tracker IDs
  ↓
┌─────────────────────────┐
│                         │
↓                         ↓
PolygonZone            LineZone
↓                         ↓
Occupancy                Flow
│                         │
└────────────┬────────────┘
             ↓
         Visualization
             ↓
         Output Video
```

---

### 06 — Segmentation with SAM

[`06-Segmentation-with-SAM/`](./06-Segmentation-with-SAM/)

Examples covering:

- YOLOv8 object detection
- Bounding-box extraction
- YOLO bounding boxes as SAM 3 prompts
- SAM 3 segmentation
- Pixel-level boolean masks
- Supervision detections
- Mask inspection
- Object pixel counting
- Image coverage calculations
- Pixel-level object extraction
- Mask area vs bounding-box area
- Base64 mask serialization
- JSON mask storage
- Mask reconstruction and validation

The examples demonstrate how object detection can be extended from rectangular bounding boxes to precise pixel-level representations.

The basic segmentation workflow is:

```text
Input Image
     ↓
YOLOv8
     ↓
Object Detections
     ↓
Bounding Boxes
     ↓
SAM 3 Prompts
     ↓
Segmentation Masks
     ↓
Mask Analysis
     ↓
Object Extraction / Serialization
```

All six Session 06 examples were successfully tested in Google Colab using an NVIDIA T4 GPU.

Validated results included:

```text
Input image: bus.jpg
Image shape: (1080, 810, 3)

YOLO detections: 6
SAM masks generated: 6
Mask array shape: (6, 1080, 810)

First mask object pixels: 265686
First mask image coverage: 30.37%

Decoded mask matches original: True
```

---

## Repository Organization

The SAM3 Learning Journey separates different types of material:

```text
03-notebooks/
    Original Google Colab / Jupyter notebooks

04-examples/
    Small runnable Python examples

05-projects/
    Larger practical projects

08-course-notes/
    Detailed class notes and explanations

09-assets/
    Images, banners, and supporting resources
```

---

## Example Workflow

Many of the computer vision examples begin with this architecture:

```text
Input
  ↓
Python
  ↓
OpenCV
  ↓
AI Model
  ↓
Predictions
  ↓
Supervision
  ↓
Processing / Analysis
  ↓
Visualization / Output
```

As the course progresses, the workflow becomes increasingly sophisticated.

### Detection Workflow

```text
Input Image
    ↓
YOLO
    ↓
Detection Results
    ↓
sv.Detections
    ↓
Supervision Annotators
    ↓
Visualization
```

### Filtering Workflow

```text
Input Image
    ↓
YOLO
    ↓
sv.Detections
    ↓
Confidence Filtering
    ↓
Class Filtering
    ↓
Size Filtering
    ↓
NMS
    ↓
Spatial Filtering
    ↓
Application-Specific Detections
```

### Tracking Workflow

```text
Input Video
    ↓
YOLO
    ↓
Detections
    ↓
ByteTrack
    ↓
Persistent IDs
    ↓
Tracking Analysis
    ↓
Annotated Video
```

### Spatial Analytics Workflow

```text
Input Video
    ↓
YOLO
    ↓
Detections
    ↓
ByteTrack
    ↓
Tracked Objects
    ↓
Spatial Zones
    ↓
Occupancy + Flow
    ↓
Annotated Video
```

### Segmentation Workflow

```text
Input Image
    ↓
YOLOv8
    ↓
Detections
    ↓
Bounding Boxes
    ↓
SAM 3
    ↓
Pixel-Level Masks
    ↓
Mask Analysis
    ↓
Object Extraction / Serialization
```

---

## Technologies

Examples in this directory may use:

- Python
- OpenCV
- NumPy
- Matplotlib
- Ultralytics YOLOv8
- SAM 3
- Supervision
- ByteTrack
- Base64
- JSON
- Google Colab

---

## Example Directory Structure

```text
04-examples/
│
├── 00-Agentic-AI-Programming/
│
├── 01-Introduction-to-Supervision/
│
├── 02-Annotation-and-Visualization/
│   ├── README.md
│   ├── 01_box_and_label.py
│   ├── 02_compare_annotators.py
│   ├── 03_customize_visualization.py
│   ├── 04_layer_order.py
│   └── 05_custom_composition.py
│
├── 03-Filtering-and-Manipulating-Detections/
│   ├── README.md
│   ├── 01_confidence_filtering.py
│   ├── 02_class_and_boolean_filtering.py
│   ├── 03_size_filtering.py
│   ├── 04_nms_and_top_n.py
│   └── 05_spatial_filtering.py
│
├── 04-Object-Tracking/
│   ├── README.md
│   ├── 01-basic-bytetrack.py
│   ├── 02-tracker-ids-across-frames.py
│   ├── 03-track-video-with-annotations.py
│   ├── 04-class-and-tracker-id.py
│   ├── 05-filter-before-tracking.py
│   ├── 06-tracking-frame-count.py
│   ├── 07-unique-object-count.py
│   ├── 08-visible-time.py
│   └── 09-complete-tracking-pipeline.py
│
├── 05-Zones-and-Counting/
│   ├── README.md
│   ├── 01-basic-polygon-zone.py
│   ├── 02-polygon-zone-current-count.py
│   ├── 03-filter-detections-inside-zone.py
│   ├── 04-tracking-with-polygon-zone.py
│   ├── 05-basic-line-zone.py
│   ├── 06-line-zone-crossing-count.py
│   ├── 07-tracking-with-line-zone.py
│   ├── 08-polygon-and-line-zone.py
│   └── 09-complete-zones-counting-pipeline.py
│
├── 06-Segmentation-with-SAM/
│   ├── README.md
│   ├── bus.jpg
│   ├── 01_yolo_detection.py
│   ├── 02_sam_bbox_segmentation.py
│   ├── 03_mask_inspection.py
│   ├── 04_object_extraction.py
│   ├── 05_mask_area_comparison.py
│   └── 06_mask_serialization.py
│
└── README.md
```

---

# 02 — Annotation and Visualization Examples

These examples focus on transforming raw object detections into clear visual results.

## Basic Box and Label

```text
01_box_and_label.py
```

Demonstrates the standard visualization pipeline:

```text
Image
  ↓
YOLO
  ↓
Detections
  ↓
BoxAnnotator
  ↓
LabelAnnotator
  ↓
Annotated Image
```

---

## Annotator Comparison

```text
02_compare_annotators.py
```

Compares:

```python
sv.BoxAnnotator()
sv.RoundBoxAnnotator()
sv.HaloAnnotator()
sv.BlurAnnotator()
sv.BoxCornerAnnotator()
```

---

## Visualization Customization

```text
03_customize_visualization.py
```

Experiments with:

```python
color=
thickness=
text_scale=
```

and:

```python
sv.Color.RED
sv.Color.GREEN
sv.ColorPalette.DEFAULT
```

---

## Layer Order

```text
04_layer_order.py
```

Demonstrates why annotation order matters:

```text
Box → Label
```

versus:

```text
Label → Box
```

The Annotator applied last appears visually on top of the previous layers.

---

## Custom Composition

```text
05_custom_composition.py
```

Combines multiple Annotators:

```text
Original Image
      ↓
EllipseAnnotator
      ↓
DotAnnotator
      ↓
LabelAnnotator
      ↓
Custom Visualization
```

---

# 03 — Filtering and Manipulating Detections Examples

These examples focus on selecting, removing, ranking, and manipulating object detections after model inference.

## Confidence Filtering

```text
01_confidence_filtering.py
```

Demonstrates how a Boolean confidence mask can remove low-confidence predictions:

```python
high_confidence = detections[
    detections.confidence > 0.50
]
```

The basic concept is:

```text
All Detections
      ↓
Confidence > Threshold
      ↓
High-Confidence Detections
```

---

## Class and Boolean Filtering

```text
02_class_and_boolean_filtering.py
```

Demonstrates how detections can be selected according to class:

```python
detections.class_id == 0
```

Multiple conditions can also be combined:

```python
(
    detections.class_id == 0
)
&
(
    detections.confidence > 0.60
)
```

This example also demonstrates class exclusion using:

```python
detections.class_id != 0
```

---

## Size Filtering

```text
03_size_filtering.py
```

Demonstrates how bounding-box area can be used to remove small detections:

```python
large_detections = detections[
    detections.area > 5000
]
```

The example also inspects:

```text
Minimum Area
Maximum Area
Average Area
```

before applying the size filter.

---

## NMS and Top-N Selection

```text
04_nms_and_top_n.py
```

Demonstrates how multiple detection collections can be merged:

```python
merged = sv.Detections.merge([
    detections_low,
    detections_high
])
```

Non-Maximum Suppression is then applied:

```python
after_nms = merged.with_nms(
    threshold=0.50
)
```

Finally, NumPy sorts the detections by confidence:

```python
indices_top = np.argsort(
    after_nms.confidence
)[::-1][:TOP_N]
```

This produces a workflow such as:

```text
Detection Set A
        +
Detection Set B
        ↓
      Merge
        ↓
       NMS
        ↓
Confidence Sorting
        ↓
      Top-N
```

---

## Spatial Filtering

```text
05_spatial_filtering.py
```

Demonstrates how bounding-box coordinates can be used to filter detections according to their location.

The horizontal center of each bounding box is calculated with:

```python
centers_x = (
    detections.xyxy[:, 0]
    + detections.xyxy[:, 2]
) / 2
```

The image midpoint is calculated with:

```python
image_midpoint = image.shape[1] / 2
```

The final filter keeps detections located in the right half:

```python
right_side_detections = detections[
    centers_x > image_midpoint
]
```

This introduces the idea of **region-based detection filtering**.

---

# 04 — Object Tracking Examples

The Object Tracking examples extend detection from individual frames into temporal analysis.

The examples progress through:

```text
Basic ByteTrack
      ↓
Persistent IDs
      ↓
Video Annotation
      ↓
Class + Tracker ID
      ↓
Filtering Before Tracking
      ↓
Frame Count Analysis
      ↓
Unique Object Count
      ↓
Visibility Duration
      ↓
Complete Tracking Pipeline
```

Tracking introduces the idea that an object should maintain its identity across multiple frames.

Conceptually:

```text
Detection
    ↓
Tracking
    ↓
Persistent Identity
    ↓
Temporal Analysis
```

This provides the foundation required for the zone and counting examples that follow.

---

# 05 — Zones and Counting Examples

The Zones and Counting examples extend tracking with **spatial analysis**.

They progress through:

```text
01-basic-polygon-zone.py
        ↓
02-polygon-zone-current-count.py
        ↓
03-filter-detections-inside-zone.py
        ↓
04-tracking-with-polygon-zone.py
        ↓
05-basic-line-zone.py
        ↓
06-line-zone-crossing-count.py
        ↓
07-tracking-with-line-zone.py
        ↓
08-polygon-and-line-zone.py
        ↓
09-complete-zones-counting-pipeline.py
```

---

## Basic PolygonZone

```text
01-basic-polygon-zone.py
```

Introduces a polygonal region defined using pixel coordinates.

```text
Frame
  ↓
Polygon Coordinates
  ↓
PolygonZone
  ↓
Spatial Region
```

---

## PolygonZone Current Count

```text
02-polygon-zone-current-count.py
```

Introduces:

```python
zone.trigger(
    detections=detections
)

zone.current_count
```

This measures **instantaneous occupancy**.

```text
PolygonZone
     ↓
Objects Inside
     ↓
Current Occupancy
```

---

## Filtering Detections Inside a Zone

```text
03-filter-detections-inside-zone.py
```

Uses the Boolean mask returned by:

```python
zone.trigger(
    detections=detections
)
```

to keep only objects inside the polygon.

```text
All Detections
      ↓
PolygonZone
      ↓
Boolean Mask
      ↓
Spatial Filtering
      ↓
Objects Inside Zone
```

---

## Tracking with PolygonZone

```text
04-tracking-with-polygon-zone.py
```

Combines:

```text
YOLO
  ↓
Detections
  ↓
ByteTrack
  ↓
Persistent IDs
  ↓
PolygonZone
  ↓
Occupancy
```

This connects temporal tracking with spatial occupancy analysis.

---

## Basic LineZone

```text
05-basic-line-zone.py
```

Introduces a virtual counting boundary using:

```python
sv.Point()
sv.LineZone()
sv.LineZoneAnnotator()
```

Unlike a polygon, a line does not measure presence.

It measures **crossing events**.

---

## LineZone Crossing Count

```text
06-line-zone-crossing-count.py
```

Introduces:

```python
line_zone.trigger(
    detections=detections
)

line_zone.in_count
line_zone.out_count
```

The counters accumulate throughout the video.

```text
Tracked Object
      ↓
Crosses Boundary
      ↓
LineZone
      ↓
Directional Count
```

---

## Tracking with LineZone

```text
07-tracking-with-line-zone.py
```

Demonstrates why persistent tracker IDs are important for crossing detection.

```text
Frame 1 → ID 7
Frame 2 → ID 7
Frame 3 → ID 7
Frame 4 → ID 7 crosses line
                     ↓
                  Count Event
```

---

## PolygonZone + LineZone

```text
08-polygon-and-line-zone.py
```

Uses the same tracked detections for both spatial systems:

```text
             Tracked Objects
                   ↓
           ┌────────┴────────┐
           ↓                 ↓
     PolygonZone          LineZone
           ↓                 ↓
       Occupancy             Flow
           │                 │
           └────────┬────────┘
                   ↓
               Visualization
```

---

## Complete Zones and Counting Pipeline

```text
09-complete-zones-counting-pipeline.py
```

Combines the complete workflow:

```text
Input Video
     ↓
YOLOv8
     ↓
Supervision Detections
     ↓
ByteTrack
     ↓
Confirmed Tracker IDs
     ↓
┌─────────────────────────┐
│                         │
↓                         ↓
PolygonZone            LineZone
↓                         ↓
Occupancy                Flow
│                         │
└────────────┬────────────┘
             ↓
         Annotation
             ↓
         Output Video
```

The practical version of this pipeline was tested successfully on:

```text
Resolution:   3840 × 2160
FPS:          25
Total Frames: 538
```

Final results:

```text
Final polygon occupancy: 1
Crossings Down: 3
Crossings Up: 3
Total Crossings: 6
```

---

## PolygonZone vs LineZone

The fundamental difference is:

| Feature | PolygonZone | LineZone |
|---|---|---|
| Represents | Area | Boundary |
| Measures | Presence | Crossings |
| Count | Current | Accumulated |
| Main value | `current_count` | `in_count`, `out_count` |
| Analytics type | Occupancy | Flow |

The easiest way to remember the difference is:

```text
PolygonZone:
How many objects are HERE NOW?

LineZone:
How many objects PASSED HERE?
```

---

# 06 — Segmentation with SAM Examples

The Segmentation with SAM examples extend object detection from bounding boxes into **pixel-level object understanding**.

They progress through:

```text
01_yolo_detection.py
        ↓
02_sam_bbox_segmentation.py
        ↓
03_mask_inspection.py
        ↓
04_object_extraction.py
        ↓
05_mask_area_comparison.py
        ↓
06_mask_serialization.py
```

All six examples were successfully executed and validated in Google Colab.

---

## YOLO Detection

```text
01_yolo_detection.py
```

Introduces the detection stage used to generate prompts for SAM 3.

```text
Input Image
     ↓
YOLOv8
     ↓
Object Detections
     ↓
Bounding Boxes
     ↓
SAM 3 Prompts
```

The validated `bus.jpg` execution produced:

```text
Detected objects: 6

1 bus
4 persons
1 stop sign
```

---

## SAM Bounding-Box Segmentation

```text
02_sam_bbox_segmentation.py
```

Uses the YOLO bounding boxes as prompts for SAM 3.

```text
YOLO Bounding Boxes
        ↓
      SAM 3
        ↓
Segmentation Masks
```

Validated result:

```text
YOLO detections: 6
SAM detections: 6
SAM masks generated: 6
Mask array shape: (6, 1080, 810)
```

Every YOLO bounding box successfully produced a SAM 3 segmentation mask.

---

## Mask Inspection

```text
03_mask_inspection.py
```

Demonstrates how segmentation masks are represented as boolean NumPy arrays.

```text
True  → Object Pixel
False → Background Pixel
```

Validated first-mask statistics:

```text
Shape: (1080, 810)
Data type: bool

Object pixels: 265686
Background pixels: 609114
Total pixels: 874800

Image coverage: 30.37%
```

The complete mask collection had the shape:

```text
(6, 1080, 810)
```

representing:

```text
(number_of_objects, image_height, image_width)
```

---

## Object Extraction

```text
04_object_extraction.py
```

Uses a segmentation mask to isolate an object from the original image.

The core operation is:

```python
object_image[~mask] = 0
```

Conceptually:

```text
Original Image
      ↓
SAM 3 Mask
      ↓
Remove Background Pixels
      ↓
Extracted Object
```

The validated execution generated:

```text
extracted_object.png
```

---

## Mask Area Comparison

```text
05_mask_area_comparison.py
```

Compares pixel-level segmentation area with rectangular bounding-box area.

The calculation is:

```text
Mask / Box Percentage
        =
Mask Area / Bounding Box Area × 100
```

Validated results:

| Object | Class | Mask Area | Bounding Box Area | Mask / Box |
|---|---|---:|---:|---:|
| 0 | bus | 265,686 px | 411,059.31 px | 64.63% |
| 1 | person | 46,648 px | 99,214.33 px | 47.02% |
| 2 | person | 20,935 px | 67,998.79 px | 30.79% |
| 3 | person | 32,911 px | 55,768.55 px | 59.01% |
| 4 | person | 10,715 px | 20,346.07 px | 52.66% |
| 5 | stop sign | 1,878 px | 2,288.43 px | 82.07% |

This demonstrates how segmentation provides more precise geometric information than bounding boxes alone.

---

## Mask Serialization

```text
06_mask_serialization.py
```

Demonstrates how a boolean segmentation mask can be stored in JSON using NumPy packbits and Base64 encoding.

```text
Boolean Mask
      ↓
Flatten
      ↓
np.packbits
      ↓
Bytes
      ↓
Base64
      ↓
JSON
```

The example then reconstructs the mask:

```text
JSON
  ↓
Base64 Decode
  ↓
np.frombuffer
  ↓
np.unpackbits
  ↓
Reshape
  ↓
Boolean Mask
```

Validated serialization statistics:

```text
Boolean pixels: 874800
Packed bytes: 109350
Base64 characters: 145800
```

Final validation:

```text
Decoded mask matches original: True
```

This confirms that the original segmentation mask can be stored and reconstructed without losing mask information.

---

## SAM 3 Model

The SAM 3 model checkpoint is not stored in the GitHub repository because of its large size.

The validated Google Colab environment used:

```text
/content/drive/MyDrive/SAM3-Models/sam3.pt
```

Model size:

```text
3.21 GB
```

The model is required by examples `02` through `06`.

Example `01_yolo_detection.py` only requires YOLOv8.

---

## Validated Session 06 Environment

The Session 06 examples were tested with:

```text
Google Colab
NVIDIA T4 GPU
Ultralytics
YOLOv8
SAM 3
Supervision
NumPy
OpenCV
```

During SAM 3 inference, Ultralytics automatically adjusted the requested image size:

```text
imgsz=[1024] must be multiple of max stride 14,
updating to [1036]
```

Inference continued normally and all six masks were successfully generated.

---

## From Examples to Practical Implementations

The repository follows a progressive learning structure.

### Filtering

```text
Individual Filtering Examples
          ↓
Detection Filtering Pipeline
```

### Tracking

```text
Individual Tracking Examples
          ↓
Complete Tracking Pipeline
```

### Zones and Counting

```text
Individual Zone Examples
          ↓
Tracking + Zones
          ↓
PolygonZone + LineZone
          ↓
Complete Spatial Analytics Pipeline
```

### Segmentation

```text
YOLO Detection
      ↓
SAM 3 Prompting
      ↓
Mask Inspection
      ↓
Object Extraction
      ↓
Mask Analysis
      ↓
Mask Serialization
      ↓
Complete Segmentation Practical
```

This creates a progression from isolated concepts to complete computer vision systems.

---

## Purpose

The purpose of this directory is to transform the concepts studied during the course into practical, reusable code.

Each example is designed to isolate an important concept so it can be studied independently before being incorporated into larger computer vision projects.

The repository therefore follows a progression from:

```text
Course Concept
      ↓
Notebook Experiment
      ↓
Reusable Example
      ↓
Practical Implementation
      ↓
Complete Computer Vision Pipeline
```

As new SAM3 sessions are completed, additional example directories will be added here.

---

## Current Progress

| # | Topic | Examples |
|---|---|---:|
| 00 | Agentic AI Programming | 2 |
| 01 | Introduction to Supervision | 8 |
| 02 | Annotation and Visualization | 5 |
| 03 | Filtering and Manipulating Detections | 5 |
| 04 | Object Tracking | 9 |
| 05 | Zones and Counting | 9 |
| 06 | Segmentation with SAM | 6 |

**Total runnable examples: 44**

---

## Related Course Material

### Session 05 — Zones and Counting

Detailed course notes:

[`../08-course-notes/05-Zones-and-Counting/`](../08-course-notes/05-Zones-and-Counting/)

Practical implementation:

[`../08-course-notes/05-Zones-and-Counting/practical/`](../08-course-notes/05-Zones-and-Counting/practical/)

Class recording:

[Watch — SAM3: Zonas y Conteo | PolygonZone, LineZone y ByteTrack](https://youtu.be/43i0z9b81Z4)

---

### Session 06 — Segmentation with SAM

Detailed course notes:

[`../08-course-notes/06-Segmentation-with-SAM/`](../08-course-notes/06-Segmentation-with-SAM/)

Practical implementation:

[`../08-course-notes/06-Segmentation-with-SAM/practical/`](../08-course-notes/06-Segmentation-with-SAM/practical/)

Code examples:

[`06-Segmentation-with-SAM/`](./06-Segmentation-with-SAM/)

Class recording:

[Watch — SAM3: Segmentation with SAM 3](https://youtu.be/1EYfpSsmHO0)

---

## Related Repository Sections

### Course Notebooks

```text
03-notebooks/
```

Contains the original Jupyter / Google Colab notebooks used during the lessons.

### Practical Projects

```text
05-projects/
```

Contains larger projects that combine multiple concepts into reusable computer vision applications.

### Course Notes

```text
08-course-notes/
```

Contains detailed explanations and documentation for each lesson.

---

## Learning Progression

The examples created so far demonstrate the evolution of the computer vision pipeline:

```text
Agentic AI Programming
        ↓
Object Detection
        ↓
Annotation and Visualization
        ↓
Detection Filtering
        ↓
Object Tracking
        ↓
Spatial Zones
        ↓
Occupancy + Flow
        ↓
Spatial Video Analytics
        ↓
YOLO Bounding-Box Prompts
        ↓
SAM 3 Segmentation
        ↓
Pixel-Level Masks
        ↓
Mask Analysis
        ↓
Object Extraction
        ↓
Mask Serialization
```

Each new session builds on concepts introduced in previous sessions.

---

## Author

**Peyman Miyandashti**

SAM3 Learning Journey  
Computer Vision · Artificial Intelligence · Machine Learning

[LinkedIn](https://www.linkedin.com/in/peyman-mxli/) | [GitHub](https://github.com/Peyman-mxli)
