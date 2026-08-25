# SAM3 — Code Examples

This directory contains clean, runnable Python examples based on the practical concepts covered throughout my **SAM3 Computer Vision Learning Journey**.

Unlike the detailed course notes, these files focus on small, reusable code examples that can be executed and studied independently.

The examples progressively build from basic AI-assisted programming and object detection toward filtering, tracking, spatial analysis, occupancy monitoring, object counting, pixel-level segmentation with SAM 3, advanced mask visualization, selective segmentation, reusable segmentation pipelines, and temporal segmentation concepts.

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

### 07 — Advanced MaskAnnotator and SAM2

[`07-Advanced-MaskAnnotator-and-SAM2/`](./07-Advanced-MaskAnnotator-and-SAM2/)

Examples covering:

- `sv.MaskAnnotator`
- `sv.BoxAnnotator`
- YOLOv8 + SAM 3 integration
- Mask visualization
- Mask opacity customization
- Mask + bounding-box composition
- Detection filtering before SAM
- Person-only segmentation
- Reusable segmentation functions
- Multiple-image processing
- SAM2 temporal segmentation concepts
- Temporal memory
- Mask propagation

The examples continue directly from the masks generated in Session 06.

Instead of focusing only on generating segmentation masks, Session 07 explores how those masks can be:

```text
Visualized
     ↓
Customized
     ↓
Combined with Bounding Boxes
     ↓
Filtered by Object Class
     ↓
Reused Across Images
     ↓
Extended Toward Temporal Segmentation
```

The progression is:

```text
SAM 3 Masks
     ↓
MaskAnnotator
     ↓
Opacity Control
     ↓
Mask + Box Composition
     ↓
Detection Filtering
     ↓
Selective Segmentation
     ↓
Reusable Pipeline
     ↓
Temporal Segmentation Concepts
```

All six Session 07 examples were successfully executed and validated in Google Colab.

Validated results:

```text
01_basic_mask_annotator.py

YOLO detections:     6
SAM masks generated: 6
```

```text
02_mask_opacity.py

Opacity 0.2: ✅
Opacity 0.5: ✅
Opacity 0.9: ✅
```

```text
03_mask_and_box_annotator.py

YOLO detections:     6
SAM masks generated: 6
```

```text
04_person_only_segmentation.py

Total YOLO detections:       6
Person detections:           4
Objects removed before SAM:  2
Person masks generated:      4
```

```text
05_reusable_segmentation_function.py

bus.jpg
YOLO detections: 6
SAM masks:       6

zidane.jpg
YOLO detections: 3
SAM masks:       3
```

```text
06_sam2_temporal_concept.py

Static segmentation concept: ✅
Temporal memory concept:     ✅
Mask propagation concept:    ✅
```

Six validated image outputs were preserved in:

[`07-Advanced-MaskAnnotator-and-SAM2/assets/output/`](./07-Advanced-MaskAnnotator-and-SAM2/assets/output/)

```text
01_basic_mask_annotator_output.png
02_mask_opacity_output.png
03_mask_and_box_annotator_output.png
04_person_only_segmentation_output.png
05_reusable_bus_output.png
05_reusable_zidane_output.png
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

### Advanced Mask Visualization Workflow

```text
Input Image
    ↓
YOLOv8
    ↓
Detections
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
Opacity Customization
    ↓
BoxAnnotator
    ↓
Combined Visualization
```

### Reusable Segmentation Workflow

```text
Image A ──────┐
              │
Image B ──────┼──→ segment_image()
              │          ↓
Image N ──────┘       YOLOv8
                         ↓
                     Detections
                         ↓
                       SAM 3
                         ↓
                       Masks
                         ↓
                    Annotation
                         ↓
                       Result
```

The same processing function can therefore be applied to different input images without duplicating the complete segmentation pipeline.

### Temporal Segmentation Concept

Session 07 introduces the conceptual progression from static segmentation toward temporal segmentation:

```text
Initial Frame
      ↓
Object Prompt
      ↓
Initial Mask
      ↓
Temporal Memory
      ↓
Next Frame
      ↓
Updated Mask
      ↓
Update Temporal Memory
      ↓
Future Frames
```

The conceptual distinction is:

```text
Static Segmentation
        ↓
Spatial Information
```

versus:

```text
Temporal Segmentation
        ↓
Spatial Information
        +
Temporal Information
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
- `sv.Detections`
- `sv.MaskAnnotator`
- `sv.BoxAnnotator`
- ByteTrack
- Base64
- JSON
- Google Colab

The latest examples additionally introduce:

- SAM2 concepts
- Temporal segmentation
- Temporal memory
- Video mask propagation

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
├── 07-Advanced-MaskAnnotator-and-SAM2/
│   ├── README.md
│   ├── 01_basic_mask_annotator.py
│   ├── 02_mask_opacity.py
│   ├── 03_mask_and_box_annotator.py
│   ├── 04_person_only_segmentation.py
│   ├── 05_reusable_segmentation_function.py
│   ├── 06_sam2_temporal_concept.py
│   └── assets/
│       ├── README.md
│       └── output/
│           ├── README.md
│           ├── 01_basic_mask_annotator_output.png
│           ├── 02_mask_opacity_output.png
│           ├── 03_mask_and_box_annotator_output.png
│           ├── 04_person_only_segmentation_output.png
│           ├── 05_reusable_bus_output.png
│           └── 05_reusable_zidane_output.png
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

The Object Tracking examples demonstrate how object detection can be extended across multiple video frames using **ByteTrack**.

Instead of treating every detection independently, tracking assigns persistent IDs to objects so their movement and visibility can be analyzed over time.

---

## Basic ByteTrack

```text
01-basic-bytetrack.py
```

Introduces:

```python
sv.ByteTrack()
```

The basic tracking workflow is:

```text
Video Frame
     ↓
YOLO
     ↓
Detections
     ↓
ByteTrack
     ↓
Tracked Detections
```

ByteTrack receives detections from each frame and attempts to associate them with previously observed objects.

---

## Tracker IDs Across Frames

```text
02-tracker-ids-across-frames.py
```

Demonstrates persistent object identities.

For example:

```text
Frame 100 → Person → ID 4
Frame 101 → Person → ID 4
Frame 102 → Person → ID 4
Frame 103 → Person → ID 4
```

The repeated tracker ID indicates that the detections correspond to the same tracked object.

The conceptual workflow is:

```text
Frame N
   ↓
Detection
   ↓
ByteTrack
   ↓
tracker_id
   ↓
Frame N + 1
   ↓
Same Object
   ↓
Same tracker_id
```

---

## Track Video with Annotations

```text
03-track-video-with-annotations.py
```

Combines tracking with visualization.

The workflow becomes:

```text
Video
  ↓
Frames
  ↓
YOLO
  ↓
Detections
  ↓
ByteTrack
  ↓
Persistent IDs
  ↓
Annotators
  ↓
Output Video
```

This allows tracked objects to remain visually identifiable throughout the video.

---

## Class + Tracker ID

```text
04-class-and-tracker-id.py
```

Combines object class information with persistent tracker IDs.

Example labels:

```text
person #1
person #2
car #3
bus #4
```

This makes it possible to understand both:

```text
WHAT the object is
```

and:

```text
WHICH tracked object it is
```

---

## Filter Before Tracking

```text
05-filter-before-tracking.py
```

Demonstrates that detections can be filtered before being passed to ByteTrack.

Conceptually:

```text
YOLO
  ↓
All Detections
  ↓
Filtering
  ↓
Relevant Detections
  ↓
ByteTrack
```

This can reduce unnecessary tracking when only particular classes or confidence levels are relevant.

---

## Tracking Frame Count

```text
06-tracking-frame-count.py
```

Demonstrates how many frames each tracked object appears in.

For each tracker ID, the system can maintain:

```text
tracker_id
     ↓
frame_count
```

Example:

```text
ID 1 → 75 frames
ID 2 → 62 frames
ID 3 → 18 frames
```

This provides a simple temporal measurement of object visibility.

---

## Unique Object Count

```text
07-unique-object-count.py
```

Demonstrates how persistent tracker IDs can be used to count unique objects.

Without tracking:

```text
Frame 1 → Person
Frame 2 → Person
Frame 3 → Person
```

could incorrectly appear to represent three separate objects.

With tracking:

```text
Frame 1 → ID 7
Frame 2 → ID 7
Frame 3 → ID 7
```

the system understands that this is one unique object.

Unique counting can therefore be based on:

```python
set(tracker_ids)
```

---

## Visible Time

```text
08-visible-time.py
```

Extends frame counting into an estimate of object visibility duration.

Conceptually:

```text
Tracked Frames
      ↓
Video FPS
      ↓
Visible Time
```

For example:

```text
75 tracked frames
÷
30 FPS
=
2.5 seconds
```

This demonstrates how tracking data can be converted into temporal analytics.

---

## Complete Tracking Pipeline

```text
09-complete-tracking-pipeline.py
```

Combines the concepts from the previous tracking examples into one workflow:

```text
Input Video
     ↓
YOLO
     ↓
Detections
     ↓
Optional Filtering
     ↓
ByteTrack
     ↓
Persistent IDs
     ↓
Class + ID Labels
     ↓
Frame Counts
     ↓
Unique Object Analysis
     ↓
Visibility Duration
     ↓
Annotated Output Video
```

This represents the transition from basic object detection toward complete temporal object analysis.

---

# 05 — Zones and Counting Examples

The Zones and Counting examples extend tracking with **spatial reasoning**.

Tracking answers:

```text
Which object is this?
```

Zones add questions such as:

```text
Is the object inside this area?
```

and:

```text
Did the object cross this line?
```

---

## Basic PolygonZone

```text
01-basic-polygon-zone.py
```

Introduces a polygon-shaped region of interest.

A polygon can be defined using coordinates:

```python
polygon = np.array([
    [x1, y1],
    [x2, y2],
    [x3, y3],
    [x4, y4]
])
```

and then used with:

```python
sv.PolygonZone(
    polygon=polygon
)
```

The zone represents an application-specific area inside the image or video frame.

---

## PolygonZone Current Count

```text
02-polygon-zone-current-count.py
```

Introduces:

```python
zone.trigger(detections)
```

and:

```python
zone.current_count
```

The key question is:

```text
How many objects are inside this polygon RIGHT NOW?
```

Conceptually:

```text
Detections
    ↓
PolygonZone
    ↓
Inside / Outside
    ↓
Current Occupancy
```

The count can change from frame to frame as objects enter or leave the region.

---

## Filter Detections Inside Zone

```text
03-filter-detections-inside-zone.py
```

Demonstrates that `PolygonZone` can generate a Boolean mask indicating which detections are located inside the region.

Conceptually:

```text
All Detections
      ↓
PolygonZone Trigger
      ↓
Boolean Zone Mask
      ↓
Detections Inside Zone
```

This allows spatial filtering based on application-defined regions.

---

## Tracking with PolygonZone

```text
04-tracking-with-polygon-zone.py
```

Combines ByteTrack with PolygonZone.

The workflow becomes:

```text
Video
  ↓
YOLO
  ↓
Detections
  ↓
ByteTrack
  ↓
Tracked Objects
  ↓
PolygonZone
  ↓
Current Occupancy
```

Persistent IDs make it possible to understand which tracked objects are present in the zone.

---

## Basic LineZone

```text
05-basic-line-zone.py
```

Introduces a virtual line used to detect object crossings.

Unlike PolygonZone, which represents an area, LineZone represents a boundary.

Conceptually:

```text
Object Trajectory
      ↓
Virtual Line
      ↓
Crossing Event
```

---

## LineZone Crossing Count

```text
06-line-zone-crossing-count.py
```

Introduces directional crossing counts:

```python
line_zone.in_count
```

and:

```python
line_zone.out_count
```

The key question becomes:

```text
How many tracked objects crossed the line?
```

The system can distinguish between the two directions of travel.

---

## Tracking with LineZone

```text
07-tracking-with-line-zone.py
```

Combines persistent tracker IDs with line-crossing analysis.

The workflow is:

```text
Video
  ↓
YOLO
  ↓
Detections
  ↓
ByteTrack
  ↓
Persistent IDs
  ↓
LineZone
  ↓
Crossing Events
  ↓
Directional Counts
```

Tracking is important because a crossing event needs object identity across multiple frames.

---

## PolygonZone + LineZone

```text
08-polygon-and-line-zone.py
```

Combines both spatial analytics concepts.

```text
Tracked Objects
      ↓
┌───────────────────────────┐
│                           │
↓                           ↓
PolygonZone              LineZone
↓                           ↓
Occupancy                  Flow
```

The two measurements answer different questions:

```text
PolygonZone
→ How many objects are inside the region now?

LineZone
→ How many objects crossed the boundary over time?
```

This introduces the distinction between **occupancy** and **flow**.

---

## Complete Zones and Counting Pipeline

```text
09-complete-zones-counting-pipeline.py
```

Combines detection, tracking, spatial zones, counting, and visualization.

```text
Input Video
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

This represents a complete spatial video analytics pipeline.

---

# 06 — Segmentation with SAM Examples

The Segmentation with SAM examples extend object detection from rectangular bounding boxes toward **pixel-level object representations**.

YOLO provides object localization:

```text
Object
  ↓
Bounding Box
```

SAM 3 extends this to:

```text
Object
  ↓
Pixel-Level Mask
```

All six Session 06 examples were successfully validated in Google Colab.

---

## YOLO Detection

```text
01_yolo_detection.py
```

Establishes the detection stage before segmentation.

The workflow is:

```text
bus.jpg
   ↓
YOLOv8
   ↓
Detections
   ↓
Bounding Boxes
```

Validated results:

```text
Input image: bus.jpg
Image shape: (1080, 810, 3)
YOLO detections: 6
```

The detected objects become the prompts for the SAM 3 segmentation stage.

---

## SAM Bounding-Box Segmentation

```text
02_sam_bbox_segmentation.py
```

Uses YOLO bounding boxes as prompts for SAM 3.

```text
Input Image
     ↓
YOLOv8
     ↓
Bounding Boxes
     ↓
SAM 3
     ↓
Segmentation Masks
```

Validated results:

```text
YOLO detections:     6
SAM masks generated: 6
```

The important transition is:

```text
Bounding Box
     ↓
SAM Prompt
     ↓
Pixel-Level Mask
```

---

## Mask Inspection

```text
03_mask_inspection.py
```

Inspects the segmentation masks produced by SAM 3.

Validated mask array:

```text
(6, 1080, 810)
```

Each mask corresponds to one segmented object.

The example calculates information such as:

```text
Object Pixels
Image Coverage
Mask Dimensions
```

Validated first-mask result:

```text
First mask object pixels: 265686
First mask image coverage: 30.37%
```

This demonstrates that masks can be treated as measurable data rather than only visual overlays.

---

## Object Extraction

```text
04_object_extraction.py
```

Uses a segmentation mask to isolate object pixels from the original image.

Conceptually:

```text
Original Image
      +
Boolean Mask
      ↓
Pixel Selection
      ↓
Extracted Object
```

This demonstrates one of the major advantages of segmentation over ordinary bounding boxes.

A bounding box contains background pixels around the object.

A segmentation mask can identify the object's visible pixels more precisely.

---

## Mask Area Comparison

```text
05_mask_area_comparison.py
```

Compares:

```text
Bounding-Box Area
```

with:

```text
Segmentation-Mask Area
```

Conceptually:

```text
Bounding Box
     ↓
Rectangular Area

Segmentation Mask
     ↓
Actual Segmented Pixels
```

This provides a more precise way to measure the visible object region.

---

## Mask Serialization

```text
06_mask_serialization.py
```

Demonstrates how segmentation masks can be converted into a format suitable for storage.

The workflow includes:

```text
Boolean Mask
     ↓
Binary Representation
     ↓
Base64 Encoding
     ↓
JSON Storage
```

The stored mask can later be reconstructed:

```text
JSON
  ↓
Base64 Decode
  ↓
Binary Data
  ↓
Mask Reconstruction
```

Validated result:

```text
Decoded mask matches original: True
```

This demonstrates that segmentation results can be preserved and reconstructed instead of existing only during inference.

---

# 07 — Advanced MaskAnnotator and SAM2 Examples

The Advanced MaskAnnotator and SAM2 examples continue directly from Session 06.

Session 06 focused primarily on:

```text
Generating Masks
      ↓
Inspecting Masks
      ↓
Analyzing Masks
      ↓
Extracting Objects
      ↓
Serializing Masks
```

Session 07 focuses on:

```text
Visualizing Masks
      ↓
Customizing Masks
      ↓
Combining Masks with Detections
      ↓
Filtering Before Segmentation
      ↓
Reusing Segmentation Pipelines
      ↓
Understanding Temporal Segmentation
```

All six examples were successfully executed and validated in Google Colab.

---

## Basic MaskAnnotator

```text
01_basic_mask_annotator.py
```

Introduces:

```python
sv.MaskAnnotator()
```

The workflow is:

```text
Input Image
     ↓
YOLOv8
     ↓
Object Detections
     ↓
Bounding Boxes
     ↓
SAM 3
     ↓
Segmentation Masks
     ↓
MaskAnnotator
     ↓
Annotated Image
```

Validated `bus.jpg` results:

```text
Image shape: (1080, 810, 3)

YOLO detections:     6
SAM masks generated: 6
```

YOLO detected:

```text
4 persons
1 bus
1 stop sign
```

Generated output:

```text
assets/output/01_basic_mask_annotator_output.png
```

---

## Mask Opacity

```text
02_mask_opacity.py
```

Demonstrates how the visual strength of segmentation masks can be customized.

The tested opacity values were:

```text
0.2
0.5
0.9
```

Interpretation:

```text
0.2
 ↓
Original Image More Visible

0.5
 ↓
Balanced Visualization

0.9
 ↓
Segmentation Masks More Visible
```

Validated results:

```text
YOLO detections:     6
SAM masks generated: 6

Opacity 0.2: ✅
Opacity 0.5: ✅
Opacity 0.9: ✅
```

Generated output:

```text
assets/output/02_mask_opacity_output.png
```

---

## Mask + Box Annotator

```text
03_mask_and_box_annotator.py
```

Combines:

```python
sv.MaskAnnotator()
```

and:

```python
sv.BoxAnnotator()
```

The visualization layers are:

```text
1. Original image
2. SAM 3 segmentation masks
3. YOLO bounding boxes
```

Workflow:

```text
Original Image
      ↓
YOLOv8
      ↓
Bounding Boxes
      ↓
SAM 3
      ↓
Segmentation Masks
      ↓
MaskAnnotator
      ↓
BoxAnnotator
      ↓
Combined Visualization
```

Validated results:

```text
YOLO detections:     6
SAM masks generated: 6
```

The example visually compares:

```text
Bounding Box
     ↓
Rectangular Localization
```

with:

```text
Segmentation Mask
     ↓
Pixel-Level Representation
```

Generated output:

```text
assets/output/03_mask_and_box_annotator_output.png
```

---

## Person-Only Segmentation

```text
04_person_only_segmentation.py
```

Demonstrates detection filtering **before** SAM inference.

For the COCO `person` class:

```python
person_detections = detections[
    detections.class_id == 0
]
```

The workflow becomes:

```text
Input Image
     ↓
YOLOv8
     ↓
All Detections
     ↓
Person Filter
     ↓
Person Bounding Boxes
     ↓
SAM 3
     ↓
Person Masks
```

Validated results:

```text
Total YOLO detections:       6
Person detections:           4
Objects removed before SAM:  2
Person masks generated:      4
```

Therefore:

```text
6 Detections
     ↓
Filter
     ↓
4 Person Detections
     ↓
SAM 3
     ↓
4 Person Masks
```

The key principle is:

```text
Filter BEFORE SAM
```

rather than:

```text
Segment Everything
      ↓
Filter Afterwards
```

Generated output:

```text
assets/output/04_person_only_segmentation_output.png
```

---

## Reusable Segmentation Function

```text
05_reusable_segmentation_function.py
```

Encapsulates the segmentation pipeline in:

```python
segment_image()
```

The reusable workflow is:

```text
Image
  ↓
segment_image()
  ↓
YOLOv8
  ↓
sv.Detections
  ↓
Bounding Boxes
  ↓
SAM 3
  ↓
Segmentation Masks
  ↓
MaskAnnotator
  ↓
BoxAnnotator
  ↓
Result
```

The same function was validated with two different images.

### bus.jpg

```text
YOLO detections: 6
SAM masks:       6
```

Generated output:

```text
assets/output/05_reusable_bus_output.png
```

### zidane.jpg

YOLO detected:

```text
2 persons
1 tie
```

Validated results:

```text
YOLO detections: 3
SAM masks:       3
```

Generated output:

```text
assets/output/05_reusable_zidane_output.png
```

The same processing logic therefore handles both:

```text
bus.jpg ─────────┐
                 ↓
           segment_image()
                 ↓
             Result A


zidane.jpg ──────┐
                 ↓
           segment_image()
                 ↓
             Result B
```

This demonstrates:

```text
Reusable Pipeline
instead of
Repeated Code
```

---

## SAM2 Temporal Concept

```text
06_sam2_temporal_concept.py
```

This example is intentionally conceptual.

It does not load a SAM2 model and does not generate an output image.

Instead, it demonstrates the conceptual transition from static-image segmentation to segmentation that uses information across video frames.

Static segmentation:

```text
Image
  ↓
Prompt
  ↓
Segmentation Model
  ↓
Mask
```

Each image is independent.

Temporal segmentation:

```text
Initial Frame
      ↓
Object Prompt
      ↓
Initial Segmentation Mask
      ↓
Store Temporal Memory
      ↓
Next Frame
      +
Previous Temporal Memory
      ↓
Updated Segmentation Mask
      ↓
Update Temporal Memory
      ↓
Future Frames
```

The conceptual distinction is:

```text
Static Segmentation
        ↓
Spatial Information
```

versus:

```text
Temporal Segmentation
        ↓
Spatial Information
        +
Temporal Information
```

Mask propagation can be represented as:

```text
Initial Frame
     ↓
Object Prompt
     ↓
Initial Mask
     ↓
Temporal Memory
     ↓
Frame 2
     ↓
Updated Mask
     ↓
Frame 3
     ↓
Updated Mask
     ↓
Future Frames
```

Validated concepts:

```text
Static segmentation: ✅
Temporal memory:     ✅
Mask propagation:    ✅
```

---

## Session 07 Visual Outputs

The first five examples generated six validated output images.

They are stored in:

[`07-Advanced-MaskAnnotator-and-SAM2/assets/output/`](./07-Advanced-MaskAnnotator-and-SAM2/assets/output/)

```text
assets/output/
├── README.md
├── 01_basic_mask_annotator_output.png
├── 02_mask_opacity_output.png
├── 03_mask_and_box_annotator_output.png
├── 04_person_only_segmentation_output.png
├── 05_reusable_bus_output.png
└── 05_reusable_zidane_output.png
```

Output summary:

```text
Example 01 → 1 image
Example 02 → 1 image
Example 03 → 1 image
Example 04 → 1 image
Example 05 → 2 images
Example 06 → 0 images (conceptual)

Total → 6 images
```

Asset documentation:

[`07-Advanced-MaskAnnotator-and-SAM2/assets/README.md`](./07-Advanced-MaskAnnotator-and-SAM2/assets/README.md)

Output documentation:

[`07-Advanced-MaskAnnotator-and-SAM2/assets/output/README.md`](./07-Advanced-MaskAnnotator-and-SAM2/assets/output/README.md)

---

## Session 07 Validation Summary

```text
01_basic_mask_annotator.py              ✅ PASSED
02_mask_opacity.py                      ✅ PASSED
03_mask_and_box_annotator.py            ✅ PASSED
04_person_only_segmentation.py          ✅ PASSED
05_reusable_segmentation_function.py    ✅ PASSED
06_sam2_temporal_concept.py             ✅ PASSED
```

Key validated results:

```text
bus.jpg
├── YOLO detections: 6
├── SAM masks: 6
└── Person-only masks: 4

zidane.jpg
├── YOLO detections: 3
└── SAM masks: 3

Mask opacity:
├── 0.2 ✅
├── 0.5 ✅
└── 0.9 ✅

Examples validated: 6
Visual outputs:     6
```

---

## Session 07 SAM 3 Environment

The SAM 3 checkpoint used during validation was:

```text
/content/drive/MyDrive/SAM3-Models/sam3.pt
```

During inference, Ultralytics automatically adjusted:

```text
imgsz=[1024]
```

to:

```text
imgsz=[1036]
```

because the requested image size must be compatible with the model stride.

This warning was non-fatal.

SAM 3 inference continued successfully and generated the expected segmentation masks.

---

## Session 06 → Session 07 Progression

Session 06:

```text
YOLO Detection
      ↓
SAM 3
      ↓
Segmentation Masks
      ↓
Mask Inspection
      ↓
Object Extraction
      ↓
Mask Analysis
      ↓
Mask Serialization
```

Session 07:

```text
Segmentation Masks
      ↓
MaskAnnotator
      ↓
Opacity Control
      ↓
Mask + Box Composition
      ↓
Selective Segmentation
      ↓
Reusable Functions
      ↓
Temporal Segmentation Concepts
```

Combined progression:

```text
Session 06
Generate + Analyze Masks
        ↓
Session 07
Visualize + Customize + Reuse Masks
        ↓
Future Work
Temporal Video Segmentation
```

---


## From Examples to Practical Implementations

The repository follows a progressive learning structure.

Individual examples isolate specific concepts.

Practical implementations combine those concepts into larger workflows.

Projects then connect multiple workflows into more complete computer vision applications.

---

### Filtering

```text
Individual Filtering Examples
          ↓
Detection Filtering Pipeline
```

The filtering examples introduce:

```text
Confidence
Class
Size
NMS
Spatial Position
```

These concepts can then be combined into a complete post-processing pipeline.

---

### Tracking

```text
Individual Tracking Examples
          ↓
Persistent Tracker IDs
          ↓
Tracking Analytics
          ↓
Complete Tracking Pipeline
```

The tracking examples progress from basic ByteTrack usage toward:

```text
Class + Tracker ID
Frame Counts
Unique Objects
Visibility Duration
Complete Video Tracking
```

---

### Zones and Counting

```text
Individual Zone Examples
          ↓
Tracking + PolygonZone
          ↓
Tracking + LineZone
          ↓
PolygonZone + LineZone
          ↓
Complete Spatial Analytics Pipeline
```

This progression combines temporal identity with spatial reasoning.

The result is a system capable of measuring:

```text
Occupancy
+
Flow
```

---

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
Mask Visualization
      ↓
Opacity Customization
      ↓
Mask + Box Composition
      ↓
Selective Segmentation
      ↓
Reusable Segmentation Pipeline
      ↓
Temporal Segmentation Concepts
      ↓
Complete Segmentation Practical
```

This creates a progression from isolated segmentation concepts toward reusable and increasingly advanced computer vision workflows.

---

## Purpose

The purpose of this directory is to transform concepts studied during the course into practical, reusable code.

Each example is designed to isolate an important concept so it can be studied independently before being incorporated into larger computer vision workflows.

The repository therefore follows the progression:

```text
Course Concept
      ↓
Notebook Experiment
      ↓
Reusable Example
      ↓
Practical Implementation
      ↓
Validated Output
      ↓
Complete Computer Vision Pipeline
```

As new SAM3 sessions are completed, additional example directories will continue to be added here.

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
| 07 | Advanced MaskAnnotator and SAM2 | 6 |

**Total runnable examples: 50**

Session 07 validation:

```text
Examples tested: 6
Examples passed: 6
Visual outputs:  6

Status: COMPLETE ✅
```

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

### Session 07 — Advanced MaskAnnotator and SAM2

Detailed course notes:

[`../08-course-notes/07-Advanced-MaskAnnotator-and-SAM2/`](../08-course-notes/07-Advanced-MaskAnnotator-and-SAM2/)

Practical implementation:

[`../08-course-notes/07-Advanced-MaskAnnotator-and-SAM2/practical/`](../08-course-notes/07-Advanced-MaskAnnotator-and-SAM2/practical/)

Code examples:

[`07-Advanced-MaskAnnotator-and-SAM2/`](./07-Advanced-MaskAnnotator-and-SAM2/)

Validated example outputs:

[`07-Advanced-MaskAnnotator-and-SAM2/assets/output/`](./07-Advanced-MaskAnnotator-and-SAM2/assets/output/)

Class recording:

[Watch — Session 07: Advanced MaskAnnotator and SAM2](https://youtu.be/GNwQl-hy8Yw)

---

## Related Repository Sections

### Course Notebooks

```text
03-notebooks/
```

Contains the original Jupyter / Google Colab notebooks used during the lessons.

---

### Practical Projects

```text
05-projects/
```

Contains larger projects that combine multiple concepts into reusable computer vision applications.

---

### Course Notes

```text
08-course-notes/
```

Contains detailed explanations, practical documentation, notebooks, and class recordings for each lesson.

---

### Repository Assets

```text
09-assets/
```

Contains banners, screenshots, and supporting visual resources used throughout the repository.

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
        ↓
MaskAnnotator
        ↓
Mask Opacity
        ↓
Mask + Bounding Box Composition
        ↓
Selective Segmentation
        ↓
Reusable Segmentation Pipelines
        ↓
SAM2 Temporal Concepts
        ↓
Temporal Memory
        ↓
Mask Propagation Concepts
```

Each new session builds on concepts introduced in previous sessions.

Session 07 extends the learning journey from simply generating and analyzing segmentation masks toward **visualizing, customizing, selectively generating, and reusing those masks**, while introducing the conceptual foundation for temporal video segmentation.

---

## Overall Example Progression

The complete progression across the example directories is:

```text
00
Agentic AI Programming
        ↓
01
Introduction to Supervision
        ↓
02
Annotation and Visualization
        ↓
03
Filtering and Manipulating Detections
        ↓
04
Object Tracking
        ↓
05
Zones and Counting
        ↓
06
Segmentation with SAM
        ↓
07
Advanced MaskAnnotator and SAM2
```

This progression moves from basic development workflows toward increasingly complete computer vision systems.

---

## Current Example Capabilities

The examples now cover:

```text
Image Loading
      ↓
Object Detection
      ↓
Detection Representation
      ↓
Annotation
      ↓
Filtering
      ↓
Non-Maximum Suppression
      ↓
Spatial Filtering
      ↓
Object Tracking
      ↓
Persistent IDs
      ↓
Tracking Analytics
      ↓
Spatial Zones
      ↓
Occupancy
      ↓
Line Crossings
      ↓
Spatial Video Analytics
      ↓
SAM 3 Prompting
      ↓
Pixel-Level Segmentation
      ↓
Mask Inspection
      ↓
Object Extraction
      ↓
Mask Serialization
      ↓
Advanced Mask Visualization
      ↓
Selective Segmentation
      ↓
Reusable Segmentation Functions
      ↓
Temporal Segmentation Concepts
```

This creates a strong foundation for future image- and video-based computer vision workflows.

---

## Validation Status

The latest completed example group is:

```text
07 — Advanced MaskAnnotator and SAM2
```

Validation status:

```text
README documentation:      ✅
Python examples:            6 / 6 ✅
Google Colab execution:     6 / 6 ✅
Generated visual outputs:   6 / 6 ✅
Asset documentation:        ✅
Output documentation:       ✅
```

Overall Session 07 example status:

```text
COMPLETE ✅
```

---

## Author

**Peyman Miyandashti**

SAM3 Learning Journey  
Computer Vision · Artificial Intelligence · Machine Learning

[LinkedIn](https://www.linkedin.com/in/peyman-mxli/) | [GitHub](https://github.com/Peyman-mxli/)
