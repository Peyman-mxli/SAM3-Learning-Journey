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
- Mask visualization
- Mask opacity customization
- `sv.BoxAnnotator`
- Mask + bounding-box composition
- Detection filtering before SAM
- Person-only segmentation
- Reusable segmentation functions
- Multiple-image processing
- SAM2 temporal segmentation concepts
- Temporal memory
- Mask propagation

The examples continue from the masks generated in Session 06 and demonstrate how those masks can be visualized, customized, filtered, combined with detections, and reused.

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

Validated results include:

```text
01_basic_mask_annotator.py
YOLO detections: 6
SAM masks generated: 6

02_mask_opacity.py
Opacity 0.2: ✅
Opacity 0.5: ✅
Opacity 0.9: ✅

03_mask_and_box_annotator.py
YOLO detections: 6
SAM masks generated: 6

04_person_only_segmentation.py
Total YOLO detections:      6
Person detections:          4
Objects removed before SAM: 2
Person masks generated:     4

05_reusable_segmentation_function.py
bus.jpg    → 6 detections → 6 masks
zidane.jpg → 3 detections → 3 masks

06_sam2_temporal_concept.py
Static segmentation:   ✅
Temporal memory:       ✅
Mask propagation:      ✅
```

Six visual outputs were also generated and preserved in:

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
