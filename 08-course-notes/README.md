# SAM3 Course Notes

This directory contains my organized notes, concepts, practical exercises, notebooks, class recordings, and supporting material from the **SAM3 — Computer Vision with Segment Anything Model 3** course.

The purpose of this section is to document each course session in a structured way, transforming the original class material into a reusable computer vision learning reference.

---

## Course Sessions

| # | Session | Status |
|---|---|---|
| 00 | [Agentic AI Programming](./00-Agentic-AI-Programming/) | Completed |
| 01 | [Introduction to Supervision](./01-Introduction-to-Supervision/) | Completed |
| 02 | [Annotation and Visualization](./02-Annotation-and-Visualization/) | Completed |
| 03 | [Filtering and Manipulating Detections](./03-Filtering-and-Manipulating-Detections/) | Completed |
| 04 | [Object Tracking](./04-Object-Tracking/) | Completed |

---

# Session 00 — Agentic AI Programming

[`00-Agentic-AI-Programming/`](./00-Agentic-AI-Programming/)

This session introduces AI-assisted programming methodologies and the use of AI tools during software and computer vision development.

Topics include:

- Agentic AI programming
- AI-assisted code generation
- Prompt engineering
- AI-assisted debugging
- Error analysis
- OpenCV
- Computer vision development workflows

**Status:** Completed

---

# Session 01 — Introduction to Supervision

[`01-Introduction-to-Supervision/`](./01-Introduction-to-Supervision/)

This session introduces the **Supervision** library and its role in computer vision workflows.

Topics include:

- Supervision
- YOLOv8
- Ultralytics
- OpenCV
- `sv.Detections`
- Bounding boxes
- Class IDs
- Confidence scores
- Confidence thresholds
- Detection visualization
- Model comparison
- JSON prediction export

The session establishes the basic workflow:

```text
Input Image
     ↓
YOLO
     ↓
Predictions
     ↓
sv.Detections
     ↓
Processing
     ↓
Visualization
```

**Status:** Completed

---

# Session 02 — Annotation and Visualization

[`02-Annotation-and-Visualization/`](./02-Annotation-and-Visualization/)

This session explores how object-detection results can be transformed into clear and useful visual representations using **Supervision Annotators**.

Topics include:

- `BoxAnnotator`
- `RoundBoxAnnotator`
- `HaloAnnotator`
- `BlurAnnotator`
- `BoxCornerAnnotator`
- `EllipseAnnotator`
- `DotAnnotator`
- `LabelAnnotator`
- Annotation customization
- Color palettes
- Bounding-box thickness
- Label text scale
- Class and confidence labels
- Annotation layers
- Annotation order
- Multi-Annotator visualization pipelines
- Detection vs. visualization

The session develops the workflow:

```text
Input Image
     ↓
YOLOv8
     ↓
Detection Results
     ↓
sv.Detections
     ↓
Supervision Annotators
     ↓
Annotation Layers
     ↓
Final Visualization
```

This lesson also connects directly to the practical project:

```text
05-projects/
└── 02-Multi-Annotator-Visualization-Pipeline/
```

**Status:** Completed

---

# Session 03 — Filtering and Manipulating Detections

[`03-Filtering-and-Manipulating-Detections/`](./03-Filtering-and-Manipulating-Detections/)

This session focuses on **post-processing object detections** and selecting exactly which predictions should continue through a computer vision pipeline.

Instead of using every raw prediction produced by YOLO, detections can be filtered according to confidence, class, size, position, and other conditions.

Topics include:

- `sv.Detections` filtering
- NumPy-style Boolean masks
- Confidence filtering
- Class filtering
- Combining multiple conditions
- Element-wise Boolean operators
- Excluding specific classes
- Bounding-box area filtering
- Detection merging
- `sv.Detections.merge()`
- Non-Maximum Suppression (NMS)
- Intersection over Union (IoU)
- NMS thresholds
- Duplicate detection removal
- Sorting detections by confidence
- `np.argsort()`
- Top-N detection selection
- Bounding-box coordinates
- Bounding-box center calculation
- Spatial filtering
- Left/right image filtering
- Region-based detection logic

The session develops the post-processing workflow:

```text
Input Image
     ↓
YOLO
     ↓
Raw Detections
     ↓
sv.Detections
     ↓
Confidence Filtering
     ↓
Class Filtering
     ↓
Size Filtering
     ↓
Merge + NMS
     ↓
Confidence Ranking
     ↓
Spatial Filtering
     ↓
Final Relevant Detections
```

## Detection Filtering Example

A computer vision application can define increasingly specific requirements:

```text
All Detections
      ↓
Keep Only People
      ↓
Confidence > 60%
      ↓
Area > 5000 px²
      ↓
Remove Duplicate Boxes
      ↓
Keep Only Required Image Region
      ↓
Final Detections
```

This demonstrates that model inference is only one part of an object-detection system.

**Post-processing determines which predictions are actually useful for the application.**

## Lesson Materials

- [Main Lesson Documentation](./03-Filtering-and-Manipulating-Detections/README.md)
- [Concept Documentation](./03-Filtering-and-Manipulating-Detections/concepts/)
- [Practical Exercises](./03-Filtering-and-Manipulating-Detections/practical-exercises/)
- [Original Course Notebook](./03-Filtering-and-Manipulating-Detections/02_a_filtrado_detecciones.ipynb)
- [Class Recording](./03-Filtering-and-Manipulating-Detections/CLASS-RECORDING.md)

**Status:** Completed

---

# Session 04 — Object Tracking

[`04-Object-Tracking/`](./04-Object-Tracking/)

This session extends computer vision from detecting objects in individual images to **tracking objects across consecutive video frames**.

Instead of treating every detection independently, object tracking attempts to maintain the identity of each object while it moves through a video.

The session introduces **ByteTrack** together with Supervision and explores how persistent tracking IDs can be used to follow objects, visualize trajectories, and prepare tracking analytics.

Topics include:

- Object tracking
- Video processing
- Frame-by-frame processing
- Multi-object tracking
- Supervision
- `sv.Detections`
- ByteTrack
- `sv.ByteTrack`
- Object association
- Persistent tracking IDs
- `tracker_id`
- Tracking labels
- Bounding-box tracking
- `BoxAnnotator`
- `LabelAnnotator`
- `TraceAnnotator`
- Object trajectories
- Tracking visualization
- Tracking analytics
- OpenCV video processing
- Video export
- H.264 conversion
- FFmpeg
- Google Colab testing

The conceptual transition is:

```text
Object Detection
       ↓
Independent Detections
       ↓
Object Association
       ↓
ByteTrack
       ↓
Persistent tracker_id
       ↓
Object Tracking
       ↓
Movement History
       ↓
Tracking Analytics
```

## Detection vs. Tracking

Object detection answers:

```text
What objects are visible in this frame?
```

Object tracking adds another question:

```text
Is this the same object that appeared in previous frames?
```

For example:

```text
Frame 1 → person #1
Frame 2 → person #1
Frame 3 → person #1
Frame 4 → person #1
```

Although the object changes position, the tracker attempts to maintain the same identity.

---

## Object Tracking Workflow

The complete conceptual workflow developed in this session is:

```text
Input Video
     ↓
Video Frames
     ↓
Object Detection
     ↓
sv.Detections
     ↓
Detection Filtering
     ↓
ByteTrack
     ↓
tracker_id
     ↓
Tracking Annotations
     ↓
Object Trajectories
     ↓
Tracking Analytics
     ↓
Processed Video
```

---

## ByteTrack

ByteTrack associates detections between consecutive frames and assigns persistent tracking identities.

Conceptually:

```text
Frame N Detections
        ↓
ByteTrack
        ↓
Compare With Previous Tracks
        ↓
Object Association
        ↓
tracker_id
```

This allows multiple objects of the same class to be distinguished.

Example:

```text
person #1
person #2
person #3
```

---

## Object Trajectories

Once an object has a persistent tracking identity, its previous positions can be stored and visualized.

Conceptually:

```text
Previous Positions
        ↓
● → ● → ● → ●
            ↑
     Current Position
```

Object trajectories can support applications such as:

- Movement analysis
- Direction analysis
- Line crossing
- Entrance and exit monitoring
- Object counting
- Traffic analysis
- Behavior analysis
- Tracking analytics

---

## Object Tracking Practical

A complete practical exercise was created for this session.

The practical uses a custom **10-second synthetic demonstration video** containing multiple moving objects.

The exercise focuses directly on tracking behavior by generating known detections and passing them through:

```text
Synthetic Detections
        ↓
sv.Detections
        ↓
ByteTrack
        ↓
tracker_id
        ↓
Bounding Boxes
        ↓
Tracking Labels
        ↓
Object Trajectories
        ↓
Processed Video
```

This approach isolates the tracking stage from object-classification uncertainty and makes ByteTrack behavior easier to study.

---

## Practical Input

The input video is:

```text
practical/assets/input/tracking_demo.mp4
```

Video properties:

```text
Duration: 10 seconds
Resolution: 960 × 540
Frame Rate: 30 FPS
Total Frames: 300
Format: MP4
```

---

## Practical Implementation

The main implementation is:

[`object_tracking_practical.py`](./04-Object-Tracking/practical/object_tracking_practical.py)

The script performs:

1. Video loading
2. Frame-by-frame processing
3. Synthetic detection generation
4. `sv.Detections` creation
5. ByteTrack processing
6. `tracker_id` assignment
7. Bounding-box annotation
8. Tracking label generation
9. Object trajectory visualization
10. Frame information annotation
11. Video export

---

## Practical Result

The practical successfully tracks three moving objects:

```text
object_a #1
object_b #2
object_c #3
```

The tracking pipeline processed:

```text
Width: 960
Height: 540
FPS: 30.0
Frames: 300
```

and completed successfully.

The initial OpenCV output was converted to H.264 for browser compatibility.

Final output:

[`tracked_demo_h264.mp4`](./04-Object-Tracking/practical/assets/output/tracked_demo_h264.mp4)

---

## Google Colab Validation

The practical was executed and verified in Google Colab.

Environment:

```text
OpenCV: 5.0.0
NumPy: 2.0.2
Supervision: 0.30.0
```

Validation result:

```text
Environment test: SUCCESS
```

The final H.264 video was displayed and visually inspected directly inside Google Colab.

---

## Lesson Materials

- [Main Lesson Documentation](./04-Object-Tracking/README.md)
- [Concept Documentation](./04-Object-Tracking/concepts/)
- [Practical Exercises](./04-Object-Tracking/practical/)
- [Practical Documentation](./04-Object-Tracking/practical/README.md)
- [Practical Python Script](./04-Object-Tracking/practical/object_tracking_practical.py)
- [Input Assets](./04-Object-Tracking/practical/assets/input/)
- [Output Assets](./04-Object-Tracking/practical/assets/output/)
- [Final Tracking Video](./04-Object-Tracking/practical/assets/output/tracked_demo_h264.mp4)
- [Class Recording Documentation](./04-Object-Tracking/CLASS-RECORDING.md)
- [Watch the Object Tracking Class Recording on YouTube](https://youtu.be/UXN0l33NqF4)

**Status:** Completed

---

# Organization

Each course session may contain:

```text
session-name/
│
├── concepts/
│   └── Detailed theoretical explanations
│
├── practical/
│   ├── README.md
│   ├── Practical Python implementations
│   │
│   └── assets/
│       ├── input/
│       └── output/
│
├── course-notebook.ipynb
│   └── Original Jupyter / Google Colab notebook
│
├── CLASS-RECORDING.md
│   └── Class recording and related information
│
└── README.md
    └── Main session documentation
```

Some earlier sessions use:

```text
practical-exercises/
```

instead of:

```text
practical/
```

The exact structure may vary depending on the material covered during each class.

---

# Learning Workflow

The course material is organized into a progressive learning workflow:

```text
Class Session
      ↓
Class Recording
      ↓
Original Notebook
      ↓
Course Notes
      ↓
Concept Documentation
      ↓
Code Examples
      ↓
Practical Exercises
      ↓
Testing and Validation
      ↓
Complete Projects
```

This allows the original course material to evolve into a structured and reusable learning resource.

---

# Course Notes vs. Examples vs. Projects

The repository separates different types of learning material.

## Course Notes

```text
08-course-notes/
```

Contains:

- Detailed explanations
- Concepts
- Class material
- Practical exercises
- Original notebooks
- Class recordings
- Practical assets
- Tested outputs

## Code Examples

```text
04-examples/
```

Contains small, focused, runnable Python examples demonstrating individual concepts.

## Projects

```text
05-projects/
```

Contains larger applications that combine multiple concepts into complete computer vision workflows.

The relationship is:

```text
Course Notes
     ↓
Understand the Concept
     ↓
Code Examples
     ↓
Practice the Concept
     ↓
Practical Exercises
     ↓
Experiment
     ↓
Test and Validate
     ↓
Projects
     ↓
Build Complete Applications
```

---

# Technologies and Concepts

The course notes currently cover technologies and concepts including:

- Python
- Computer Vision
- OpenCV
- NumPy
- Matplotlib
- Ultralytics
- YOLOv8
- Supervision
- `sv.Detections`
- Object detection
- Object tracking
- Multi-object tracking
- ByteTrack
- `sv.ByteTrack`
- `tracker_id`
- Object association
- Persistent tracking identities
- Bounding boxes
- Confidence scores
- Confidence thresholds
- Detection labels
- Tracking labels
- Boolean masks
- Detection filtering
- Class filtering
- Size filtering
- Spatial filtering
- Bounding-box area
- Bounding-box center calculation
- Detection merging
- Non-Maximum Suppression (NMS)
- Intersection over Union (IoU)
- Top-N detection selection
- Supervision Annotators
- `BoxAnnotator`
- `LabelAnnotator`
- `TraceAnnotator`
- Annotation customization
- Annotation layers
- Multi-Annotator pipelines
- Detection post-processing
- Video processing
- Frame-by-frame processing
- Object trajectories
- Tracking visualization
- Tracking analytics
- Video export
- H.264
- FFmpeg
- Google Colab
- AI-assisted programming

As the course progresses, this list will expand toward more advanced computer vision and **Segment Anything Model 3 (SAM3)** workflows.

---

# Current Progress

| # | Course Session | Notes | Concepts | Exercises | Status |
|---|---|---|---|---|---|
| 00 | Agentic AI Programming | Completed | Completed | Completed | Completed |
| 01 | Introduction to Supervision | Completed | Completed | Completed | Completed |
| 02 | Annotation and Visualization | Completed | Completed | Completed | Completed |
| 03 | Filtering and Manipulating Detections | Completed | Completed | Completed | Completed |
| 04 | Object Tracking | Completed | Completed | Completed | Completed |

---

# Progress Overview

```text
00 — Agentic AI Programming                 Completed
01 — Introduction to Supervision            Completed
02 — Annotation and Visualization           Completed
03 — Filtering and Manipulating Detections  Completed
04 — Object Tracking                        Completed
05 — Next Course Session                    Upcoming
```

**Completed course sessions: 5**

---

# Purpose

The goal of these notes is not only to preserve the course material, but also to document my learning process and build a reusable reference for:

- Computer Vision
- Supervision
- YOLO
- Segment Anything Model 3 (SAM3)
- AI-assisted development
- Model inference and evaluation
- Computer vision pipelines
- Detection visualization
- Annotation systems
- Detection filtering
- Detection post-processing
- Non-Maximum Suppression
- Spatial analysis
- Object tracking
- Multi-object tracking
- Persistent object identities
- Object trajectories
- Tracking analytics
- Video processing
- Practical AI development

Each completed session expands the repository from basic concepts toward complete computer vision applications.

---

# Repository Learning Progression

The overall learning progression can now be represented as:

```text
Agentic AI Programming
        ↓
Introduction to Supervision
        ↓
Annotation and Visualization
        ↓
Filtering and Manipulating Detections
        ↓
Object Tracking
        ↓
Real-World Detection + Tracking
        ↓
Advanced Computer Vision Concepts
        ↓
SAM3 Workflows
        ↓
Complete AI Applications
```

The first five sessions establish an increasingly complete computer vision pipeline:

```text
AI-Assisted Development
        ↓
Object Detection
        ↓
Detection Visualization
        ↓
Detection Filtering
        ↓
Object Tracking
        ↓
Tracking Analytics
        ↓
Advanced Vision Pipelines
```

---

# Next Learning Direction

After completing Object Tracking, the next natural extension is to combine real-world YOLO detections with ByteTrack.

The extended workflow becomes:

```text
Real-World Video
        ↓
YOLO
        ↓
sv.Detections
        ↓
Detection Filtering
        ↓
ByteTrack
        ↓
tracker_id
        ↓
Tracking Annotations
        ↓
Object Trajectories
        ↓
Tracking Analytics
        ↓
Processed Video
```

This connects the detection, filtering, annotation, and tracking concepts developed throughout the course into a single video-based computer vision pipeline.

---

# Author

**Peyman Miyandashti**

SAM3 Computer Vision Learning Journey  
Computer Vision · Artificial Intelligence · Machine Learning

[LinkedIn](https://www.linkedin.com/in/peyman-mxli/) | [GitHub](https://github.com/Peyman-mxli)
