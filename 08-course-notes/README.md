# SAM3 Course Notes

This directory contains my organized notes, concepts, practical exercises, notebooks, class recordings, and supporting material from the **SAM3 — Computer Vision with Segment Anything Model 3** course.

The purpose of this section is to document each course session in a structured way, transforming the original class material into a reusable computer vision learning reference.

---

## Course Sessions

| # | Session | Status |
|---|---|---|
| 00 | [Agentic AI Programming](./00-Agentic-AI-Programming/) | ✅ Completed |
| 01 | [Introduction to Supervision](./01-Introduction-to-Supervision/) | ✅ Completed |
| 02 | [Annotation and Visualization](./02-Annotation-and-Visualization/) | ✅ Completed |
| 03 | [Filtering and Manipulating Detections](./03-Filtering-and-Manipulating-Detections/) | ✅ Completed |

---

## Session 00 — Agentic AI Programming

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

## Session 01 — Introduction to Supervision

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

## Session 02 — Annotation and Visualization

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

## Session 03 — Filtering and Manipulating Detections

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

### Detection Filtering Example

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

### Lesson Materials

- [Main Lesson Documentation](./03-Filtering-and-Manipulating-Detections/README.md)
- [Concept Documentation](./03-Filtering-and-Manipulating-Detections/concepts/)
- [Practical Exercises](./03-Filtering-and-Manipulating-Detections/practical-exercises/)
- [Original Course Notebook](./03-Filtering-and-Manipulating-Detections/02_a_filtrado_detecciones.ipynb)
- [Class Recording](./03-Filtering-and-Manipulating-Detections/CLASS-RECORDING.md)

**Status:** Completed

---

## Organization

Each course session may contain:

```text
session-name/
│
├── concepts/
│   └── Detailed theoretical explanations
│
├── practical-exercises/
│   └── Hands-on exercises
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

The exact structure may vary depending on the material covered during each class.

---

## Learning Workflow

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
Complete Projects
```

This allows the original course material to evolve into a structured and reusable learning resource.

---

## Course Notes vs. Examples vs. Projects

The repository separates different types of learning material.

### Course Notes

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

### Code Examples

```text
04-examples/
```

Contains small, focused, runnable Python examples demonstrating individual concepts.

### Projects

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
Projects
     ↓
Build Complete Applications
```

---

## Technologies and Concepts

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
- Bounding boxes
- Confidence scores
- Confidence thresholds
- Detection labels
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
- Annotation customization
- Annotation layers
- Multi-Annotator pipelines
- Detection post-processing
- Google Colab
- AI-assisted programming

As the course progresses, this list will expand toward more advanced computer vision and **Segment Anything Model 3 (SAM3)** workflows.

---

## Current Progress

| # | Course Session | Notes | Concepts | Exercises | Status |
|---|---|---|---|---|---|
| 00 | Agentic AI Programming | ✅ | ✅ | ✅ | Completed |
| 01 | Introduction to Supervision | ✅ | ✅ | ✅ | Completed |
| 02 | Annotation and Visualization | ✅ | ✅ | ✅ | Completed |
| 03 | Filtering and Manipulating Detections | ✅ | ✅ | ✅ | Completed |

---

## Progress Overview

```text
00 — Agentic AI Programming                ✅ Completed
01 — Introduction to Supervision           ✅ Completed
02 — Annotation and Visualization          ✅ Completed
03 — Filtering and Manipulating Detections ✅ Completed
04 — Next Course Session                   ⏳ Upcoming
```

**Completed course sessions: 4**

---

## Purpose

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
- Practical AI development

Each completed session expands the repository from basic concepts toward complete computer vision applications.

---

## Repository Learning Progression

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
Advanced Computer Vision Concepts
        ↓
SAM3 Workflows
        ↓
Complete AI Applications
```

---

## Author

**Peyman Miyandashti**

SAM3 Computer Vision Learning Journey  
Computer Vision · Artificial Intelligence · Machine Learning

[LinkedIn](https://www.linkedin.com/in/peyman-mxli/) | [GitHub](https://github.com/Peyman-mxli)
