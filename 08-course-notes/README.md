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
- Supervision Annotators
- Annotation customization
- Annotation layers
- Multi-Annotator pipelines
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

---

## Progress Overview

```text
00 — Agentic AI Programming          ✅ Completed
01 — Introduction to Supervision     ✅ Completed
02 — Annotation and Visualization    ✅ Completed
03 — Next Course Session             ⏳ Upcoming
```

**Completed course sessions: 3**

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
- Practical AI development

Each completed session expands the repository from basic concepts toward complete computer vision applications.

---

## Repository Learning Progression

The overall learning progression can be represented as:

```text
Agentic AI Programming
        ↓
Introduction to Supervision
        ↓
Annotation and Visualization
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
