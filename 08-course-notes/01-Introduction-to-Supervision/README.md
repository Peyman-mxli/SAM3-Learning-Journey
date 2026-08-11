# 01 — Introduction to Supervision

## Computer Vision with Supervision and YOLO

This chapter documents the first practical introduction to **Supervision**, a Python library developed by Roboflow for building computer vision applications.

The main objective is to understand how Supervision provides a common interface for working with detections produced by different computer vision models.

---

## What We Will Build

By the end of this chapter, we will have a complete object detection and annotation pipeline:

```text
Image
  ↓
YOLO Model
  ↓
sv.Detections
  ↓
Annotators
  ↓
Annotated Image

The workflow can also be summarized as:

Image → YOLO → sv.Detections → Annotated Image
Learning Objectives

During this chapter, we will learn how to:

Understand what problem Supervision solves.
Install and import the required computer vision libraries.
Load and process images with OpenCV.
Understand bounding boxes and their coordinates.
Load a pretrained YOLO model.
Perform object detection with YOLO.
Convert YOLO results into sv.Detections.
Inspect detected objects, confidence scores, and class IDs.
Annotate images using Supervision.
Experiment with confidence thresholds.
Compare different YOLO model sizes.
Build a complete computer vision pipeline.
Estimated Class Structure
Section	Estimated Time
Introduction and Core Concepts	15 min
Development and Practical Demonstration	25 min
Analysis and Edge Cases	20 min
Total	~1 hour
1. The Problem Supervision Solves

Computer vision applications can use many different models and frameworks, including:

YOLO
SAM
Detectron2
Transformers

The problem is that each framework can return its predictions in a different format.

This makes it difficult to create reusable computer vision pipelines.

Supervision solves this problem by providing a standardized representation called:

sv.Detections

Conceptually:

YOLO results ─────────┐
SAM results ──────────┼──→ sv.Detections ──→ Annotators
Transformers results ─┘                     ├──→ Trackers
                                           └──→ Zones

Think of Supervision as a universal adapter.

Different models may produce different output formats, but Supervision converts those results into a common structure that the rest of our application can understand.

Why sv.Detections Matters

Once predictions have been converted into sv.Detections, the rest of the computer vision pipeline can remain largely independent of the model that generated them.

For example:

detections = sv.Detections.from_ultralytics(results)

After this conversion, Supervision can be used to:

Inspect bounding boxes
Read confidence scores
Identify detected classes
Draw annotations
Track objects
Work with detection zones

This separation makes computer vision applications easier to build, maintain, and extend.

Core Idea

Instead of designing the entire application around one specific model:

Application → YOLO-specific code

we can design it around a common detection representation:

Model
  ↓
sv.Detections
  ↓
Computer Vision Application

This is one of the fundamental concepts we will use throughout the rest of the course.


This section follows the notebook's explanation that different models produce incompatible result formats and that Supervision standardizes them through `sv.Detections`. :contentReference[oaicite:1]{index=1}

### Then

Click **Commit changes...**

Commit message:

```text
Add Supervision introduction and learning objectives

# 2. Environment Setup

Before building the computer vision pipeline, we need to prepare the Python environment and install the required libraries.

The main libraries used in this notebook are:

- **Supervision** — provides standardized tools for detections, annotations, tracking, and computer vision workflows.
- **Ultralytics** — provides the YOLO models used for object detection.
- **OpenCV** — used for image loading and image processing.
- **Matplotlib** — used to display images inside the notebook.
- **NumPy** — provides numerical array operations.

---

## 2.1 Install Supervision and Ultralytics

The notebook installs the main dependencies with:

```python
!pip install supervision ultralytics
```

This command installs both the Supervision library and the Ultralytics package required to use YOLO.

> In Google Colab, commands beginning with `!` are executed as shell commands rather than normal Python code.

---

## 2.2 Import the Required Libraries

After installation, we import the libraries that will be used throughout the notebook:

```python
import supervision as sv
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
```

Each library has a specific role in the pipeline.

| Library | Purpose |
|---|---|
| `supervision` | Standardizes detections and provides annotation utilities |
| `ultralytics` | Loads and runs YOLO object detection models |
| `cv2` | Image processing with OpenCV |
| `matplotlib` | Displays images and results |
| `numpy` | Numerical and array operations |

---

## 2.3 Verify the Installation

A simple way to verify that Supervision was installed correctly is to print its version:

```python
print("Supervision version:", sv.__version__)
```

If a version number is displayed without an error, the library has been imported successfully.

---

## 2.4 Google Colab Runtime

For computer vision and deep learning workloads, Google Colab can provide access to GPU acceleration.

The course environment uses a **T4 GPU**.

In Google Colab:

```text
Runtime
   ↓
Change runtime type
   ↓
Hardware accelerator
   ↓
T4 GPU
```

The GPU can be verified with:

```python
!nvidia-smi
```

A successful configuration should show an NVIDIA GPU such as:

```text
Tesla T4
```

---

## 2.5 Why Use a GPU?

YOLO performs neural-network inference over images.

Although YOLO can run on a CPU, a GPU can process the mathematical operations required by deep-learning models much faster.

Conceptually:

```text
Input Image
     ↓
YOLO Neural Network
     ↓
GPU Processing
     ↓
Object Predictions
```

This becomes especially important when processing:

- Large images
- Many images
- Video
- Larger YOLO models
- Real-time computer vision applications

---

## 2.6 Environment Ready

At this point, our development environment contains the main components needed for the practical exercises:

```text
Google Colab
     │
     ├── Python
     ├── T4 GPU
     ├── Supervision
     ├── Ultralytics / YOLO
     ├── OpenCV
     ├── Matplotlib
     └── NumPy
```

With the environment prepared, the next step is to understand one of the most important concepts in object detection:

**bounding boxes and their coordinates.**

# 3. Understanding Bounding Boxes

One of the fundamental concepts in object detection is the **bounding box**.

A bounding box is a rectangle used to represent the position of an object detected inside an image.

For example, if YOLO detects a person, car, dog, or another object, the model returns coordinates describing where that object is located.

---

## 3.1 Bounding Box Coordinates

A bounding box can be represented using four coordinates:

```text
[x1, y1, x2, y2]
```

Where:

| Coordinate | Meaning |
|---|---|
| `x1` | Left position of the bounding box |
| `y1` | Top position of the bounding box |
| `x2` | Right position of the bounding box |
| `y2` | Bottom position of the bounding box |

Visually:

```text
(x1, y1)
    ●─────────────────────┐
    │                     │
    │       OBJECT        │
    │                     │
    └─────────────────────●
                       (x2, y2)
```

These coordinates allow the computer to determine exactly where an object appears in an image.

---

## 3.2 Image Coordinate System

Image coordinates work slightly differently from a traditional mathematical graph.

The origin `(0, 0)` is located in the **top-left corner** of the image.

```text
(0,0) ─────────────────────────→ X
  │
  │
  │        IMAGE
  │
  │
  ↓
  Y
```

Therefore:

- `x` increases from **left to right**
- `y` increases from **top to bottom**

Understanding this coordinate system is essential when working with OpenCV, YOLO, and Supervision.

---

## 3.3 Example Bounding Box

Suppose an object is represented by:

```python
box = [100, 50, 400, 300]
```

This means:

```text
x1 = 100
y1 = 50
x2 = 400
y2 = 300
```

The bounding box begins at:

```text
(100, 50)
```

and finishes at:

```text
(400, 300)
```

---

## 3.4 Width and Height

Using the bounding-box coordinates, we can calculate its width and height.

```python
width = x2 - x1
height = y2 - y1
```

For our example:

```python
x1, y1, x2, y2 = 100, 50, 400, 300

width = x2 - x1
height = y2 - y1

print("Width:", width)
print("Height:", height)
```

Output:

```text
Width: 300
Height: 250
```

---

## 3.5 Bounding Boxes with Supervision

Supervision represents object detections using the `sv.Detections` structure.

A detection can contain information such as:

```text
Bounding Box
     │
     ├── Coordinates
     ├── Confidence
     ├── Class ID
     └── Additional detection data
```

The bounding-box coordinates are commonly available through:

```python
detections.xyxy
```

The name `xyxy` represents:

```text
x1, y1, x2, y2
```

For example:

```python
print(detections.xyxy)
```

could produce data conceptually similar to:

```text
[
    [100, 50, 400, 300],
    [500, 120, 700, 450]
]
```

Each row represents one detected object's bounding box.

---

## 3.6 YOLO + Supervision

YOLO is responsible for detecting the objects.

Supervision helps us work with the resulting detections.

```text
Image
   ↓
YOLO Model
   ↓
Predictions
   ↓
sv.Detections
   ↓
Bounding Boxes
   ↓
Annotations / Analysis
```

This separation is useful because the model performs the inference while Supervision provides convenient tools for processing and visualizing the results.

---

## 3.7 Why Bounding Boxes Matter

Bounding boxes are used in many computer vision applications, including:

- Object detection
- Object tracking
- Vehicle detection
- People detection
- Security cameras
- Traffic analysis
- Industrial inspection
- Sports analytics
- Counting objects

They also provide the foundation for more advanced computer vision tasks.

---

## 3.8 Key Concept

Remember:

```text
x1 = LEFT
y1 = TOP

x2 = RIGHT
y2 = BOTTOM
```

or simply:

```text
[x1, y1, x2, y2]
```

This format is commonly called:

```text
XYXY
```

Understanding `XYXY` will make the next steps with `sv.Detections`, YOLO predictions, and annotations much easier.
