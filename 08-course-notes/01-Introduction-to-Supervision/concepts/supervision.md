# Supervision

## What is Supervision?

**Supervision** is an open-source Python library designed to simplify common computer vision tasks.

It provides reusable tools for working with:

- Object detection
- Bounding boxes
- Labels
- Confidence scores
- Image annotations
- Detection filtering
- Computer vision pipelines

Instead of manually implementing many visualization and detection-processing functions, Supervision provides structured utilities that make computer vision code easier to read, maintain, and reuse.

---

## Supervision in Our Course

In the SAM3 course, Supervision is useful for processing and visualizing the results produced by computer vision models.

A typical workflow can look like:

```text
Image
   ↓
Computer Vision Model
   ↓
Predictions
   ↓
sv.Detections
   ↓
Filtering / Processing
   ↓
Annotations
   ↓
Final Visualization

Importing Supervision

The library is normally imported using:

import supervision as sv

The alias:

sv

is then used to access Supervision classes and utilities.

Example:

import supervision as sv

detections = sv.Detections(...)
Why Use Supervision?

Supervision helps separate the different stages of a computer vision pipeline.

Instead of mixing model inference, detection processing, filtering, and visualization into one large block of code, each operation can be handled independently.

This makes the project:

Easier to understand
Easier to debug
Easier to modify
More reusable
Better organized
Key Idea

The model generates predictions. Supervision helps us organize, process, filter, and visualize those predictions.

Related Concepts
YOLO
sv.Detections
Bounding Boxes
Confidence Scores
Detection Filtering
Annotations

