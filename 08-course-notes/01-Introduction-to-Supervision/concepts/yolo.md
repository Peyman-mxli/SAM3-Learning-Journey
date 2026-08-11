# YOLO Object Detection

## What is YOLO?

**YOLO (You Only Look Once)** is a family of computer vision models designed for fast object detection.

YOLO analyzes an image and identifies objects by predicting:

- Bounding boxes
- Object classes
- Confidence scores

---

## Object Detection

Object detection answers two main questions:

1. **What object is present?**
2. **Where is the object located?**

For example, an image may contain:

```text
Person → 95% confidence
Car    → 91% confidence
Dog    → 87% confidence

Each detected object is associated with a bounding box.

YOLO Detection Pipeline

A typical detection workflow is:

Input Image
     ↓
YOLO Model
     ↓
Model Inference
     ↓
Detected Objects
     ↓
Bounding Boxes
     ↓
Class IDs
     ↓
Confidence Scores
     ↓
Supervision
     ↓
Visualization
YOLO with Supervision

YOLO performs the object detection, while Supervision helps organize, filter, annotate, and visualize the results.

Example:

from ultralytics import YOLO
import supervision as sv

model = YOLO("yolo11n.pt")

results = model("image.jpg")[0]

detections = sv.Detections.from_ultralytics(results)

Now the YOLO predictions are represented as a Supervision Detections object.

Detection Information

A YOLO detection normally contains information such as:

Bounding Box
Class ID
Class Name
Confidence Score

Example:

Object: Person
Confidence: 0.94
Bounding Box: [120, 80, 350, 500]
Why YOLO?

YOLO is widely used because it can perform object detection quickly and efficiently.

Common applications include:

People detection
Vehicle detection
Security systems
Industrial inspection
Robotics
Autonomous systems
Video analysis
Real-time computer vision
Key Idea

YOLO detects the objects. Supervision helps us work with the detection results.
