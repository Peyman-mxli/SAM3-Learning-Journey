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


---


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

# 4. Understanding `sv.Detections`

One of the most important structures in the Supervision library is:

```python
sv.Detections
```

`sv.Detections` is used to store and manage the results produced by an object detection model.

Instead of manually handling separate arrays for bounding boxes, confidence scores, and class IDs, Supervision organizes this information into one convenient structure.

---

## 4.1 What Does `sv.Detections` Store?

A detection can contain several pieces of information:

```text
sv.Detections
│
├── xyxy
│   └── Bounding-box coordinates
│
├── confidence
│   └── Confidence score
│
├── class_id
│   └── Detected class
│
├── tracker_id
│   └── Tracking identifier
│
└── data
    └── Additional information
```

The most important attributes when starting are:

```python
detections.xyxy
detections.confidence
detections.class_id
```

---

## 4.2 `xyxy` — Bounding Boxes

The `xyxy` attribute contains the coordinates of every detected object.

```python
detections.xyxy
```

Each detection follows this format:

```text
[x1, y1, x2, y2]
```

Example:

```python
print(detections.xyxy)
```

Possible output:

```text
[
    [100, 50, 400, 300],
    [500, 120, 700, 450]
]
```

This means the model detected two objects.

---

## 4.3 `confidence` — Detection Confidence

Object-detection models normally return a confidence score indicating how confident the model is about a prediction.

Supervision stores these values in:

```python
detections.confidence
```

Example:

```python
print(detections.confidence)
```

Possible output:

```text
[0.94, 0.87, 0.76]
```

These values can be interpreted approximately as:

```text
0.94 → 94% confidence
0.87 → 87% confidence
0.76 → 76% confidence
```

A higher confidence means the model is more certain about the detection.

---

## 4.4 `class_id` — Object Class

Each detected object belongs to a class.

For example:

```text
person
car
dog
bicycle
```

Models normally represent these classes internally using numbers.

Supervision stores those numbers in:

```python
detections.class_id
```

Example:

```python
print(detections.class_id)
```

Possible output:

```text
[0, 2, 16]
```

The meaning of each number depends on the classes used by the model.

Conceptually:

```text
0  → person
2  → car
16 → dog
```

---

## 4.5 One Detection Contains Multiple Values

Suppose the model detects a car.

The information could conceptually look like this:

```text
Detection
│
├── Bounding Box: [120, 80, 500, 350]
├── Confidence:   0.93
└── Class ID:     2
```

Supervision keeps these pieces of information aligned inside the same `Detections` object.

---

## 4.6 Creating `sv.Detections`

A simple example:

```python
import supervision as sv
import numpy as np

detections = sv.Detections(
    xyxy=np.array([
        [100, 50, 400, 300],
        [500, 120, 700, 450]
    ]),
    confidence=np.array([
        0.94,
        0.87
    ]),
    class_id=np.array([
        0,
        2
    ])
)
```

Now we can inspect the detections:

```python
print(detections.xyxy)
print(detections.confidence)
print(detections.class_id)
```

---

## 4.7 Number of Detected Objects

We can determine how many objects were detected with:

```python
len(detections)
```

Example:

```python
print("Number of detections:", len(detections))
```

Possible output:

```text
Number of detections: 2
```

---

## 4.8 Iterating Through Detections

We can also process detected objects individually.

Conceptually:

```python
for detection in detections:
    print(detection)
```

This becomes useful when we want to analyze each detected object separately.

For example, we may want to:

- Check its confidence
- Read its class
- Calculate its position
- Track it across frames
- Count objects
- Apply filters

---

## 4.9 Filtering Detections

One powerful feature of Supervision is the ability to filter detections.

For example, suppose we only want detections with confidence greater than `0.80`.

```python
filtered_detections = detections[
    detections.confidence > 0.80
]
```

Now:

```python
print(len(filtered_detections))
```

returns only the detections that satisfy our condition.

This pattern is extremely useful in computer vision pipelines.

---

## 4.10 Model Output → `sv.Detections`

A typical computer vision workflow looks like this:

```text
Image
   ↓
Object Detection Model
   ↓
Raw Predictions
   ↓
sv.Detections
   ↓
Filter / Analyze
   ↓
Annotate
   ↓
Final Result
```

For models supported by Supervision, predictions can often be converted into a `Detections` object.

Conceptually:

```python
results = model(image)

detections = sv.Detections.from_ultralytics(results[0])
```

After conversion, we can work with:

```python
detections.xyxy
detections.confidence
detections.class_id
```

instead of manually processing the raw model output.

---

## 4.11 Why `sv.Detections` Is Important

`sv.Detections` becomes a central connection between the AI model and the rest of the computer vision pipeline.

```text
YOLO / Detection Model
          │
          ▼
    sv.Detections
          │
    ┌─────┼─────┐
    ▼     ▼     ▼
 Filter  Count  Track
    │     │     │
    └─────┼─────┘
          ▼
      Annotate
          │
          ▼
       Output
```

This allows us to separate **model inference** from **post-processing and visualization**.

---

## 4.12 Key Concepts

Remember these three attributes:

| Attribute | Purpose |
|---|---|
| `detections.xyxy` | Bounding-box coordinates |
| `detections.confidence` | Confidence scores |
| `detections.class_id` | Object class IDs |

The basic idea is:

```text
MODEL
  ↓
DETECTIONS
  ↓
UNDERSTAND
  ↓
FILTER
  ↓
ANNOTATE
  ↓
ANALYZE
```

Understanding `sv.Detections` is essential because we will use it repeatedly when working with Supervision, YOLO, object detection, tracking, and segmentation.


# 5. Annotating Images with Supervision

After a model detects objects and stores the results in `sv.Detections`, the next step is usually to **visualize those detections on the image**.

Supervision provides annotators that make this process simple.

Instead of manually drawing rectangles, text, and labels with OpenCV, we can use Supervision's annotation tools.

---

## 5.1 What Is an Annotator?

An annotator takes:

```text
Original Image
      +
sv.Detections
      ↓
   Annotator
      ↓
Annotated Image
```

The resulting image can display information such as:

- Bounding boxes
- Class names
- Confidence scores
- Tracking IDs
- Points
- Masks
- Other visual information

---

## 5.2 `BoundingBoxAnnotator`

One of the most common annotators is:

```python
sv.BoundingBoxAnnotator
```

It draws bounding boxes around detected objects.

Example:

```python
import supervision as sv

box_annotator = sv.BoundingBoxAnnotator()

annotated_image = box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)
```

The important elements are:

```text
image
   ↓
image.copy()
   ↓
BoundingBoxAnnotator
   +
detections
   ↓
annotated_image
```

---

## 5.3 Why Use `image.copy()`?

Notice that we use:

```python
image.copy()
```

instead of directly modifying:

```python
image
```

This allows us to preserve the original image.

For example:

```python
original_image = image

annotated_image = box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)
```

Now we have:

```text
image
│
├── Original image
│
└── annotated_image
    └── Image with detections
```

This is useful when comparing results or applying different visualization techniques.

---

## 5.4 `LabelAnnotator`

Bounding boxes show **where** an object is located.

Labels help us understand **what** the object is.

Supervision provides:

```python
sv.LabelAnnotator
```

Example:

```python
label_annotator = sv.LabelAnnotator()
```

Then:

```python
annotated_image = label_annotator.annotate(
    scene=annotated_image,
    detections=detections
)
```

Now the image can contain both bounding boxes and labels.

---

## 5.5 Combining Annotators

We can use multiple annotators on the same image.

Example:

```python
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = image.copy()

annotated_image = box_annotator.annotate(
    scene=annotated_image,
    detections=detections
)

annotated_image = label_annotator.annotate(
    scene=annotated_image,
    detections=detections
)
```

The workflow becomes:

```text
Original Image
      ↓
Bounding Boxes
      ↓
Labels
      ↓
Final Annotated Image
```

---

## 5.6 Custom Labels

We can also create our own labels.

For example, suppose our detections contain:

```python
detections.class_id
detections.confidence
```

We can create labels that display both the class and confidence.

Conceptually:

```text
person 0.95
car 0.91
dog 0.87
```

A common pattern is:

```python
labels = [
    f"{class_id} {confidence:.2f}"
    for class_id, confidence
    in zip(detections.class_id, detections.confidence)
]
```

Then pass the labels to the annotator:

```python
annotated_image = label_annotator.annotate(
    scene=annotated_image,
    detections=detections,
    labels=labels
)
```

---

## 5.7 Using Class Names

Class IDs such as:

```text
0
2
16
```

are useful internally, but humans usually prefer names such as:

```text
person
car
dog
```

Suppose we have:

```python
class_names = [
    "person",
    "bicycle",
    "car",
    "motorcycle"
]
```

We could obtain the corresponding name using:

```python
class_names[class_id]
```

A label could then contain:

```python
labels = [
    f"{class_names[class_id]} {confidence:.2f}"
    for class_id, confidence
    in zip(detections.class_id, detections.confidence)
]
```

Possible result:

```text
person 0.95
car 0.91
```

---

## 5.8 Complete Annotation Pipeline

A simplified annotation pipeline could look like this:

```python
import supervision as sv

box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = image.copy()

annotated_image = box_annotator.annotate(
    scene=annotated_image,
    detections=detections
)

annotated_image = label_annotator.annotate(
    scene=annotated_image,
    detections=detections
)
```

Conceptually:

```text
INPUT IMAGE
     │
     ▼
AI MODEL
     │
     ▼
sv.Detections
     │
     ├──────────────┐
     ▼              ▼
BoundingBox      Label
Annotator        Annotator
     │              │
     └──────┬───────┘
            ▼
     ANNOTATED IMAGE
```

---

## 5.9 Displaying the Result

After creating the annotated image, we need to visualize it.

One option is Matplotlib:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.imshow(annotated_image)
plt.axis("off")
plt.show()
```

Supervision also provides visualization utilities that can be useful in computer vision workflows.

---

## 5.10 Detection Visualization Workflow

At this point, we can understand the complete process:

```text
1. Load Image
      ↓
2. Run Detection Model
      ↓
3. Obtain Predictions
      ↓
4. Convert to sv.Detections
      ↓
5. Inspect / Filter Detections
      ↓
6. Create Annotators
      ↓
7. Draw Bounding Boxes
      ↓
8. Add Labels
      ↓
9. Display Result
```

This is one of the fundamental workflows used throughout computer vision.

---

## 5.11 Why Supervision Annotators Are Useful

Without Supervision, we might manually write OpenCV code for:

```text
Rectangle coordinates
Text positioning
Font size
Label placement
Detection iteration
Visualization
```

With Supervision, much of this repetitive visualization logic is already implemented.

This allows us to focus more on:

```text
Model
Data
Detection Logic
Filtering
Tracking
Segmentation
Analysis
```

rather than repeatedly implementing visualization code.

---

## 5.12 Key Concepts

Remember:

```python
sv.BoundingBoxAnnotator()
```

is used to draw bounding boxes.

```python
sv.LabelAnnotator()
```

is used to display labels.

And:

```python
annotator.annotate(
    scene=image,
    detections=detections
)
```

connects the image with the detection results.

The overall idea is:

```text
MODEL
  ↓
sv.Detections
  ↓
ANNOTATORS
  ↓
VISUAL RESULT
```

This gives us the foundation for visualizing the predictions generated by object detection models.

# 6. Exploring Detection Confidence

Object detection models do not simply predict which objects appear in an image.

They also provide a **confidence score** that represents how certain the model is about each prediction.

A confidence score is normally represented between:

```text
0.0 → Very uncertain
1.0 → Very confident
```

For example:

```text
0.95 → 95% confidence
0.82 → 82% confidence
0.51 → 51% confidence
```

---

## 6.1 Inspecting Confidence Scores

Supervision stores the confidence values of detected objects in:

```python
detections.confidence
```

We can inspect them with:

```python
print(detections.confidence)
```

This allows us to understand how confident the model is about each detection.

---

## 6.2 Confidence Threshold

A **confidence threshold** determines how confident the model must be before a prediction is accepted.

For example:

```text
Low threshold
     ↓
More detections
     ↓
Potentially more uncertain predictions
```

While:

```text
High threshold
     ↓
Fewer detections
     ↓
Only more confident predictions remain
```

---

## 6.3 Experiment: Increasing the Confidence Threshold

In the course notebook, we make YOLO more strict by setting:

```python
results_estricto = model(image, conf=0.8)[0]

detections_estricto = sv.Detections.from_ultralytics(
    results_estricto
)
```

The important parameter is:

```python
conf=0.8
```

This tells YOLO to keep detections with a sufficiently high confidence score.

---

## 6.4 Comparing the Results

We can compare the original detections with the stricter configuration:

```python
print(
    f"Con conf=0.5 (por defecto): "
    f"{len(detections)} objetos detectados"
)

print(
    f"Con conf=0.8 (estricto): "
    f"{len(detections_estricto)} objetos detectados"
)
```

Conceptually:

```text
Original Detection
conf = 0.5
     │
     ├── Object A
     ├── Object B
     ├── Object C
     └── Object D

Stricter Detection
conf = 0.8
     │
     ├── Object A
     └── Object B
```

Increasing the confidence requirement can remove detections where the model is less certain.

---

## 6.5 Why Does a Higher Threshold Detect Fewer Objects?

The course asks us to reflect on:

> Why are fewer objects detected when the confidence threshold is higher?

The notebook explains that the model tends to have greater confidence in objects that are:

- Larger
- Clearly visible
- Easier to recognize

Increasing the threshold removes more **doubtful detections**.

---

## 6.6 Inspecting a Specific Detection

`sv.Detections` also supports indexing.

For example:

```python
primera = detections[0]
```

This selects the first detected object.

The result is still an:

```python
sv.Detections
```

object, but containing only one detection.

---

## 6.7 Inspecting Its Bounding Box

We can obtain the coordinates of that detection:

```python
x1, y1, x2, y2 = primera.xyxy[0]
```

Remember:

```text
(x1, y1)
    ┌───────────────────┐
    │                   │
    │      OBJECT       │
    │                   │
    └───────────────────┘
                     (x2, y2)
```

---

## 6.8 Inspecting Class and Confidence

The notebook uses:

```python
print(f"Primera detección:")
print(f"  Clase:     {results.names[primera.class_id[0]]}")
print(f"  Confianza: {primera.confidence[0]:.1%}")
```

This allows us to inspect:

```text
Detection
│
├── Class
├── Confidence
└── Position
```

---

## 6.9 Calculating Detection Size

The width and height of the bounding box can be calculated with:

```python
width = x2 - x1
height = y2 - y1
```

The notebook displays this information with:

```python
print(
    f"Tamaño: {x2-x1:.0f} px de ancho × "
    f"{y2-y1:.0f} px de alto"
)
```

Therefore, one individual detection can provide information such as:

```text
Class:       person
Confidence:  94.2%
Position:    (x1, y1) → (x2, y2)
Width:       ... pixels
Height:      ... pixels
```

---

## 6.10 Why This Matters

Confidence thresholds are important because different computer vision applications have different requirements.

For example:

```text
Security Application
        ↓
May require high-confidence detections

Exploratory Analysis
        ↓
May accept lower-confidence detections
```

There is no single threshold that is ideal for every application.

The appropriate value depends on the problem being solved.

---

## 6.11 Key Concept

The important relationship is:

```text
Lower Confidence Threshold
          ↓
    More Predictions
          ↓
Potentially More Uncertain Results
```

versus:

```text
Higher Confidence Threshold
          ↓
     Fewer Predictions
          ↓
More Confident Results Remain
```

Understanding confidence is essential when evaluating and refining an object detection pipeline.

# 7. YOLO Models and the COCO Dataset

Now that we understand detections, bounding boxes, and confidence scores, the next step is to understand the model generating those predictions.

In this course, we use **YOLO** for object detection.

---

## 7.1 What Is YOLO?

**YOLO** stands for:

> **You Only Look Once**

YOLO is a family of real-time object detection models.

The name describes the basic idea: the model analyzes an image and predicts objects and their locations efficiently in a single detection pipeline.

Conceptually:

```text
Input Image
     ↓
   YOLO
     ↓
Object Detection
     ↓
Bounding Boxes + Classes + Confidence
```

YOLO is especially useful when fast object detection is required.

---

## 7.2 YOLO and Ultralytics

In this notebook, YOLO is loaded through the Ultralytics Python library:

```python
from ultralytics import YOLO
```

We can then create a model with:

```python
model = YOLO("yolov8n.pt")
```

The first time this command is executed, the model weights are downloaded automatically.

---

## 7.3 YOLO Model Sizes

YOLO models are available in different sizes.

The suffix in the model filename indicates its size.

| Suffix | Model | Approx. Size | Speed | Accuracy |
|---|---|---:|:---:|:---:|
| `n` | Nano | ~6 MB | +++ | + |
| `s` | Small | ~22 MB | ++ | ++ |
| `m` | Medium | ~52 MB | + | +++ |

For example:

```text
yolov8n.pt
       ↑
       n = Nano
```

and:

```text
yolov8s.pt
       ↑
       s = Small
```

---

## 7.4 Speed vs. Accuracy

Choosing a model often involves a trade-off:

```text
Smaller Model
     ↓
Faster Inference
     ↓
Lower Computational Cost
```

versus:

```text
Larger Model
     ↓
More Computation
     ↓
Potentially Better Detection Accuracy
```

The course uses:

```python
YOLO("yolov8n.pt")
```

to maximize speed.

If GPU resources are available, larger models can also be tested.

---

# 8. The COCO Dataset

The pretrained YOLOv8 model used in this notebook can recognize objects from the **COCO dataset**.

COCO stands for:

> **Common Objects in Context**

It contains approximately **330,000 real-world photographs** and includes **80 object categories**.

These categories include common objects such as:

```text
person
car
bus
truck
animals
furniture
and many others
```

---

## 8.1 Class IDs

Computer vision models commonly represent categories internally using numerical IDs.

Some classes frequently used in the course examples are:

| `class_id` | Class |
|---:|---|
| `0` | person |
| `2` | car |
| `5` | bus |
| `7` | truck |

Therefore:

```text
class_id = 0
```

means:

```text
person
```

while:

```text
class_id = 5
```

means:

```text
bus
```

---

## 8.2 Translating Class IDs to Names

YOLO provides a dictionary that translates class IDs into human-readable names:

```python
results.names
```

For example:

```python
results.names[0]
```

returns:

```text
person
```

We can inspect the classes detected in an image with:

```python
for class_id in sorted(set(detections.class_id)):
    print(
        f"Class {class_id}: "
        f"{results.names[class_id]}"
    )
```

This connects:

```text
Numerical Prediction
        ↓
     class_id
        ↓
   results.names
        ↓
Human-Readable Class
```

---

# 9. Loading the YOLO Model

The notebook loads the Nano model with:

```python
model = YOLO("yolov8n.pt")
```

The `n` version is useful because it is relatively small and fast.

Conceptually:

```text
yolov8n.pt
     ↓
Ultralytics YOLO
     ↓
Loaded Detection Model
     ↓
Ready for Inference
```

---

## 9.1 Running Object Detection

Once the model has been loaded, we can run it on an image:

```python
results = model(image)[0]
```

The model returns a list of results.

The:

```python
[0]
```

selects the first result.

For a single image:

```text
Image
  ↓
model(image)
  ↓
List of Results
  ↓
[0]
  ↓
First Image Result
```

---

## 9.2 Converting YOLO Results to Supervision

This is where YOLO and Supervision connect.

The raw YOLO result is converted with:

```python
detections = sv.Detections.from_ultralytics(results)
```

The pipeline now becomes:

```text
IMAGE
  ↓
YOLO
  ↓
Ultralytics Result
  ↓
sv.Detections.from_ultralytics()
  ↓
sv.Detections
  ↓
Analysis / Filtering / Annotation
```

This conversion is one of the central ideas of the lesson.

Supervision gives us a common representation that can be used after the model performs inference.

---

# 10. Experiment — Changing the YOLO Model

The notebook also compares the Nano model with the Small model.

First:

```python
model_s = YOLO("yolov8s.pt")
```

Then:

```python
results_s = model_s(image)[0]

detections_s = sv.Detections.from_ultralytics(
    results_s
)
```

Now we can compare them:

```python
print(
    f"yolov8n (nano):  "
    f"{len(detections)} objects"
)

print(
    f"yolov8s (small): "
    f"{len(detections_s)} objects"
)
```

---

## 10.1 What Are We Comparing?

The objective is not simply to ask:

```text
Which model detected more objects?
```

We should also consider:

```text
Did both models detect the same objects?

Did their confidence scores change?

Did the larger model detect smaller objects?

Did it detect partially occluded objects?

Was inference slower?
```

The notebook notes that a larger model may detect more small or occluded objects.

---

## 10.2 Important Lesson

Changing the YOLO model does **not** require us to redesign the Supervision portion of the pipeline.

For example:

```text
YOLO Nano ────┐
              │
YOLO Small ───┼──→ sv.Detections → Annotation
              │
Other Model ──┘
```

This demonstrates why a standardized detection representation is useful.

The model can change while much of the downstream processing remains the same.

---

## 10.3 Complete Architecture So Far

We can now understand the architecture of our first object detection application:

```text
              INPUT IMAGE
                   │
                   ▼
             YOLO MODEL
                   │
                   ▼
          ULTRALYTICS RESULT
                   │
                   ▼
     sv.Detections.from_ultralytics()
                   │
                   ▼
            sv.Detections
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
    Bounding    Confidence   Class ID
      Boxes       Scores
        │          │          │
        └──────────┼──────────┘
                   ▼
               FILTERING
                   │
                   ▼
              ANNOTATORS
                   │
                   ▼
            ANNOTATED IMAGE
```

This is the foundation of the practical Supervision pipeline used throughout this chapter.


# 11. Complete Practical Pipeline

Now we will combine everything we have learned into a complete computer vision pipeline.

Our objective is:

```text
Download Image
      ↓
Load with OpenCV
      ↓
Run YOLO
      ↓
Convert to sv.Detections
      ↓
Inspect Predictions
      ↓
Annotate Image
      ↓
Display Final Result
```

---

## 11.1 Download the Test Image

For this example, we use a public image provided by Ultralytics.

First, import `urllib.request` and create an `assets` directory:

```python
import urllib.request
from pathlib import Path

Path("assets").mkdir(exist_ok=True)
```

Then download the image:

```python
urllib.request.urlretrieve(
    "https://ultralytics.com/images/bus.jpg",
    "assets/bus.jpg"
)
```

The image will be stored locally as:

```text
assets/
└── bus.jpg
```

---

## 11.2 Load the Image with OpenCV

Now we load the downloaded image:

```python
image = cv2.imread("assets/bus.jpg")
```

We can inspect its dimensions:

```python
print(f"Imagen cargada: {image.shape}")
```

The shape of an OpenCV image follows:

```text
(height, width, channels)
```

For a standard color image:

```text
channels = 3
```

---

## 11.3 BGR vs RGB

An important detail when using OpenCV is that images are loaded using:

```text
BGR
```

rather than:

```text
RGB
```

The channel order is:

```text
OpenCV
B → Blue
G → Green
R → Red
```

Matplotlib normally expects:

```text
R → Red
G → Green
B → Blue
```

Therefore, before displaying an OpenCV image with Matplotlib, we convert it:

```python
cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```

---

## 11.4 Display the Original Image

We can visualize the image with:

```python
plt.figure(figsize=(12, 7))

plt.imshow(
    cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )
)

plt.axis("off")
plt.title("Imagen de prueba")
plt.show()
```

At this point, we have successfully completed:

```text
Internet
   ↓
Download Image
   ↓
OpenCV
   ↓
NumPy Image
   ↓
Matplotlib
   ↓
Displayed Image
```

---

# 12. Run YOLO Object Detection

Now we load the pretrained YOLO Nano model:

```python
model = YOLO("yolov8n.pt")
```

The model weights are downloaded automatically the first time they are required.

Then we run inference:

```python
results = model(image)[0]
```

Conceptually:

```text
bus.jpg
   ↓
OpenCV Image
   ↓
YOLOv8 Nano
   ↓
Object Predictions
```

---

# 13. Convert YOLO Predictions to `sv.Detections`

YOLO produces its own result structure.

We convert it into the standardized Supervision format:

```python
detections = sv.Detections.from_ultralytics(results)
```

Now the pipeline becomes:

```text
Image
  ↓
YOLO
  ↓
Ultralytics Results
  ↓
sv.Detections.from_ultralytics()
  ↓
sv.Detections
```

From this point onward, our processing can use the Supervision API.

---

## 13.1 Inspect the Number of Objects

We can determine how many objects were detected:

```python
print(
    f"Número de objetos detectados: "
    f"{len(detections)}"
)
```

---

## 13.2 Inspect Bounding Boxes

The bounding boxes are stored in:

```python
detections.xyxy
```

We can display them with:

```python
print(
    "\n--- xyxy: coordenadas del bounding box ---"
)

print(
    "Formato: "
    "[x_izquierda, y_arriba, x_derecha, y_abajo]"
)

print(detections.xyxy)
```

Each row represents one detected object:

```text
[
    [x1, y1, x2, y2],
    [x1, y1, x2, y2],
    ...
]
```

---

## 13.3 Inspect Confidence Scores

Confidence values are available through:

```python
detections.confidence
```

We can inspect them with:

```python
print(
    "\n--- confidence: certeza del modelo "
    "(0 = inseguro, 1 = muy seguro) ---"
)

print(detections.confidence)
```

The closer the value is to:

```text
1.0
```

the more confident the model is about that prediction.

---

## 13.4 Inspect Class IDs

The detected class IDs are stored in:

```python
detections.class_id
```

Display them with:

```python
print(
    "\n--- class_id: número de la "
    "categoría detectada ---"
)

print(detections.class_id)
```

---

## 13.5 Translate Class IDs to Names

Numbers are useful to the model, but humans usually want class names.

YOLO provides:

```python
results.names
```

We can translate the detected IDs with:

```python
for class_id in sorted(
    set(detections.class_id)
):
    print(
        f"Clase {class_id}: "
        f"{results.names[class_id]}"
    )
```

The transformation is:

```text
class_id
   ↓
results.names[class_id]
   ↓
Human-readable class
```

For example:

```text
0 → person
2 → car
5 → bus
7 → truck
```

---

# 14. Annotate the Image

Now we visualize the predictions.

Create the annotators:

```python
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
```

---

## 14.1 Create Detection Labels

We want each detection to display:

```text
Class Name + Confidence
```

For example:

```text
person 94%
bus 91%
```

The notebook creates the labels with:

```python
labels = [
    f"{results.names[class_id]} {conf:.0%}"
    for class_id, conf in zip(
        detections.class_id,
        detections.confidence
    )
]
```

---

## 14.2 Draw Bounding Boxes

First, draw the bounding boxes:

```python
annotated = box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)
```

Notice the use of:

```python
image.copy()
```

This is important because it prevents the original image from being modified.

---

## 14.3 Add Labels

Next, add the labels:

```python
annotated = label_annotator.annotate(
    scene=annotated,
    detections=detections,
    labels=labels
)
```

Our visualization pipeline is now:

```text
Original Image
      ↓
image.copy()
      ↓
BoxAnnotator
      ↓
Bounding Boxes
      ↓
LabelAnnotator
      ↓
Bounding Boxes + Labels
```

---

# 15. Display the Final Result

Finally, display the annotated image:

```python
plt.figure(figsize=(12, 7))

plt.imshow(
    cv2.cvtColor(
        annotated,
        cv2.COLOR_BGR2RGB
    )
)

plt.axis("off")

plt.title(
    "Pipeline completo: "
    "detección + anotación con Supervision"
)

plt.show()
```

The final output contains:

```text
Original Image
      +
YOLO Predictions
      +
Bounding Boxes
      +
Class Names
      +
Confidence Scores
      ↓
Annotated Image
```

---

# 16. Complete Pipeline Summary

We have now created our first complete YOLO + Supervision computer vision pipeline:

```text
                  INTERNET
                     │
                     ▼
                Download Image
                     │
                     ▼
                  OpenCV
                     │
                     ▼
                 YOLOv8
                     │
                     ▼
            Ultralytics Result
                     │
                     ▼
       sv.Detections.from_ultralytics()
                     │
                     ▼
               sv.Detections
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
       xyxy      confidence    class_id
        │            │            │
        └────────────┼────────────┘
                     ▼
                  Labels
                     │
                     ▼
               BoxAnnotator
                     │
                     ▼
              LabelAnnotator
                     │
                     ▼
              Annotated Image
                     │
                     ▼
                Matplotlib
```

This demonstrates the central idea of the lesson:

```text
Image → Model → sv.Detections → Annotated Image
```

Supervision acts as the bridge between the model's predictions and the tools we use to analyze and visualize those predictions.


# 17. Extension Challenge — Test Your Own Image

Now that the complete object detection pipeline is working, the next challenge is to apply the same workflow to a different image.

The objective is to confirm that we understand the complete process rather than only executing the provided example.

The workflow remains exactly the same:

```text
New Image
    ↓
OpenCV
    ↓
YOLO
    ↓
sv.Detections
    ↓
Annotations
    ↓
Analysis
```

---

## 17.1 Challenge

Choose an image from the internet and run the complete detection pipeline.

The exercise consists of four main steps:

1. Download an image.
2. Load it with OpenCV.
3. Run YOLO and convert the predictions to `sv.Detections`.
4. Annotate and analyze the results.

An important part of the exercise is to observe:

> What objects does YOLO detect, and are any of the detections incorrect?

---

## 17.2 Download a New Image

We can download an image using:

```python
import urllib.request

urllib.request.urlretrieve(
    "IMAGE_URL",
    "mi_imagen.jpg"
)
```

Replace:

```text
IMAGE_URL
```

with the URL of the image you want to analyze.

The downloaded file will be saved as:

```text
mi_imagen.jpg
```

---

## 17.3 Example Image

If we do not have an image available, the course provides the following example:

```python
urllib.request.urlretrieve(
    "https://ultralytics.com/images/zidane.jpg",
    "assets/zidane.jpg"
)
```

This gives us another image that can be used to test the pipeline.

---

## 17.4 Load the Image

Load the new image using OpenCV:

```python
mi_imagen = cv2.imread(
    "mi_imagen.jpg"
)
```

If using the example image:

```python
mi_imagen = cv2.imread(
    "assets/zidane.jpg"
)
```

We can verify that the image loaded correctly:

```python
print(mi_imagen.shape)
```

---

## 17.5 Display the Original Image

Because OpenCV uses BGR and Matplotlib expects RGB, convert the image before displaying it:

```python
plt.figure(figsize=(12, 7))

plt.imshow(
    cv2.cvtColor(
        mi_imagen,
        cv2.COLOR_BGR2RGB
    )
)

plt.axis("off")
plt.title("Mi imagen")
plt.show()
```

---

# 18. Run YOLO on the New Image

We can reuse the YOLO model that was already loaded:

```python
results_mi_imagen = model(
    mi_imagen
)[0]
```

The model analyzes the image and produces object detections.

```text
My Image
   ↓
YOLOv8
   ↓
Raw Predictions
```

---

## 18.1 Convert the Results to Supervision

Convert the YOLO results:

```python
detections_mi_imagen = (
    sv.Detections.from_ultralytics(
        results_mi_imagen
    )
)
```

Now our predictions use the standard Supervision representation.

```text
YOLO Results
     ↓
from_ultralytics()
     ↓
sv.Detections
```

---

## 18.2 Inspect the Predictions

Check how many objects were detected:

```python
print(
    "Objetos detectados:",
    len(detections_mi_imagen)
)
```

We can also inspect:

```python
print(
    detections_mi_imagen.xyxy
)

print(
    detections_mi_imagen.confidence
)

print(
    detections_mi_imagen.class_id
)
```

This gives us information about:

```text
Detection
│
├── Position
├── Confidence
└── Class
```

---

# 19. Create Labels

Create labels containing the class name and confidence:

```python
labels_mi_imagen = [
    f"{results_mi_imagen.names[class_id]} {conf:.0%}"
    for class_id, conf in zip(
        detections_mi_imagen.class_id,
        detections_mi_imagen.confidence
    )
]
```

Example:

```text
person 96%
person 92%
tie 84%
```

---

# 20. Annotate the New Image

We can reuse the same Supervision annotators:

```python
annotated_mi_imagen = (
    box_annotator.annotate(
        scene=mi_imagen.copy(),
        detections=detections_mi_imagen
    )
)
```

Then add the labels:

```python
annotated_mi_imagen = (
    label_annotator.annotate(
        scene=annotated_mi_imagen,
        detections=detections_mi_imagen,
        labels=labels_mi_imagen
    )
)
```

Notice again:

```python
mi_imagen.copy()
```

This protects the original image from being modified.

---

# 21. Display the Result

Finally:

```python
plt.figure(figsize=(12, 7))

plt.imshow(
    cv2.cvtColor(
        annotated_mi_imagen,
        cv2.COLOR_BGR2RGB
    )
)

plt.axis("off")
plt.title(
    "YOLO + Supervision — Mi imagen"
)
plt.show()
```

We have now applied the complete pipeline to a different image.

---

# 22. Analyze the Predictions

Running the model is only part of the exercise.

We should also evaluate the results.

Ask the following questions:

### What objects were detected?

Compare the predicted labels with the objects actually visible in the image.

### Were any objects missed?

An object may exist in the image but not be detected by YOLO.

This is known as a:

```text
False Negative
```

### Were any objects detected incorrectly?

YOLO may sometimes identify an object as something that is not actually present.

This can produce a:

```text
False Positive
```

### How confident was the model?

Inspect:

```python
detections_mi_imagen.confidence
```

and compare confidence scores between different objects.

---

# 23. Experiment with Confidence

We can make the detector more strict:

```python
results_strict = model(
    mi_imagen,
    conf=0.8
)[0]
```

Then convert the results:

```python
detections_strict = (
    sv.Detections.from_ultralytics(
        results_strict
    )
)
```

Compare:

```python
print(
    "Original:",
    len(detections_mi_imagen)
)

print(
    "Confidence 0.8:",
    len(detections_strict)
)
```

This allows us to observe how the confidence threshold affects the number of predictions.

---

# 24. Experiment with a Larger YOLO Model

We can also repeat the experiment using:

```python
model_s = YOLO(
    "yolov8s.pt"
)
```

Run inference:

```python
results_s = model_s(
    mi_imagen
)[0]
```

Convert:

```python
detections_s = (
    sv.Detections.from_ultralytics(
        results_s
    )
)
```

Then compare:

```python
print(
    "YOLOv8 Nano:",
    len(detections_mi_imagen)
)

print(
    "YOLOv8 Small:",
    len(detections_s)
)
```

Now we can investigate whether the larger model:

- Detects additional objects
- Detects smaller objects
- Produces different confidence scores
- Produces fewer incorrect detections

---

# 25. Challenge Questions

After completing the experiment, answer:

1. What image did I choose?
2. How many objects did YOLO detect?
3. Which classes were detected?
4. Which detection had the highest confidence?
5. Which detection had the lowest confidence?
6. Did YOLO miss any visible objects?
7. Were there any incorrect detections?
8. What changed when the confidence threshold increased?
9. Did `yolov8s.pt` perform differently from `yolov8n.pt`?

These questions help transform the exercise from simply executing code into **analyzing model behavior**.

---

## Key Lesson

The most important concept is that the architecture does not change when we change the input image:

```text
                ANY IMAGE
                    │
                    ▼
                  YOLO
                    │
                    ▼
               Predictions
                    │
                    ▼
              sv.Detections
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
       Analyze    Filter    Inspect
          │         │         │
          └─────────┼─────────┘
                    ▼
                Annotate
                    │
                    ▼
              Final Result
```

We have moved from following a predefined example to building and evaluating our own computer vision experiment.

# 26. Saving Predictions to JSON

Visualizing detections is useful, but sometimes we also need to preserve the prediction data for later analysis.

Instead of running YOLO again every time, we can export the detection results to a structured file.

A convenient format for this is:

```text
JSON
```

JSON stands for:

> **JavaScript Object Notation**

It is widely used to store and exchange structured data.

---

## 26.1 Why Save Predictions?

Our detection pipeline produces information such as:

```text
Detection
│
├── Bounding Box
├── Confidence
├── Class ID
└── Class Name
```

If this information is saved to a file, we can analyze it later without needing to perform model inference again.

Conceptually:

```text
Image
  ↓
YOLO
  ↓
sv.Detections
  ↓
Prediction Data
  ↓
JSON File
```

This is useful for:

- Experiment documentation
- Comparing model predictions
- Analyzing detections later
- Building datasets
- Debugging
- Creating reports
- Sharing structured prediction results

---

## 26.2 Creating Prediction Records

We can convert each detection into a Python dictionary.

For example:

```python
predictions = []

for xyxy, confidence, class_id in zip(
    detections.xyxy,
    detections.confidence,
    detections.class_id
):
    prediction = {
        "class_id": int(class_id),
        "class_name": results.names[int(class_id)],
        "confidence": float(confidence),
        "bounding_box": {
            "x1": float(xyxy[0]),
            "y1": float(xyxy[1]),
            "x2": float(xyxy[2]),
            "y2": float(xyxy[3])
        }
    }

    predictions.append(prediction)
```

Each detected object is now represented by structured information.

Conceptually:

```text
Prediction
│
├── class_id
├── class_name
├── confidence
└── bounding_box
    ├── x1
    ├── y1
    ├── x2
    └── y2
```

---

## 26.3 Example JSON Structure

A saved prediction could look like:

```json
{
  "class_id": 0,
  "class_name": "person",
  "confidence": 0.94,
  "bounding_box": {
    "x1": 100.0,
    "y1": 50.0,
    "x2": 400.0,
    "y2": 300.0
  }
}
```

If multiple objects are detected, the JSON file can contain a list:

```json
[
  {
    "class_id": 0,
    "class_name": "person",
    "confidence": 0.94,
    "bounding_box": {
      "x1": 100.0,
      "y1": 50.0,
      "x2": 400.0,
      "y2": 300.0
    }
  },
  {
    "class_id": 5,
    "class_name": "bus",
    "confidence": 0.91,
    "bounding_box": {
      "x1": 450.0,
      "y1": 120.0,
      "x2": 700.0,
      "y2": 500.0
    }
  }
]
```

---

## 26.4 Save the JSON File

Python provides the built-in `json` module:

```python
import json
```

Make sure the `assets` directory exists:

```python
from pathlib import Path

Path("assets").mkdir(exist_ok=True)
```

Then save the predictions:

```python
with open(
    "assets/predictions.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        predictions,
        file,
        indent=4,
        ensure_ascii=False
    )
```

The project structure will now contain:

```text
assets/
├── bus.jpg
└── predictions.json
```

---

## 26.5 Reading the Predictions Later

The JSON file can later be loaded without running YOLO again:

```python
import json

with open(
    "assets/predictions.json",
    "r",
    encoding="utf-8"
) as file:
    saved_predictions = json.load(file)

print(saved_predictions)
```

The workflow becomes:

```text
FIRST EXECUTION

Image
  ↓
YOLO
  ↓
Predictions
  ↓
predictions.json
```

Later:

```text
predictions.json
       ↓
   json.load()
       ↓
Prediction Analysis
```

This separates **model inference** from **later data analysis**.

---

# 27. Complete Chapter Architecture

Throughout this chapter, we built the following computer vision workflow:

```text
                    INPUT IMAGE
                         │
                         ▼
                       OpenCV
                         │
                         ▼
                       YOLO
                         │
                         ▼
                Ultralytics Results
                         │
                         ▼
          sv.Detections.from_ultralytics()
                         │
                         ▼
                   sv.Detections
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
        xyxy         confidence      class_id
          │              │              │
          └──────────────┼──────────────┘
                         │
                ┌────────┴────────┐
                ▼                 ▼
             Analysis         Filtering
                │                 │
                └────────┬────────┘
                         ▼
                    Annotators
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
       Bounding Boxes              Labels
             │                       │
             └───────────┬───────────┘
                         ▼
                  Annotated Image
                         │
                  ┌──────┴──────┐
                  ▼             ▼
              Matplotlib       JSON
                  │             │
                  ▼             ▼
             Visualization   Saved Data
```

---

# 28. What We Learned

In this chapter, we learned how to:

- Understand the purpose of the Supervision library.
- Install Supervision and Ultralytics.
- Load images using OpenCV.
- Understand the difference between BGR and RGB.
- Understand bounding-box coordinates.
- Work with the `xyxy` format.
- Load pretrained YOLO models.
- Run object detection.
- Understand the COCO dataset.
- Translate class IDs into class names.
- Convert Ultralytics results into `sv.Detections`.
- Inspect bounding boxes.
- Inspect confidence scores.
- Inspect class IDs.
- Select individual detections.
- Calculate bounding-box dimensions.
- Change confidence thresholds.
- Compare YOLO model sizes.
- Draw bounding boxes.
- Add labels and confidence scores.
- Apply the pipeline to a different image.
- Analyze correct and incorrect detections.
- Preserve prediction results as structured JSON data.

---

# 29. Key Technologies

| Technology | Role in This Chapter |
|---|---|
| Python | Main programming language |
| Google Colab | Notebook development environment |
| NVIDIA T4 | GPU acceleration |
| OpenCV | Image processing |
| NumPy | Numerical array processing |
| Matplotlib | Image visualization |
| Ultralytics | YOLO model framework |
| YOLOv8 | Object detection |
| Supervision | Detection processing and annotation |
| COCO | Object categories used by the pretrained model |
| JSON | Structured prediction storage |

---

# 30. Final Takeaway

The most important concept from this chapter is not simply how to run YOLO.

It is understanding how different components work together:

```text
MODEL
  ↓
STANDARDIZED DETECTIONS
  ↓
PROCESSING
  ↓
VISUALIZATION
  ↓
ANALYSIS
```

Supervision provides the bridge between model predictions and the rest of the computer vision application.

The core pipeline can therefore be summarized as:

```text
Image → YOLO → sv.Detections → Annotated Image
```

Once the model output has been converted into `sv.Detections`, we can build reusable tools for filtering, visualization, tracking, zones, and other computer vision tasks.

---

## Chapter Status

**01 — Introduction to Supervision: COMPLETE**

