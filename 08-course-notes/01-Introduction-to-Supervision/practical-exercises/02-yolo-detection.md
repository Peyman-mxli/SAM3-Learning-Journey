# Exercise 02 — YOLO Object Detection

## Objective

The objective of this exercise is to use a pretrained **YOLOv8** model to detect objects in the image loaded during the previous exercise.

We will learn how to:

- Import YOLO from Ultralytics.
- Load a pretrained YOLO model.
- Run inference on an image.
- Understand the basic model output.
- Identify the role of YOLO in our computer vision pipeline.

---

## 1. Import YOLO

YOLO is provided through the Ultralytics library:

```python
from ultralytics import YOLO
```

This gives us access to pretrained YOLO models.

---

## 2. Load the YOLO Model

The course uses the YOLOv8 Nano model:

```python
model = YOLO("yolov8n.pt")
```

The filename tells us which model is being loaded:

```text
yolov8n.pt
│      │
│      └── n = Nano
│
└── YOLOv8
```

The Nano model is small and fast, making it useful for learning and experimentation.

---

## 3. Model Weights

The file:

```text
yolov8n.pt
```

contains the pretrained model weights.

When the model is used for the first time, Ultralytics can automatically download the required weights.

Conceptually:

```text
YOLO Architecture
       +
Pretrained Weights
       ↓
Object Detection Model
```

---

## 4. Run Object Detection

We already have our image from the previous exercise:

```python
image = cv2.imread("assets/bus.jpg")
```

Now pass it to YOLO:

```python
results = model(image)[0]
```

This operation is called:

```text
Inference
```

Inference means using a trained model to make predictions on new data.

---

## 5. Understanding `[0]`

When we execute:

```python
model(image)
```

YOLO returns a collection of results.

Because we are processing one image, we select its result using:

```python
[0]
```

Therefore:

```python
results = model(image)[0]
```

can be understood as:

```text
Image
  ↓
YOLO
  ↓
List of Results
  ↓
[0]
  ↓
Result for the First Image
```

---

## 6. What Does YOLO Predict?

For each detected object, YOLO can provide information such as:

```text
Object Detection
│
├── Bounding Box
├── Class
└── Confidence
```

For example:

```text
Person
├── Bounding Box: [x1, y1, x2, y2]
└── Confidence: 95%
```

The model therefore answers three important questions:

```text
WHAT?
→ Object class

WHERE?
→ Bounding box

HOW SURE?
→ Confidence score
```

---

## 7. COCO Classes

The pretrained YOLO model used in this exercise recognizes object categories from the **COCO dataset**.

Some relevant examples are:

```text
0 → person
2 → car
5 → bus
7 → truck
```

YOLO stores the class names in:

```python
results.names
```

For example:

```python
print(results.names[0])
```

returns:

```text
person
```

---

## 8. Inspect the YOLO Result

We can inspect the result:

```python
print(results)
```

The raw YOLO output contains the information generated during inference.

However, rather than manually processing the Ultralytics-specific result format, the next exercise will convert it into Supervision's standardized representation.

---

## 9. YOLO's Role in the Pipeline

At this point, our pipeline is:

```text
Image URL
    ↓
Download Image
    ↓
OpenCV
    ↓
Image Array
    ↓
YOLOv8 Nano
    ↓
Inference
    ↓
Ultralytics Results
```

YOLO is responsible for the **detection stage**.

It does not represent the entire computer vision application.

---

## 10. Complete Exercise

```python
from ultralytics import YOLO
import cv2

# Load image
image = cv2.imread("assets/bus.jpg")

# Load pretrained YOLOv8 Nano model
model = YOLO("yolov8n.pt")

# Run inference
results = model(image)[0]

# Inspect result
print(results)
```

---

## 11. Connection to Supervision

The output currently belongs to Ultralytics:

```text
Image
  ↓
YOLO
  ↓
Ultralytics Result
```

In the next exercise, we will perform the critical conversion:

```python
detections = sv.Detections.from_ultralytics(
    results
)
```

The architecture will then become:

```text
Image
  ↓
YOLO
  ↓
Ultralytics Result
  ↓
Supervision
  ↓
sv.Detections
```

---

## Key Takeaways

```text
YOLO
→ Performs object detection

YOLO("yolov8n.pt")
→ Loads the pretrained Nano model

model(image)
→ Runs inference

[0]
→ Selects the result for the first image

results.names
→ Maps class IDs to object names
```

The next practical exercise will transform these YOLO predictions into `sv.Detections`, allowing us to inspect and process them using Supervision.
