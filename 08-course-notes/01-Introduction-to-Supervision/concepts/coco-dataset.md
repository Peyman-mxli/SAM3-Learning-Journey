# COCO Dataset

## What is COCO?

**COCO** stands for:

> **Common Objects in Context**

COCO is a large computer vision dataset containing real-world images with objects labeled into different categories.

The pretrained YOLOv8 models used in this course can detect objects from **80 COCO categories**.

---

## Why COCO Matters

When we use a pretrained YOLO model such as:

```python
model = YOLO("yolov8n.pt")
```

the model already knows how to recognize the object categories it learned during training.

This means we can immediately perform object detection without training our own model first.

```text
COCO Dataset
      ↓
YOLO Training
      ↓
Pretrained YOLO Model
      ↓
New Image
      ↓
Object Detection
```

---

## Examples of COCO Objects

COCO contains common objects such as:

- People
- Cars
- Buses
- Trucks
- Animals
- Furniture
- Everyday objects

Some classes frequently used in our course examples are:

| Class ID | Object |
|---:|---|
| `0` | person |
| `2` | car |
| `5` | bus |
| `7` | truck |

---

## What is a Class ID?

Computer vision models represent object categories internally using numbers.

For example:

```text
0 → person
2 → car
5 → bus
7 → truck
```

These numbers are called:

```text
class_id
```

Supervision stores them in:

```python
detections.class_id
```

---

## Translating Class IDs to Names

YOLO provides a dictionary containing the names of its classes:

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

And:

```python
results.names[5]
```

returns:

```text
bus
```

---

## Inspecting Detected Classes

We can inspect the classes detected in an image:

```python
for class_id in sorted(
    set(detections.class_id)
):
    print(
        f"Class {class_id}: "
        f"{results.names[class_id]}"
    )
```

Conceptually:

```text
YOLO Prediction
      ↓
class_id
      ↓
results.names
      ↓
Object Name
```

---

## COCO and Our Bus Example

When we run YOLO on the example image:

```python
results = model(image)[0]
```

YOLO searches for objects belonging to the categories it already knows.

The predictions are then converted to Supervision:

```python
detections = sv.Detections.from_ultralytics(
    results
)
```

We can then access:

```python
detections.class_id
```

and translate those IDs using:

```python
results.names
```

---

## Pretrained Models

A major advantage of pretrained models is that we do not need to start from zero.

Instead:

```text
Large Labeled Dataset
        ↓
Model Training
        ↓
Pretrained Model
        ↓
Our Application
```

For this lesson, the pretrained YOLO model allows us to focus on learning the computer vision pipeline and Supervision rather than training a detection model ourselves.

---

## Key Concept

The relationship between COCO, YOLO, and Supervision can be summarized as:

```text
COCO
  ↓
Defines the Object Categories
  ↓
Pretrained YOLO
  ↓
Detects Those Categories
  ↓
class_id
  ↓
sv.Detections
  ↓
results.names
  ↓
Human-Readable Object Name
```

COCO provides the object categories learned by the pretrained YOLO model, while Supervision helps us process and visualize the resulting detections.
