# YOLO Model Comparison

## Why Are There Different YOLO Models?

YOLO models are available in different sizes.

Smaller models are designed to prioritize speed and efficiency, while larger models can provide greater detection capability at the cost of additional computational resources.

In this lesson, we compare:

```text
yolov8n.pt → Nano
yolov8s.pt → Small
```

---

## YOLOv8 Nano

The course initially uses:

```python
model = YOLO("yolov8n.pt")
```

The `n` means:

```text
Nano
```

YOLOv8 Nano is designed to be small and fast.

It is useful for:

- Fast experiments
- Learning object detection
- Limited computing resources
- Real-time applications
- Rapid prototyping

---

## YOLOv8 Small

The lesson also experiments with:

```python
model_s = YOLO("yolov8s.pt")
```

The `s` means:

```text
Small
```

This model is larger than the Nano version and may detect some objects differently.

---

## Model Size Comparison

The course introduces the following general comparison:

| Model | Approximate Size | Speed | Accuracy |
|---|---:|:---:|:---:|
| Nano | ~6 MB | +++ | + |
| Small | ~22 MB | ++ | ++ |
| Medium | ~52 MB | + | +++ |

This demonstrates an important trade-off:

```text
Smaller Model
     ↓
Faster
     ↓
Less Computational Cost
```

versus:

```text
Larger Model
     ↓
More Computation
     ↓
Potentially Better Detection Capability
```

---

## Running the Nano Model

First:

```python
model = YOLO("yolov8n.pt")
```

Then:

```python
results = model(image)[0]
```

Convert the results:

```python
detections = sv.Detections.from_ultralytics(
    results
)
```

---

## Running the Small Model

Load the Small model:

```python
model_s = YOLO("yolov8s.pt")
```

Run inference:

```python
results_s = model_s(image)[0]
```

Convert the predictions:

```python
detections_s = sv.Detections.from_ultralytics(
    results_s
)
```

---

## Comparing Detection Counts

The course compares how many objects each model detects:

```python
print(
    f"yolov8n (nano): {len(detections)} objects"
)

print(
    f"yolov8s (small): {len(detections_s)} objects"
)
```

This gives us a simple first comparison between the two models.

---

## What Should We Analyze?

Detection count alone does not tell the complete story.

We should also investigate:

- Did both models detect the same objects?
- Did the confidence scores change?
- Did the Small model detect additional objects?
- Were small objects detected differently?
- Were partially occluded objects detected differently?
- Did one model produce incorrect detections?

---

## Small and Occluded Objects

The course asks us to consider whether the larger model detects more:

```text
Small Objects
      +
Partially Occluded Objects
```

An **occluded object** is an object that is partially hidden by another object or part of the scene.

For example:

```text
Visible Person
     ↓
Another Object Covers Part of Person
     ↓
Partially Occluded Person
```

These situations can make object detection more difficult.

---

## The Supervision Pipeline Does Not Change

One important concept is that changing the YOLO model does not require us to redesign the Supervision portion of the pipeline.

```text
YOLOv8 Nano ─────┐
                 │
YOLOv8 Small ────┼──→ sv.Detections
                 │
Other Models ────┘
                        ↓
                    Processing
                        ↓
                    Annotation
```

For Nano:

```python
detections = sv.Detections.from_ultralytics(
    results
)
```

For Small:

```python
detections_s = sv.Detections.from_ultralytics(
    results_s
)
```

Both produce the same type of Supervision representation.

---

## Model Selection

Choosing a model depends on the requirements of the application.

```text
Need More Speed?
      ↓
Smaller Model
```

```text
Need Greater Detection Capability?
      ↓
Consider a Larger Model
```

Other factors can include:

- Available GPU resources
- Required inference speed
- Image resolution
- Number of images
- Video processing requirements
- Detection difficulty

---

## Key Concept

The lesson demonstrates the relationship between:

```text
MODEL SIZE
    ↕
SPEED
    ↕
COMPUTATIONAL COST
    ↕
DETECTION PERFORMANCE
```

The goal is not simply to choose the largest model.

The goal is to choose a model appropriate for the computer vision task while keeping the rest of the pipeline reusable through `sv.Detections`.
