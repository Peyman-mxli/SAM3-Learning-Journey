# Understanding `sv.Detections`

## What is `sv.Detections`?

`sv.Detections` is the central data structure used by **Supervision** to represent detected objects.

Different computer vision models can return predictions in different formats.

Supervision converts those predictions into a common structure:

```text
YOLO Results ────────┐
SAM Results ─────────┼──→ sv.Detections
Transformers Results ┘
```

Once the predictions are stored as `sv.Detections`, we can process them using the same Supervision tools.

---

## Creating Detections from YOLO

After running YOLO:

```python
results = model(image)[0]
```

we can convert the results into Supervision:

```python
detections = sv.Detections.from_ultralytics(results)
```

The workflow becomes:

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

---

## What is Inside `sv.Detections`?

Think of `sv.Detections` as a table.

Each row represents **one detected object**.

Important information includes:

```text
sv.Detections
│
├── xyxy
│   └── Bounding-box coordinates
│
├── confidence
│   └── Confidence score
│
└── class_id
    └── Object category
```

---

## Number of Detected Objects

We can determine how many objects were detected using:

```python
len(detections)
```

Example:

```python
print(
    f"Number of objects detected: {len(detections)}"
)
```

---

## Bounding Box Coordinates

Bounding-box coordinates are stored in:

```python
detections.xyxy
```

We can inspect them with:

```python
print(detections.xyxy)
```

The format is:

```text
[x1, y1, x2, y2]
```

Each row represents one object.

Example:

```text
[
    [100, 50, 400, 300],
    [450, 120, 700, 500]
]
```

---

## Confidence Scores

The confidence score represents how certain the model is about a prediction.

Supervision stores these values in:

```python
detections.confidence
```

Example:

```python
print(detections.confidence)
```

Possible result:

```text
[0.95, 0.89, 0.76]
```

Conceptually:

```text
0.95 → 95% confidence
0.89 → 89% confidence
0.76 → 76% confidence
```

---

## Class IDs

Each detected object belongs to a category.

Supervision stores its numerical category in:

```python
detections.class_id
```

Example:

```python
print(detections.class_id)
```

Possible output:

```text
[0, 0, 5]
```

With the COCO classes used by YOLO:

```text
0 → person
2 → car
5 → bus
7 → truck
```

---

## Translating Class IDs

YOLO provides the class names through:

```python
results.names
```

We can translate the detected IDs with:

```python
for class_id in sorted(set(detections.class_id)):
    print(
        f"Class {class_id}: "
        f"{results.names[class_id]}"
    )
```

This converts:

```text
class_id
   ↓
results.names
   ↓
Class Name
```

---

## Selecting One Detection

`sv.Detections` supports indexing.

To select the first detection:

```python
first = detections[0]
```

`first` is still an `sv.Detections` object, but it contains only one detected object.

We can inspect its bounding box:

```python
x1, y1, x2, y2 = first.xyxy[0]
```

Its class:

```python
results.names[first.class_id[0]]
```

And its confidence:

```python
first.confidence[0]
```

---

## Example

```python
first = detections[0]

x1, y1, x2, y2 = first.xyxy[0]

print("First detection:")
print(
    f"Class: {results.names[first.class_id[0]]}"
)
print(
    f"Confidence: {first.confidence[0]:.1%}"
)
print(
    f"Position: ({x1:.0f}, {y1:.0f}) "
    f"to ({x2:.0f}, {y2:.0f})"
)
print(
    f"Size: {x2-x1:.0f}px wide × "
    f"{y2-y1:.0f}px high"
)
```

---

## Why `sv.Detections` is Important

Without a common representation, every model could require different processing code.

Supervision gives us a standardized architecture:

```text
Different Models
      │
      ▼
sv.Detections
      │
 ┌────┼─────┐
 ▼    ▼     ▼
Boxes Classes Confidence
 │     │     │
 └─────┼─────┘
       ▼
 Processing
       ▼
 Annotation
```

This means much of our downstream code can remain the same even when the model changes.

---

## Key Concept

The three most important properties to remember are:

```python
detections.xyxy
detections.confidence
detections.class_id
```

Together they tell us:

```text
WHERE is the object?  → xyxy
HOW SURE is the model? → confidence
WHAT is the object?    → class_id
```
