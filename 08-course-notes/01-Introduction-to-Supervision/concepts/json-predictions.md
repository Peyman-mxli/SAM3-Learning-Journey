# Saving Predictions to JSON

## What is JSON?

**JSON (JavaScript Object Notation)** is a structured format commonly used to store and exchange data.

In computer vision, JSON can be useful for saving model predictions so that they can be analyzed later without running the model again.

---

## Why Save Predictions?

After YOLO performs object detection, each prediction contains useful information such as:

```text
Detection
│
├── Class ID
├── Class Name
├── Confidence Score
└── Bounding Box
```

Instead of keeping this information only in memory, we can save it.

```text
Image
  ↓
YOLO
  ↓
sv.Detections
  ↓
Prediction Data
  ↓
JSON
```

---

## Prediction Structure

A detection can conceptually be represented as:

```json
{
  "class_id": 0,
  "class_name": "person",
  "confidence": 0.94,
  "bounding_box": {
    "x1": 100,
    "y1": 50,
    "x2": 400,
    "y2": 300
  }
}
```

This gives us both the classification information and the object's location.

---

## Multiple Predictions

An image normally contains multiple detections.

These can be represented as a JSON list:

```json
[
  {
    "class_id": 0,
    "class_name": "person",
    "confidence": 0.94
  },
  {
    "class_id": 5,
    "class_name": "bus",
    "confidence": 0.91
  }
]
```

---

## Saving Detection Data

Python provides the built-in:

```python
import json
```

A list of predictions can be saved with:

```python
with open(
    "assets/predictions.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        predictions,
        file,
        indent=4
    )
```

The result can be organized as:

```text
assets/
├── bus.jpg
└── predictions.json
```

---

## Reading Saved Predictions

Later, the prediction data can be loaded again:

```python
with open(
    "assets/predictions.json",
    "r",
    encoding="utf-8"
) as file:
    predictions = json.load(file)
```

Now we can analyze the stored results without performing YOLO inference again.

---

## Why This is Useful

Saving predictions can help with:

- Model evaluation
- Experiment comparison
- Debugging
- Reports
- Dataset analysis
- Reproducibility
- Sharing results
- Separating inference from analysis

---

## Inference vs Analysis

Without saving:

```text
Need Results
    ↓
Run YOLO Again
    ↓
Analyze
```

With saved predictions:

```text
Run YOLO Once
    ↓
Save JSON
    ↓
predictions.json
    ↓
Analyze Later
```

This can become especially useful when inference is computationally expensive.

---

## Relationship with `sv.Detections`

The information stored in JSON can originate from:

```python
detections.xyxy
detections.confidence
detections.class_id
```

and class names can be obtained from:

```python
results.names
```

Therefore:

```text
sv.Detections
      │
      ├── xyxy
      ├── confidence
      └── class_id
             │
             ▼
        Python Data
             │
             ▼
            JSON
```

---

## Key Concept

JSON allows us to preserve the **data behind the visualization**.

An annotated image shows us the result visually:

```text
Predictions → Annotated Image
```

JSON preserves the structured prediction information:

```text
Predictions → Structured Data → JSON
```

Together, they provide both visual and machine-readable outputs for a computer vision experiment.
