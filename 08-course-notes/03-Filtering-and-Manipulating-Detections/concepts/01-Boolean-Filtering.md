# Boolean Filtering with `sv.Detections`

## Overview

One of the most useful features of the Supervision library is the ability to filter `sv.Detections` using **Boolean masks**.

This works similarly to filtering data with NumPy.

Instead of manually looping through every detected object, we can create a condition and use it to select only the detections that satisfy that condition.

---

## What Is a Boolean Mask?

A Boolean mask is an array containing only:

```text
True
False
```

Each Boolean value corresponds to one detection.

For example, imagine that YOLO detected four objects with these confidence scores:

```text
0.92
0.71
0.43
0.85
```

If we create the condition:

```python
detections.confidence > 0.5
```

Python evaluates every detection individually.

The resulting mask would be similar to:

```python
[True, True, False, True]
```

This means:

```text
0.92 > 0.5 → True
0.71 > 0.5 → True
0.43 > 0.5 → False
0.85 > 0.5 → True
```

---

## Applying the Boolean Mask

The mask can be used directly with `sv.Detections`.

```python
mask = detections.confidence > 0.5

high_confidence = detections[mask]
```

Only the detections corresponding to `True` remain.

Conceptually:

```text
Original Detections
        │
        ▼
[0.92, 0.71, 0.43, 0.85]
        │
        │ confidence > 0.5
        ▼
[True, True, False, True]
        │
        ▼
Filtered Detections
        │
        ▼
[0.92, 0.71, 0.85]
```

---

## Filtering by Confidence

A common use of Boolean filtering is removing low-confidence detections.

```python
high_confidence = detections[
    detections.confidence > 0.5
]
```

This means:

> Keep only detections whose confidence score is greater than 50%.

We can easily change the threshold.

For example:

```python
detections.confidence > 0.3
```

is less restrictive.

While:

```python
detections.confidence > 0.8
```

is more restrictive.

---

## Filtering by Class

Boolean masks can also filter detections according to their class.

For example:

```python
persons = detections[
    detections.class_id == 0
]
```

For models trained on the COCO dataset:

```text
class_id 0 = person
```

Therefore, this condition means:

> Keep only detections classified as people.

---

## Excluding a Class

The opposite can be done using `!=`.

For example:

```python
without_buses = detections[
    detections.class_id != 5
]
```

For COCO:

```text
class_id 5 = bus
```

Therefore:

```text
class_id != 5
```

means:

> Keep everything except buses.

---

## Combining Boolean Conditions

Multiple conditions can be combined.

For example, suppose we want:

- Only people
- Confidence greater than 60%

We can write:

```python
persons_high_confidence = detections[
    (detections.class_id == 0)
    & (detections.confidence > 0.6)
]
```

Both conditions must be `True`.

Conceptually:

```text
Is Person?       Confidence > 60%?
    │                    │
    └─────────┬──────────┘
              │
              ▼
             AND
              │
              ▼
      Keep Detection
```

---

## Why Use `&` Instead of `and`?

This is an important distinction.

When working with NumPy arrays, we use:

```python
&
```

for element-wise logical AND.

Correct:

```python
(detections.class_id == 0) & (detections.confidence > 0.6)
```

Do not use:

```python
(detections.class_id == 0) and (detections.confidence > 0.6)
```

The Python keyword `and` expects individual Boolean values.

However, detection conditions produce **arrays of Boolean values**.

Because of this, NumPy-style filtering requires element-wise Boolean operators.

---

## Other Useful Boolean Operators

### AND

Both conditions must be true:

```python
condition_a & condition_b
```

### OR

At least one condition must be true:

```python
condition_a | condition_b
```

### NOT

Invert a Boolean condition:

```python
~condition
```

### Equal

```python
==
```

### Not Equal

```python
!=
```

### Greater Than

```python
>
```

### Less Than

```python
<
```

### Greater Than or Equal

```python
>=
```

### Less Than or Equal

```python
<=
```

---

## Example: Advanced Filter

Multiple conditions can be combined to create more specific filters.

For example:

```python
filtered = detections[
    (detections.class_id == 0)
    & (detections.confidence > 0.6)
    & (detections.area > 5000)
]
```

This means:

```text
Keep detection if:

Class = Person
      AND
Confidence > 60%
      AND
Area > 5000 px²
```

This is much more powerful than simply accepting every prediction produced by the model.

---

## Why Boolean Filtering Matters

Real-world computer vision applications usually need specific information.

For example, a surveillance application may only care about:

```text
Person
+
Confidence > 70%
+
Large enough to analyze
+
Inside a specific region
```

Boolean filtering allows these requirements to be expressed directly in code.

Instead of manually checking every object, the entire detection collection can be filtered efficiently.

---

## Key Takeaways

- `sv.Detections` supports NumPy-style Boolean filtering.
- A Boolean mask contains `True` and `False` values.
- Only detections corresponding to `True` values are kept.
- `detections.confidence` can be used for confidence filtering.
- `detections.class_id` can be used for class filtering.
- `==` selects matching values.
- `!=` excludes matching values.
- `&` combines conditions using AND.
- `|` combines conditions using OR.
- `~` reverses a Boolean condition.
- Complex detection rules can be created by combining multiple masks.

---

## Related Documentation

[Back to Concepts](README.md)

[Lesson 03 — Filtering and Manipulating Detections](../README.md)

[Main SAM3 Learning Journey Repository](https://github.com/Peyman-mxli/SAM3-Learning-Journey)

---

## Author

**Peyman Miyandashti**

SAM3 Computer Vision Learning Journey  
Computer Vision · Artificial Intelligence · Machine Learning

[LinkedIn](https://www.linkedin.com/in/peyman-mxli/) | [GitHub](https://github.com/Peyman-mxli)
