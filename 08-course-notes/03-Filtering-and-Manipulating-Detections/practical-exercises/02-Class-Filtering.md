   # Exercise 02 — Class Filtering

## Objective

The goal of this exercise is to learn how to filter object detections according to their **class ID**.

Object detection models such as YOLO can detect multiple types of objects in the same image.

Using Supervision, we can select only the classes that are relevant to our application.

---

## Starting Point

After running YOLO and converting the predictions to `sv.Detections`:

```python
results = model(image)[0]

detections = sv.Detections.from_ultralytics(
    results
)
```

The `detections` object contains all objects detected in the image.

Each detection includes a `class_id` that identifies the predicted object category.

---

## Step 1 — Inspect the Detected Classes

Before filtering, it is useful to see which classes were detected.

```python
print("Detected objects:")

for class_id in sorted(set(detections.class_id)):

    n = (
        detections.class_id == class_id
    ).sum()

    print(
        f"Class {class_id} "
        f"({results.names[class_id]}): "
        f"{n} detections"
    )
```

This shows:

- Class ID
- Class name
- Number of detections

---

## Understanding Class IDs

YOLO models trained on the COCO dataset use numeric class IDs.

Some examples are:

| Class ID | Object |
|---:|---|
| `0` | person |
| `1` | bicycle |
| `2` | car |
| `3` | motorcycle |
| `5` | bus |
| `7` | truck |

Instead of comparing text labels directly, we can filter using these numeric IDs.

---

## Step 2 — Select Only People

In COCO:

```text
class_id 0 = person
```

Therefore, we can select only people using:

```python
persons = detections[
    detections.class_id == 0
]
```

This creates a Boolean condition for every detection.

Conceptually:

```text
All Detections
      │
      ▼
class_id == 0?
      │
   ┌──┴──┐
  Yes    No
   │      │
 Keep   Remove
   │
   ▼
Person Detections
```

---

## Step 3 — Count the Results

We can check how many people remain:

```python
print(
    f"Only persons: {len(persons)}"
)
```

Then visualize them:

```python
mostrar(
    persons,
    "Only persons (class 0)"
)
```

---

## Complete Person Filter

```python
# COCO class 0 = person
persons = detections[
    detections.class_id == 0
]

print(
    f"Only persons: {len(persons)}"
)

mostrar(
    persons,
    "Only persons (class 0)"
)
```

---

# Combining Class and Confidence

Class filtering can be combined with confidence filtering.

Suppose we want:

```text
Person
+
Confidence > 60%
```

The code is:

```python
safe_persons = detections[
    (detections.class_id == 0)
    & (detections.confidence > 0.6)
]
```

Then:

```python
print(
    f"Persons with confidence > 60%: "
    f"{len(safe_persons)}"
)
```

And visualize:

```python
mostrar(
    safe_persons,
    "Persons with confidence > 60%"
)
```

---

## Understanding the Combined Condition

Consider these detections:

| Object | Class ID | Confidence |
|---|---:|---:|
| Person | 0 | 0.93 |
| Person | 0 | 0.55 |
| Bus | 5 | 0.95 |
| Person | 0 | 0.81 |

Our condition is:

```python
(detections.class_id == 0) & (
    detections.confidence > 0.6
)
```

The evaluation becomes:

| Object | Is Person? | Confidence > 60%? | Result |
|---|---|---|---|
| Person 93% | True | True | Keep |
| Person 55% | True | False | Remove |
| Bus 95% | False | True | Remove |
| Person 81% | True | True | Keep |

Both conditions must be true.

---

# Why We Use `&`

When combining NumPy-style Boolean conditions, we use:

```python
&
```

Correct:

```python
(
    detections.class_id == 0
) & (
    detections.confidence > 0.6
)
```

Do not use:

```python
(
    detections.class_id == 0
) and (
    detections.confidence > 0.6
)
```

The normal Python `and` operator is designed for individual Boolean values.

Here we are working with arrays containing multiple Boolean values.

Therefore, we need the element-wise operator:

```text
&
```

---

# Excluding a Class

We can also remove a specific class instead of selecting it.

In COCO:

```text
class_id 5 = bus
```

To remove buses:

```python
without_buses = detections[
    detections.class_id != 5
]
```

Then:

```python
print(
    f"With buses: {len(detections)} | "
    f"Without buses: {len(without_buses)}"
)
```

Visualize:

```python
mostrar(
    without_buses,
    "Without buses (class 5 excluded)"
)
```

---

## Understanding `!=`

The operator:

```text
!=
```

means:

```text
NOT EQUAL TO
```

Therefore:

```python
detections.class_id != 5
```

means:

> Keep detections whose class ID is not 5.

---

# Selecting Multiple Classes

We can also select more than one class.

For example:

```text
Car = 2
Bus = 5
```

We could write:

```python
vehicles = detections[
    (detections.class_id == 2)
    | (detections.class_id == 5)
]
```

The `|` operator means element-wise **OR**.

Therefore:

```text
Car
 OR
Bus
```

will be accepted.

---

# AND vs OR

### AND

Use:

```python
&
```

when **both conditions must be true**.

Example:

```python
(
    detections.class_id == 0
) & (
    detections.confidence > 0.6
)
```

Meaning:

```text
Person AND confidence > 60%
```

---

### OR

Use:

```python
|
```

when **either condition can be true**.

Example:

```python
(
    detections.class_id == 2
) | (
    detections.class_id == 5
)
```

Meaning:

```text
Car OR Bus
```

---

# Practical Example — Person Monitoring

Imagine a camera system that only needs reliable person detections.

The filtering pipeline could be:

```text
YOLO
  │
  ▼
All Objects
  │
  ▼
class_id == 0
  │
  ▼
Only People
  │
  ▼
confidence > 0.6
  │
  ▼
Reliable Person Detections
```

The code:

```python
persons = detections[
    (detections.class_id == 0)
    & (detections.confidence > 0.6)
]
```

---

# Practical Example — Traffic Monitoring

Suppose we only want:

```text
Cars
Buses
Trucks
```

These COCO classes are:

```text
2 = car
5 = bus
7 = truck
```

One possible filter is:

```python
vehicles = detections[
    (detections.class_id == 2)
    | (detections.class_id == 5)
    | (detections.class_id == 7)
]
```

Now unrelated objects are ignored.

---

# Combining More Conditions

We can make the filter more specific.

For example:

```python
filtered = detections[
    (
        (detections.class_id == 2)
        | (detections.class_id == 5)
        | (detections.class_id == 7)
    )
    & (detections.confidence > 0.7)
]
```

This means:

```text
Car OR Bus OR Truck
        AND
Confidence > 70%
```

---

# Why Class Filtering Matters

Computer vision models often detect many categories, but real applications usually need only a subset.

For example:

### Security

```text
Keep → Person
Ignore → Cars, buses, objects
```

### Traffic Monitoring

```text
Keep → Cars, buses, trucks
Ignore → People and unrelated objects
```

### Retail Analytics

```text
Keep → Relevant products or customers
Ignore → Unnecessary categories
```

Class filtering allows the same object detection model to be adapted to different applications.

---

# What I Practiced

In this exercise, I practiced:

- Accessing `detections.class_id`
- Inspecting detected classes
- Selecting a specific class
- Excluding a specific class
- Combining class and confidence filters
- Using the `&` operator
- Using the `|` operator
- Understanding COCO class IDs
- Creating application-specific detection rules

---

# Key Takeaways

- Every object detection contains a class ID.
- `detections.class_id` provides access to class information.
- `==` selects a specific class.
- `!=` excludes a specific class.
- `&` combines conditions using AND.
- `|` combines conditions using OR.
- Class filtering can be combined with confidence filtering.
- Filtering allows us to keep only objects relevant to the application.
- The same YOLO model can support different applications by changing the filtering logic.

---

## Related Exercises

[Previous: Confidence Filtering](01-Confidence-Filtering.md)

[Back to Practical Exercises](README.md)

[Next: Merge and NMS](03-Merge-and-NMS.md)

---

## Related Concepts

[Boolean Filtering](../concepts/01-Boolean-Filtering.md)

[Confidence and Class Filtering](../concepts/02-Confidence-and-Class-Filtering.md)

---

## Main Lesson

[03 — Filtering and Manipulating Detections](../README.md)

---

## Repository

[SAM3 Learning Journey](https://github.com/Peyman-mxli/SAM3-Learning-Journey)

---

## Author

**Peyman Miyandashti**

SAM3 Computer Vision Learning Journey  
Computer Vision · Artificial Intelligence · Machine Learning

[LinkedIn](https://www.linkedin.com/in/peyman-mxli/) | [GitHub](https://github.com/Peyman-mxli)
