# Exercise 04 — Top Confidence Detections

## Objective

The goal of this exercise is to learn how to **rank detections by confidence** and select only the most confident predictions.

Instead of using a fixed confidence threshold, we can ask:

> Which are the N most confident detections?

For example:

- Top 1 detection
- Top 3 detections
- Top 5 detections

This technique is useful when an application needs a limited number of the strongest predictions.

---

## Starting Point

After running YOLO and converting the results to Supervision:

```python
results = model(image)[0]

detections = sv.Detections.from_ultralytics(
    results
)
```

Each detection contains a confidence score accessible through:

```python
detections.confidence
```

---

# Step 1 — Inspect Confidence Scores

We can inspect all confidence values:

```python
print(detections.confidence)
```

A possible result could look like:

```text
[0.91, 0.87, 0.63, 0.95, 0.72]
```

These values are not necessarily stored from highest to lowest.

To find the strongest predictions, we need to sort them.

---

# Step 2 — Understanding `np.argsort()`

NumPy provides:

```python
np.argsort()
```

Instead of returning the sorted values themselves, `argsort()` returns the **indices that would sort the array**.

Example:

```python
confidence = np.array([
    0.91,
    0.63,
    0.95,
    0.72
])

indices = np.argsort(confidence)

print(indices)
```

Conceptually:

```text
Original:

Index       0      1      2      3
Confidence 0.91   0.63   0.95   0.72
```

Sorted from lowest to highest:

```text
0.63
0.72
0.91
0.95
```

Therefore, `argsort()` returns the corresponding indices.

---

# Step 3 — Sort Detection Confidence

We can sort the detections using:

```python
indices = np.argsort(
    detections.confidence
)
```

By default, NumPy sorts from:

```text
Lowest → Highest
```

But we want:

```text
Highest → Lowest
```

---

# Step 4 — Reverse the Order

Python slicing can reverse an array using:

```python
[::-1]
```

Therefore:

```python
indices = np.argsort(
    detections.confidence
)[::-1]
```

Now the indices are ordered from the highest-confidence detection to the lowest-confidence detection.

Conceptually:

```text
np.argsort()
      │
      ▼
Lowest → Highest
      │
      ▼
[::-1]
      │
      ▼
Highest → Lowest
```

---

# Step 5 — Select the Top 3

We only want the first three indices.

Use:

```python
[:3]
```

The complete operation becomes:

```python
indices_top3 = np.argsort(
    detections.confidence
)[::-1][:3]
```

This means:

```text
np.argsort(...)
      │
      ▼
Sort indices
      │
      ▼
[::-1]
      │
      ▼
Reverse order
      │
      ▼
[:3]
      │
      ▼
Keep first 3
```

---

# Step 6 — Select the Detections

Now use the indices to select detections:

```python
top3 = detections[
    indices_top3
]
```

The `top3` variable contains only the three most confident predictions.

---

# Step 7 — Display the Results

The lesson prints the Top 3 detections:

```python
print(
    "Top 3 detections by confidence:"
)

for i in range(len(top3)):

    print(
        f"{results.names[top3.class_id[i]]}: "
        f"{top3.confidence[i]:.1%}"
    )
```

The `.1%` formatting converts confidence values into percentages.

For example:

```text
0.934
```

becomes:

```text
93.4%
```

---

# Step 8 — Visualize the Top 3

The selected detections can be visualized using:

```python
mostrar(
    top3,
    "Top 3 most confident detections"
)
```

Only the three strongest predictions are displayed.

---

# Complete Exercise

```python
# Sort confidence scores from highest to lowest
# and keep the first three indices

indices_top3 = np.argsort(
    detections.confidence
)[::-1][:3]


# Select those detections

top3 = detections[
    indices_top3
]


# Print results

print(
    "Top 3 detections by confidence:"
)

for i in range(len(top3)):

    print(
        f"{results.names[top3.class_id[i]]}: "
        f"{top3.confidence[i]:.1%}"
    )


# Visualize

mostrar(
    top3,
    "Top 3 most confident detections"
)
```

---

# Understanding the Complete Expression

The most important line is:

```python
np.argsort(
    detections.confidence
)[::-1][:3]
```

It can be broken into three operations.

### Operation 1

```python
np.argsort(
    detections.confidence
)
```

Sort the indices according to confidence.

```text
Lowest → Highest
```

---

### Operation 2

```python
[::-1]
```

Reverse the order.

```text
Highest → Lowest
```

---

### Operation 3

```python
[:3]
```

Keep only the first three.

```text
Top 3
```

---

# Selecting Top N Detections

The same technique works for any number of detections.

## Top 1

```python
indices = np.argsort(
    detections.confidence
)[::-1][:1]
```

---

## Top 3

```python
indices = np.argsort(
    detections.confidence
)[::-1][:3]
```

---

## Top 5

```python
indices = np.argsort(
    detections.confidence
)[::-1][:5]
```

---

## Top 10

```python
indices = np.argsort(
    detections.confidence
)[::-1][:10]
```

We can generalize this using a variable:

```python
n = 3

indices_top = np.argsort(
    detections.confidence
)[::-1][:n]

top_detections = detections[
    indices_top
]
```

Now changing:

```python
n
```

changes how many detections are selected.

---

# Top-N vs Confidence Threshold

These are two different filtering strategies.

## Confidence Threshold

```python
detections[
    detections.confidence > 0.5
]
```

means:

> Keep every detection above 50% confidence.

The number of resulting detections is not fixed.

---

## Top-N

```python
np.argsort(
    detections.confidence
)[::-1][:3]
```

means:

> Keep exactly the three strongest detections, if at least three detections exist.

---

## Comparison

| Method | Selection Rule |
|---|---|
| Confidence threshold | Keep everything above a minimum confidence |
| Top-N | Keep the N highest-confidence detections |

Both approaches are useful depending on the application.

---

# Example

Suppose the detections have these confidence scores:

```text
Person → 96%
Bus    → 91%
Person → 87%
Car    → 72%
Person → 61%
```

Using:

```python
[:3]
```

the result is:

```text
Person → 96%
Bus    → 91%
Person → 87%
```

The remaining detections are ignored.

---

# Are the Largest Objects Always the Most Confident?

No.

Object size and model confidence are different properties.

A large object can have low confidence because it may be:

- Partially hidden
- Blurry
- Unusual
- Difficult to classify

A smaller object can sometimes receive a higher confidence score if its visual features are easier for the model to recognize.

Therefore:

```text
Large Object ≠ Automatically High Confidence
```

and:

```text
Small Object ≠ Automatically Low Confidence
```

---

# Practical Applications

Selecting Top-N detections can be useful when:

- Only the strongest prediction matters
- An application has limited processing capacity
- Only a fixed number of objects should be analyzed
- The highest-confidence objects need priority
- Results need to be ranked
- A downstream model should receive only the strongest detections

---

# Example Pipeline

A computer vision system could use:

```text
Input Image
     │
     ▼
YOLO
     │
     ▼
All Detections
     │
     ▼
Sort by Confidence
     │
     ▼
Highest → Lowest
     │
     ▼
Select Top N
     │
     ▼
Final Detections
```

---

# Combining Top-N with Other Filters

Top-N selection can also happen after other filters.

For example:

```python
persons = detections[
    detections.class_id == 0
]
```

Then select the Top 3 most confident people:

```python
indices = np.argsort(
    persons.confidence
)[::-1][:3]

top3_persons = persons[
    indices
]
```

The pipeline becomes:

```text
All Detections
      │
      ▼
Person Filter
      │
      ▼
Only People
      │
      ▼
Sort by Confidence
      │
      ▼
Select Top 3
      │
      ▼
Top 3 Person Detections
```

---

# What I Practiced

In this exercise, I practiced:

- Accessing detection confidence scores
- Using `np.argsort()`
- Understanding index-based sorting
- Reversing arrays with `[::-1]`
- Selecting elements with `[:N]`
- Ranking detections by confidence
- Selecting Top-N predictions
- Comparing Top-N selection with threshold filtering

---

# Key Takeaways

- `detections.confidence` contains confidence scores.
- `np.argsort()` returns indices that sort an array.
- `argsort()` sorts from lowest to highest by default.
- `[::-1]` reverses the sorting order.
- `[:3]` selects the first three results.
- Combining these operations gives the Top 3 detections.
- The same approach can select any Top-N value.
- Top-N selection and confidence thresholds solve different problems.
- Object size does not necessarily determine confidence.
- Ranking detections is useful when only the strongest predictions should continue through the pipeline.

---

## Related Exercises

[Previous: Merge and NMS](03-Merge-and-NMS.md)

[Back to Practical Exercises](README.md)

[Next: Right-Half Detection Challenge](05-Right-Half-Challenge.md)

---

## Related Concepts

[Confidence and Class Filtering](../concepts/02-Confidence-and-Class-Filtering.md)

[Size and Position Filtering](../concepts/04-Size-and-Position-Filtering.md)

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
