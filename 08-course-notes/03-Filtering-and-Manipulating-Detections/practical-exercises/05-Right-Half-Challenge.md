# Exercise 05 — Right-Half Detection Challenge

## Objective

The goal of this challenge is to filter detections according to their **position inside the image**.

Instead of filtering by class or confidence, we want to answer:

> Is the center of this detected object located in the right half of the image?

To solve this challenge, we need to:

1. Understand bounding-box coordinates
2. Calculate the horizontal center of every bounding box
3. Find the horizontal midpoint of the image
4. Compare each detection center with the image midpoint
5. Create a Boolean mask
6. Filter the detections

---

## The Challenge

The original task is:

> Keep only objects located in the **right half** of the image.

More specifically, the horizontal center of the bounding box must satisfy:

```text
center_x > image_width / 2
```

---

# Step 1 — Understand Bounding-Box Coordinates

Supervision stores bounding boxes in:

```python
detections.xyxy
```

Each bounding box follows this format:

```text
[x1, y1, x2, y2]
```

Where:

```text
x1 = left edge
y1 = top edge
x2 = right edge
y2 = bottom edge
```

Conceptually:

```text
(x1, y1)
    ●────────────────────┐
    │                    │
    │      OBJECT        │
    │                    │
    └────────────────────●
                    (x2, y2)
```

---

# Step 2 — Calculate the Horizontal Center

To determine whether an object is on the left or right side, we calculate the horizontal center of its bounding box.

The formula is:

```text
center_x = (x1 + x2) / 2
```

For all detections:

```python
centers_x = (
    detections.xyxy[:, 0]
    + detections.xyxy[:, 2]
) / 2
```

---

## Understanding the NumPy Indexing

Consider:

```python
detections.xyxy[:, 0]
```

This means:

> Get `x1` from every detection.

And:

```python
detections.xyxy[:, 2]
```

means:

> Get `x2` from every detection.

Therefore:

```python
(
    detections.xyxy[:, 0]
    + detections.xyxy[:, 2]
) / 2
```

calculates the horizontal center of every bounding box.

---

# Step 3 — Example Center Calculation

Suppose a bounding box is:

```text
[100, 50, 300, 400]
```

Then:

```text
x1 = 100
x2 = 300
```

The center is:

```text
center_x = (100 + 300) / 2
```

Result:

```text
center_x = 200
```

---

# Step 4 — Get the Image Width

OpenCV images are NumPy arrays.

Their shape is:

```python
image.shape
```

A possible result:

```text
(1080, 1920, 3)
```

This represents:

```text
Height   = 1080
Width    = 1920
Channels = 3
```

Therefore, the image width is:

```python
image.shape[1]
```

---

# Step 5 — Calculate the Image Midpoint

The horizontal midpoint is:

```python
image_midpoint = image.shape[1] / 2
```

For example, if:

```text
image width = 1920
```

then:

```text
image midpoint = 960
```

Conceptually:

```text
0                  960                 1920
│                   │                    │
│     LEFT HALF     │     RIGHT HALF     │
│                   │                    │
└───────────────────┼────────────────────┘
                 midpoint
```

---

# Step 6 — Create the Boolean Mask

Now compare every detection center with the image midpoint:

```python
mask = centers_x > image_midpoint
```

This produces a Boolean array.

For example:

```text
centers_x:

[250, 430, 1100, 1450]
```

If the midpoint is:

```text
960
```

the mask becomes:

```text
[False, False, True, True]
```

Because:

```text
250  > 960 → False
430  > 960 → False
1100 > 960 → True
1450 > 960 → True
```

---

# Step 7 — Filter the Detections

Apply the Boolean mask:

```python
right_detections = detections[
    mask
]
```

Only detections whose centers are on the right side remain.

---

# Complete Challenge Solution

```python
# Calculate horizontal center of every bounding box

centers_x = (
    detections.xyxy[:, 0]
    + detections.xyxy[:, 2]
) / 2


# Calculate horizontal midpoint of the image

image_midpoint = image.shape[1] / 2


# Create Boolean mask

mask = centers_x > image_midpoint


# Keep detections located on the right side

right_detections = detections[
    mask
]


# Display result

print(
    f"Objects in right half: "
    f"{len(right_detections)}"
)

mostrar(
    right_detections,
    "Objects in the right half"
)
```

---

# Challenge Logic

The complete logic can be represented as:

```text
Bounding Boxes
      │
      ▼
Extract x1 and x2
      │
      ▼
Calculate center_x
      │
      ▼
Get image width
      │
      ▼
Calculate image midpoint
      │
      ▼
center_x > midpoint?
      │
   ┌──┴──┐
  Yes    No
   │      │
 Keep   Remove
   │
   ▼
Right-Side Detections
```

---

# Extension 1 — Left-Half Detections

Once we know how to select the right side, selecting the left side is simple.

Instead of:

```python
centers_x > image_midpoint
```

use:

```python
centers_x < image_midpoint
```

Complete example:

```python
left_detections = detections[
    centers_x < image_midpoint
]
```

---

# Extension 2 — Calculate Vertical Centers

We can apply the same idea vertically.

The vertical center is:

```text
center_y = (y1 + y2) / 2
```

For all detections:

```python
centers_y = (
    detections.xyxy[:, 1]
    + detections.xyxy[:, 3]
) / 2
```

---

# Extension 3 — Top-Half Detections

The image height is:

```python
image.shape[0]
```

Calculate the vertical midpoint:

```python
vertical_midpoint = image.shape[0] / 2
```

Then select objects in the top half:

```python
top_detections = detections[
    centers_y < vertical_midpoint
]
```

---

# Extension 4 — Bottom-Half Detections

For the bottom half:

```python
bottom_detections = detections[
    centers_y > vertical_midpoint
]
```

The same spatial-filtering concept now works in both dimensions.

---

# Extension 5 — Top-Right Region

We can combine horizontal and vertical conditions.

For example:

```python
top_right = detections[
    (centers_x > image_midpoint)
    & (centers_y < vertical_midpoint)
]
```

This keeps objects located in the:

```text
TOP-RIGHT
```

section of the image.

---

# Dividing the Image into Four Regions

Using horizontal and vertical centers, the image can be divided into four regions:

```text
┌──────────────────┬──────────────────┐
│                  │                  │
│     TOP LEFT     │    TOP RIGHT     │
│                  │                  │
├──────────────────┼──────────────────┤
│                  │                  │
│   BOTTOM LEFT    │   BOTTOM RIGHT   │
│                  │                  │
└──────────────────┴──────────────────┘
```

The conditions become:

### Top Left

```python
top_left = detections[
    (centers_x < image_midpoint)
    & (centers_y < vertical_midpoint)
]
```

### Top Right

```python
top_right = detections[
    (centers_x > image_midpoint)
    & (centers_y < vertical_midpoint)
]
```

### Bottom Left

```python
bottom_left = detections[
    (centers_x < image_midpoint)
    & (centers_y > vertical_midpoint)
]
```

### Bottom Right

```python
bottom_right = detections[
    (centers_x > image_midpoint)
    & (centers_y > vertical_midpoint)
]
```

---

# Combining Position with Confidence

Spatial filtering can also be combined with confidence filtering.

For example:

```python
filtered = detections[
    (centers_x > image_midpoint)
    & (detections.confidence > 0.6)
]
```

This means:

```text
Right side
    AND
Confidence > 60%
```

---

# Combining Position, Class, Confidence, and Size

We can combine everything learned in this lesson.

```python
filtered = detections[
    (detections.class_id == 0)
    & (detections.confidence > 0.6)
    & (detections.area > 5000)
    & (centers_x > image_midpoint)
]
```

This means:

```text
Person
  AND
Confidence > 60%
  AND
Area > 5000 px²
  AND
Located on the right side
```

This demonstrates the power of Boolean filtering with `sv.Detections`.

---

# Real-World Example — Traffic Monitoring

Imagine a road camera:

```text
LEFT LANE               RIGHT LANE
    │                        │
    ▼                        ▼
┌─────────────────┬─────────────────┐
│                 │                 │
│     Vehicles    │     Vehicles    │
│                 │                 │
└─────────────────┴─────────────────┘
```

Bounding-box centers could help determine which lane contains each vehicle.

This information could later support:

- Vehicle counting
- Lane occupancy
- Traffic analysis
- Direction estimation
- Congestion monitoring

---

# Real-World Example — Restricted Area

A security camera could define part of an image as a restricted region.

```text
Camera Image

┌──────────────────────────────┐
│                              │
│                              │
│             ┌──────────────┐ │
│             │  RESTRICTED  │ │
│             │     AREA     │ │
│             └──────────────┘ │
│                              │
└──────────────────────────────┘
```

Instead of responding to every detected person, the system could analyze only detections whose position enters the restricted region.

This is the foundation of more advanced **Region of Interest (ROI)** processing.

---

# What I Practiced

In this challenge, I practiced:

- Accessing `detections.xyxy`
- Understanding `[x1, y1, x2, y2]`
- Calculating bounding-box centers
- Reading image width with `image.shape[1]`
- Reading image height with `image.shape[0]`
- Creating spatial Boolean masks
- Filtering left and right regions
- Filtering top and bottom regions
- Combining multiple spatial conditions
- Combining position with class, confidence, and size filters

---

# Key Takeaways

- Bounding-box coordinates can be used for spatial filtering.
- `detections.xyxy` contains `[x1, y1, x2, y2]`.
- Horizontal center is calculated with `(x1 + x2) / 2`.
- Vertical center is calculated with `(y1 + y2) / 2`.
- `image.shape[1]` provides image width.
- `image.shape[0]` provides image height.
- Comparing centers with image midpoints allows left/right and top/bottom filtering.
- Boolean operators allow multiple spatial conditions to be combined.
- Spatial filtering can be combined with confidence, class, and size filtering.
- These techniques provide a foundation for Regions of Interest and more advanced computer vision applications.

---

## Challenge Completed

The original challenge was:

> Keep only objects whose horizontal center is located in the right half of the image.

Final solution:

```python
centers_x = (
    detections.xyxy[:, 0]
    + detections.xyxy[:, 2]
) / 2

image_midpoint = image.shape[1] / 2

mask = centers_x > image_midpoint

right_detections = detections[mask]
```

**Challenge status: Completed**

---

## Related Exercises

[Previous: Top Confidence Detections](04-Top-Confidence-Detections.md)

[Back to Practical Exercises](README.md)

---

## Related Concepts

[Size and Position Filtering](../concepts/04-Size-and-Position-Filtering.md)

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
