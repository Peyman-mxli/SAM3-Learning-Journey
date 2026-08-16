# Size and Position Filtering

## Overview

Object detections can be filtered not only by **confidence** and **class**, but also according to their physical characteristics inside an image.

Two especially useful characteristics are:

- **Bounding-box size**
- **Bounding-box position**

These techniques allow us to answer questions such as:

- Is the detected object large enough to be relevant?
- Is the object located on the left or right side of the image?
- Is the object inside a specific region?
- Is the object close to the center of the image?
- Should very small detections be ignored?

This type of filtering is especially important when building real-world computer vision systems.

---

# 1. Bounding-Box Coordinates

Object detections are commonly represented using four coordinates:

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
    ●─────────────────┐
    │                 │
    │     OBJECT      │
    │                 │
    └─────────────────●
                  (x2, y2)
```

Supervision stores these coordinates in:

```python
detections.xyxy
```

---

# 2. Accessing Bounding Boxes

We can inspect all bounding boxes using:

```python
print(detections.xyxy)
```

A possible result could look like:

```text
[
    [100, 50, 250, 400],
    [300, 80, 500, 420],
    [600, 90, 750, 350]
]
```

Each row represents one detected object.

---

# 3. Bounding-Box Width

The width of a bounding box can be calculated as:

```text
width = x2 - x1
```

Using NumPy-style indexing:

```python
widths = (
    detections.xyxy[:, 2]
    - detections.xyxy[:, 0]
)
```

This calculates the width of every detected bounding box.

---

# 4. Bounding-Box Height

The height is:

```text
height = y2 - y1
```

For every detection:

```python
heights = (
    detections.xyxy[:, 3]
    - detections.xyxy[:, 1]
)
```

---

# 5. Bounding-Box Area

The area of a bounding box can be calculated using:

```text
Area = Width × Height
```

Conceptually:

```text
(x1, y1)
    ┌────────────────────┐
    │                    │
    │       AREA         │ Height
    │                    │
    └────────────────────┘
            Width
```

Supervision makes this easier by providing:

```python
detections.area
```

---

# 6. Inspecting Detection Areas

In the lesson:

```python
areas = detections.area
```

We can inspect useful statistics:

```python
print(
    f"Minimum area: {areas.min():.0f} px²"
)

print(
    f"Maximum area: {areas.max():.0f} px²"
)

print(
    f"Average area: {areas.mean():.0f} px²"
)
```

These values help us understand the relative sizes of detected objects.

---

# 7. Filtering Large Objects

Suppose we only want objects larger than:

```text
5000 px²
```

We can create the condition:

```python
detections.area > 5000
```

Then filter:

```python
large_objects = detections[
    detections.area > 5000
]
```

Only detections satisfying the area requirement remain.

---

# 8. Filtering Small Objects

The same logic can be reversed.

For example:

```python
small_objects = detections[
    detections.area < 5000
]
```

This keeps only detections smaller than `5000 px²`.

---

# 9. Why Filter by Size?

Small detections may sometimes represent:

- Very distant objects
- Detection noise
- False positives
- Objects too small for further analysis
- Objects outside the application's area of interest

For example, an application might require:

```text
Person detected
      +
Confidence > 60%
      +
Area > 5000 px²
```

This can be expressed as:

```python
filtered = detections[
    (detections.class_id == 0)
    & (detections.confidence > 0.6)
    & (detections.area > 5000)
]
```

---

# 10. Understanding Image Dimensions

OpenCV images are stored as NumPy arrays.

We can inspect their dimensions using:

```python
image.shape
```

A typical result may look like:

```text
(1080, 1920, 3)
```

This means:

```text
1080 = image height
1920 = image width
3    = color channels
```

Therefore:

```python
image.shape[0]
```

returns the image height.

And:

```python
image.shape[1]
```

returns the image width.

---

# 11. Finding the Horizontal Midpoint

To divide an image into left and right halves, we need its horizontal midpoint.

```python
image_midpoint = image.shape[1] / 2
```

For an image with width:

```text
1920 pixels
```

the midpoint is:

```text
960 pixels
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

# 12. Calculating the Horizontal Center of a Detection

Each bounding box contains:

```text
x1
```

and:

```text
x2
```

The horizontal center is:

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

# 13. Example of Center Calculation

Suppose a bounding box is:

```text
[100, 50, 300, 400]
```

Then:

```text
x1 = 100
x2 = 300
```

The horizontal center is:

```text
(100 + 300) / 2
```

which gives:

```text
200
```

Therefore:

```text
center_x = 200
```

---

# 14. Filtering the Right Half of the Image

The extension challenge from the lesson asks us to keep only detections located in the **right half of the image**.

First calculate the bounding-box centers:

```python
centers_x = (
    detections.xyxy[:, 0]
    + detections.xyxy[:, 2]
) / 2
```

Then calculate the image midpoint:

```python
image_midpoint = image.shape[1] / 2
```

Create the mask:

```python
mask = centers_x > image_midpoint
```

Finally:

```python
right_detections = detections[mask]
```

---

# 15. Complete Right-Half Solution

```python
centers_x = (
    detections.xyxy[:, 0]
    + detections.xyxy[:, 2]
) / 2

image_midpoint = image.shape[1] / 2

mask = centers_x > image_midpoint

right_detections = detections[mask]
```

The logic is:

```text
Bounding Box
     │
     ▼
Calculate center_x
     │
     ▼
Compare with image midpoint
     │
     ▼
center_x > midpoint?
     │
   ┌─┴─┐
  Yes  No
   │    │
 Keep  Remove
```

---

# 16. Filtering the Left Half

Once the right-side logic is understood, selecting the left side is simple.

```python
left_detections = detections[
    centers_x < image_midpoint
]
```

Now we have:

```text
center_x < midpoint
        ↓
Left side
```

and:

```text
center_x > midpoint
        ↓
Right side
```

---

# 17. Calculating the Vertical Center

The same idea works vertically.

The vertical center of a bounding box is:

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

# 18. Finding the Vertical Image Midpoint

The image height is:

```python
image.shape[0]
```

Therefore:

```python
vertical_midpoint = image.shape[0] / 2
```

This divides the image into:

```text
┌─────────────────────────┐
│                         │
│        TOP HALF         │
│                         │
├─────────────────────────┤
│                         │
│       BOTTOM HALF       │
│                         │
└─────────────────────────┘
```

---

# 19. Filtering Top and Bottom

Objects in the top half:

```python
top_detections = detections[
    centers_y < vertical_midpoint
]
```

Objects in the bottom half:

```python
bottom_detections = detections[
    centers_y > vertical_midpoint
]
```

This demonstrates how bounding-box coordinates can define spatial rules.

---

# 20. Combining Position with Other Filters

Spatial filtering becomes much more useful when combined with other detection properties.

For example:

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
Keep only detections that are:

Person
  AND
Confidence > 60%
  AND
Area > 5000 px²
  AND
Located on the right side
```

---

# 21. Regions of Interest

The same concepts can be extended beyond simply dividing an image into halves.

We can define a **Region of Interest (ROI)**.

For example:

```text
┌────────────────────────────────┐
│                                │
│        Ignore this area        │
│                                │
│      ┌──────────────────┐      │
│      │                  │      │
│      │ REGION OF        │      │
│      │ INTEREST         │      │
│      │                  │      │
│      └──────────────────┘      │
│                                │
└────────────────────────────────┘
```

Only detections whose centers fall inside the region could be processed.

This idea is extremely useful in real computer vision applications.

---

# 22. Practical Application: Traffic Monitoring

Imagine a camera monitoring a road.

The image could be divided into:

```text
LEFT LANE       RIGHT LANE
    │               │
    ▼               ▼
┌───────────┬───────────┐
│           │           │
│ Vehicles  │ Vehicles  │
│           │           │
└───────────┴───────────┘
```

Using bounding-box centers, detections could be assigned to different lanes.

This could later support:

- Vehicle counting
- Traffic density analysis
- Lane occupancy
- Direction monitoring

---

# 23. Practical Application: Security Camera

A security system may only care about objects entering a specific area.

For example:

```text
Camera Image
┌──────────────────────────────┐
│                              │
│                              │
│              ┌─────────────┐ │
│              │ RESTRICTED  │ │
│              │    AREA     │ │
│              └─────────────┘ │
│                              │
└──────────────────────────────┘
```

Instead of processing every detected person, the application can focus only on detections inside the restricted region.

---

# 24. Size and Position Together

Size and position filtering answer two different questions.

### Size Filtering

```text
How large is the detected object?
```

Using:

```python
detections.area
```

### Position Filtering

```text
Where is the detected object?
```

Using:

```python
detections.xyxy
```

Together, these provide powerful control over detection results.

---

# Detection Filtering Pipeline

A complete pipeline could look like:

```text
Input Image
     │
     ▼
Object Detector
     │
     ▼
Raw Detections
     │
     ▼
Class Filter
     │
     ▼
Confidence Filter
     │
     ▼
Size Filter
     │
     ▼
NMS
     │
     ▼
Position Filter
     │
     ▼
Relevant Detections
```

---

# Important Properties

| Property | Purpose |
|---|---|
| `detections.xyxy` | Bounding-box coordinates |
| `detections.area` | Bounding-box area |
| `image.shape[0]` | Image height |
| `image.shape[1]` | Image width |
| `xyxy[:, 0]` | `x1` coordinate |
| `xyxy[:, 1]` | `y1` coordinate |
| `xyxy[:, 2]` | `x2` coordinate |
| `xyxy[:, 3]` | `y2` coordinate |

---

# Key Takeaways

- Bounding boxes use the format `[x1, y1, x2, y2]`.
- `detections.xyxy` provides access to bounding-box coordinates.
- `detections.area` provides the area of every detection.
- Area filtering can remove very small or irrelevant objects.
- `image.shape[1]` provides the image width.
- `image.shape[0]` provides the image height.
- Bounding-box centers can be calculated from their coordinates.
- Horizontal centers allow left/right filtering.
- Vertical centers allow top/bottom filtering.
- Size, confidence, class, and position conditions can be combined.
- Spatial filtering provides the foundation for Regions of Interest.
- These techniques are useful in traffic monitoring, surveillance, counting, and many other computer vision applications.

---

## Related Documentation

[Boolean Filtering](01-Boolean-Filtering.md)

[Confidence and Class Filtering](02-Confidence-and-Class-Filtering.md)

[Non-Maximum Suppression](03-Non-Maximum-Suppression.md)

[Back to Concepts](README.md)

[Lesson 03 — Filtering and Manipulating Detections](../README.md)

[Main SAM3 Learning Journey Repository](https://github.com/Peyman-mxli/SAM3-Learning-Journey)

---

## Author

**Peyman Miyandashti**

SAM3 Computer Vision Learning Journey  
Computer Vision · Artificial Intelligence · Machine Learning

[LinkedIn](https://www.linkedin.com/in/peyman-mxli/) | [GitHub](https://github.com/Peyman-mxli)
