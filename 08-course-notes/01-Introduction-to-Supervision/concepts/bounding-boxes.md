# Bounding Boxes

## What is a Bounding Box?

A **bounding box** is a rectangular box used to represent the location of a detected object inside an image.

For example, if YOLO detects a person, car, or bus, a bounding box can be drawn around that object to show exactly where it appears.

```text
Image
   ↓
Object Detection
   ↓
Detected Object
   ↓
Bounding Box
```

---

## Bounding Box Coordinates

In this course, bounding boxes are represented using the `xyxy` format:

```text
[x1, y1, x2, y2]
```

Where:

- `x1` = left coordinate
- `y1` = top coordinate
- `x2` = right coordinate
- `y2` = bottom coordinate

Visually:

```text
(x1, y1)  ┌───────────────┐
          │               │
          │    OBJECT     │
          │               │
          └───────────────┘  (x2, y2)
```

The bounding box represents the smallest rectangle that completely surrounds the detected object.

---

## Image Coordinate System

In computer vision, the coordinate system begins at the **top-left corner** of the image.

```text
(0,0) ─────────────────────→ X
  │
  │
  │        IMAGE
  │
  │
  ↓
  Y
```

This means:

- X increases from left to right.
- Y increases from top to bottom.
- Coordinates are measured in pixels.

This is different from the coordinate system commonly used in mathematics.

---

## Example

Suppose YOLO returns this bounding box:

```python
[100, 50, 400, 300]
```

This means:

```text
x1 = 100
y1 = 50
x2 = 400
y2 = 300
```

The top-left corner is:

```text
(100, 50)
```

The bottom-right corner is:

```text
(400, 300)
```

---

## Calculating Width and Height

We can calculate the size of the detected object using:

```python
width = x2 - x1
height = y2 - y1
```

Example:

```python
x1, y1, x2, y2 = 100, 50, 400, 300

width = x2 - x1
height = y2 - y1

print("Width:", width)
print("Height:", height)
```

Result:

```text
Width: 300
Height: 250
```

---

## Bounding Boxes in Supervision

Supervision stores the bounding boxes of detected objects inside:

```python
detections.xyxy
```

For example:

```python
print(detections.xyxy)
```

The result is a NumPy array where **each row represents one detected object**:

```text
[
    [x1, y1, x2, y2],
    [x1, y1, x2, y2],
    [x1, y1, x2, y2]
]
```

For example:

```text
[
    [100, 50, 400, 300],
    [450, 120, 700, 500]
]
```

This means that two objects were detected.

---

## Inspecting One Detection

Supervision allows us to select an individual detection:

```python
first_detection = detections[0]
```

Then we can extract its bounding box:

```python
x1, y1, x2, y2 = first_detection.xyxy[0]
```

And inspect its position:

```python
print(f"Top-left: ({x1:.0f}, {y1:.0f})")
print(f"Bottom-right: ({x2:.0f}, {y2:.0f})")
```

We can also calculate its size:

```python
print(
    f"Size: {x2-x1:.0f}px wide × "
    f"{y2-y1:.0f}px high"
)
```

---

## Bounding Boxes in the Pipeline

Bounding boxes are part of the information produced during object detection:

```text
Input Image
     ↓
YOLO
     ↓
Detected Objects
     ↓
sv.Detections
     ↓
detections.xyxy
     ↓
Bounding Boxes
     ↓
Annotation
```

Supervision can then use these coordinates to draw boxes around the detected objects.

---

## Why Bounding Boxes Matter

Bounding boxes are fundamental to many computer vision applications:

- Object detection
- Object tracking
- People counting
- Vehicle detection
- Traffic analysis
- Security systems
- Industrial inspection
- Video analytics

They allow the system to understand not only **what** was detected, but also **where** it was detected.

---

## Key Concept

Remember the `xyxy` format:

```text
[x1, y1, x2, y2]

 x1 = LEFT
 y1 = TOP
 x2 = RIGHT
 y2 = BOTTOM
```

In Supervision:

```python
detections.xyxy
```

gives us the bounding-box coordinates for the detected objects.
