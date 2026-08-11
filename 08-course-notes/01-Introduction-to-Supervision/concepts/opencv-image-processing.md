# OpenCV Image Processing

## What is OpenCV?

**OpenCV (Open Source Computer Vision Library)** is a library used for image and video processing.

In this lesson, OpenCV is responsible for loading and preparing images before they are passed to the YOLO model.

It is imported with:

```python
import cv2
```

---

## Loading an Image

The course notebook loads the example image using:

```python
image = cv2.imread("assets/bus.jpg")
```

`cv2.imread()` reads the image from disk and converts it into an array that Python can process.

Conceptually:

```text
Image File
    ↓
cv2.imread()
    ↓
NumPy Array
    ↓
Computer Vision Processing
```

---

## Understanding Image Shape

We can inspect the dimensions of an image using:

```python
print(image.shape)
```

The result follows:

```text
(height, width, channels)
```

For example:

```text
(1080, 810, 3)
```

means:

```text
Height   → 1080 pixels
Width    → 810 pixels
Channels → 3
```

The three channels represent the color information of the image.

---

## OpenCV Uses BGR

One important concept is that OpenCV normally loads color images using:

```text
BGR
```

This means:

```text
B → Blue
G → Green
R → Red
```

However, many visualization libraries such as Matplotlib expect:

```text
RGB
```

which means:

```text
R → Red
G → Green
B → Blue
```

---

## BGR to RGB Conversion

If an OpenCV image is displayed directly with Matplotlib, the colors may appear incorrect.

We therefore convert the image:

```python
rgb_image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)
```

The transformation is:

```text
OpenCV Image
    ↓
BGR
    ↓
cv2.cvtColor()
    ↓
RGB
    ↓
Matplotlib
```

---

## Displaying an Image

After conversion, Matplotlib can display the image correctly:

```python
plt.figure(figsize=(12, 7))

plt.imshow(
    cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )
)

plt.axis("off")
plt.title("Test Image")
plt.show()
```

---

## OpenCV and YOLO

The image loaded by OpenCV can be passed directly to YOLO:

```python
results = model(image)[0]
```

This creates an important connection in our pipeline:

```text
Image File
    ↓
OpenCV
    ↓
NumPy Image
    ↓
YOLO
    ↓
Object Detection
```

---

## OpenCV and Supervision

After YOLO performs inference:

```python
results = model(image)[0]
```

the predictions are converted to:

```python
detections = sv.Detections.from_ultralytics(
    results
)
```

Supervision can then annotate a copy of the OpenCV image:

```python
annotated = box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)
```

---

## Why Use `image.copy()`?

When annotating an image, the course uses:

```python
image.copy()
```

This allows us to preserve the original image.

```text
Original OpenCV Image
        │
        ├──────────────→ Original remains unchanged
        │
        ▼
     image.copy()
        │
        ▼
     Annotation
        │
        ▼
  Annotated Image
```

This is useful when comparing different models, confidence thresholds, or annotation styles.

---

## Role of OpenCV in the Pipeline

In this lesson:

```text
OpenCV
  │
  ├── Loads the image
  ├── Represents image data
  ├── Converts BGR ↔ RGB
  └── Provides the image to YOLO
```

The complete relationship is:

```text
Image
  ↓
OpenCV
  ↓
YOLO
  ↓
sv.Detections
  ↓
Supervision Annotators
  ↓
OpenCV Image
  ↓
BGR → RGB
  ↓
Matplotlib
```

---

## Key Concept

Remember these three operations:

### Load

```python
image = cv2.imread("image.jpg")
```

### Convert BGR to RGB

```python
cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)
```

### Preserve the Original

```python
image.copy()
```

OpenCV provides the image-processing foundation that connects our image files to the YOLO and Supervision computer vision pipeline.
