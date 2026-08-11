# Exercise 01 — Image Loading with OpenCV

## Objective

The objective of this exercise is to learn how to:

- Download an image from the internet.
- Save the image locally.
- Load the image using OpenCV.
- Inspect the image dimensions.
- Understand how images are represented in Python.
- Understand the difference between BGR and RGB.
- Display the image correctly using Matplotlib.

---

## 1. Import the Required Libraries

We begin by importing the libraries required for this exercise:

```python
import urllib.request
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
```

Each library has a specific purpose:

| Library | Purpose |
|---|---|
| `urllib.request` | Download files from a URL |
| `pathlib.Path` | Create and manage directories |
| `cv2` | Load and process images with OpenCV |
| `matplotlib.pyplot` | Display images |

---

## 2. Create the Assets Directory

Before downloading the image, create a directory where the course assets can be stored:

```python
Path("assets").mkdir(exist_ok=True)
```

The option:

```python
exist_ok=True
```

means Python will not generate an error if the directory already exists.

Our structure becomes:

```text
project/
└── assets/
```

---

## 3. Download the Test Image

The course uses the Ultralytics bus image:

```python
urllib.request.urlretrieve(
    "https://ultralytics.com/images/bus.jpg",
    "assets/bus.jpg"
)
```

The workflow is:

```text
Internet
   ↓
Image URL
   ↓
urllib.request.urlretrieve()
   ↓
assets/bus.jpg
```

After downloading, the directory contains:

```text
assets/
└── bus.jpg
```

---

## 4. Load the Image with OpenCV

Now load the image:

```python
image = cv2.imread("assets/bus.jpg")
```

OpenCV reads the image and represents it internally as a numerical array.

Conceptually:

```text
bus.jpg
   ↓
cv2.imread()
   ↓
Image Array
   ↓
Python
```

---

## 5. Inspect the Image Shape

We can inspect the dimensions of the image using:

```python
print(image.shape)
```

The structure follows:

```text
(height, width, channels)
```

For a standard color image:

```text
channels = 3
```

These three channels represent the color information stored in the image.

---

## 6. Understanding Image Coordinates

Images use a coordinate system where the origin is located in the top-left corner:

```text
(0,0) ─────────────────────→ X
  │
  │
  │          IMAGE
  │
  │
  ▼
  Y
```

This means:

```text
X increases → left to right
Y increases → top to bottom
```

This coordinate system becomes important when we begin working with bounding boxes.

---

## 7. Understanding BGR

OpenCV loads color images using:

```text
BGR
```

The channels are ordered as:

```text
B → Blue
G → Green
R → Red
```

However, Matplotlib expects:

```text
RGB
```

with:

```text
R → Red
G → Green
B → Blue
```

Therefore, displaying an OpenCV image directly with Matplotlib may produce incorrect colors.

---

## 8. Convert BGR to RGB

We can convert the image using:

```python
rgb_image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)
```

The conversion process is:

```text
OpenCV
  ↓
BGR Image
  ↓
cv2.cvtColor()
  ↓
RGB Image
  ↓
Matplotlib
```

---

## 9. Display the Image

Now display the image:

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

## 10. Complete Exercise

The complete code for this exercise is:

```python
import urllib.request
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

# Create assets directory
Path("assets").mkdir(exist_ok=True)

# Download image
urllib.request.urlretrieve(
    "https://ultralytics.com/images/bus.jpg",
    "assets/bus.jpg"
)

# Load image
image = cv2.imread("assets/bus.jpg")

# Inspect dimensions
print("Image shape:", image.shape)

# Display image
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

## 11. Exercise Pipeline

```text
Image URL
    ↓
Download
    ↓
assets/bus.jpg
    ↓
cv2.imread()
    ↓
OpenCV Image
    ↓
Inspect Shape
    ↓
BGR → RGB
    ↓
Matplotlib
    ↓
Displayed Image
```

---

## Key Takeaways

After completing this exercise, we understand:

```text
cv2.imread()
→ Loads an image

image.shape
→ Shows height, width, and channels

OpenCV
→ Uses BGR

Matplotlib
→ Expects RGB

cv2.cvtColor()
→ Converts between color formats
```

This image will become the input for the YOLO object detection pipeline in the next practical exercise.
