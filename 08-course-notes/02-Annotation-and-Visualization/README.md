# Annotation and Visualization with Supervision

## Overview

This lesson explores how object detections can be transformed into clear and useful visual representations using the **Supervision** library.

After an object detection model such as **YOLO** identifies objects in an image, the raw detections contain information such as:

- Bounding box coordinates
- Class IDs
- Class names
- Confidence scores

However, these detections are only data. To make the results understandable to humans, we need to **visualize them**.

Supervision provides a collection of components called **Annotators** that can draw different visual elements on top of an image.

---

## Learning Objectives

By the end of this lesson, I will be able to:

- Understand what an Annotator is in Supervision
- Visualize YOLO detections using different Annotators
- Use `BoxAnnotator`
- Use `RoundBoxAnnotator`
- Use `HaloAnnotator`
- Use `BlurAnnotator`
- Use `BoxCornerAnnotator`
- Use `LabelAnnotator`
- Customize annotation colors and line thickness
- Customize label text size
- Combine multiple Annotators
- Understand why the order of annotation layers matters

---

## 1. Annotators in Supervision

An **Annotator** is a Supervision component responsible for adding a specific visual representation to an image.

For example, an Annotator can:

- Draw a bounding box around an object
- Add a text label
- Blur a detected object
- Add a halo around a detection
- Draw only the corners of a bounding box

This allows us to separate **object detection** from **object visualization**.

The model determines:

> **What and where is the object?**

The Annotator determines:

> **How should that detection be displayed?**

---

## 2. Annotators as Layers

A useful way to understand Annotators is to think about applications such as **Canva** or **Photoshop**.

In an image editor, we can add multiple layers:

1. Original image
2. Bounding boxes
3. Labels
4. Visual effects
5. Additional annotations

Supervision Annotators work in a similar way.

```text
image.copy()
    ↓
BoxAnnotator
    ↓
Image + Bounding Boxes
    ↓
LabelAnnotator
    ↓
Image + Bounding Boxes + Labels
    ↓
HaloAnnotator
    ↓
Image + Bounding Boxes + Labels + Halos
```

Each Annotator receives an image, adds its visualization, and returns the resulting image.

---

## 3. Why Does Layer Order Matter?

The order in which Annotators are applied is important because each new annotation is drawn **on top of the previous result**.

For example:

```python
scene = box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)

scene = label_annotator.annotate(
    scene=scene,
    detections=detections,
    labels=labels
)
```

In this example:

```text
Original Image
      ↓
BoxAnnotator
      ↓
Bounding Boxes
      ↓
LabelAnnotator
      ↓
Bounding Boxes + Labels
```

The boxes are drawn first and the labels are drawn afterward.

This makes the labels appear visually **on top of the boxes**.

### Key Concept

> **Detection tells us what the model found. Annotation determines how we communicate those results visually.**

---

## 4. Preparing YOLO Detections for Visualization

Before applying any Annotator, we first need detection results.

The general workflow is:

```text
Input Image
    ↓
YOLO Model
    ↓
YOLO Results
    ↓
Supervision Detections
    ↓
Annotators
    ↓
Final Visualization
```

YOLO performs the actual object detection, while Supervision makes the results easier to process and visualize.

---

## 5. Converting YOLO Results to Supervision Detections

After running YOLO, the results can be converted into a Supervision `Detections` object.

```python
results = model(image)[0]

detections = sv.Detections.from_ultralytics(
    results
)
```

The `Detections` object contains structured information about every detected object.

This can include:

- Bounding box coordinates
- Confidence score
- Class ID
- Additional detection information

---

## 6. Inspecting `detections`

Before visualizing the detections, it is useful to inspect what information is available.

This helps us understand what the model detected and which Annotators make sense for the result.

Important properties include:

```python
detections.xyxy
detections.confidence
detections.class_id
```

### `detections.xyxy`

Contains the coordinates of each bounding box.

```text
[x1, y1, x2, y2]
```

Where:

- `x1` = left coordinate
- `y1` = top coordinate
- `x2` = right coordinate
- `y2` = bottom coordinate

These coordinates define the rectangular region containing the detected object.

---

### `detections.confidence`

Contains the model's confidence score for each detection.

For example:

```text
0.92
0.87
0.76
```

A confidence of:

```text
0.92
```

means the model has approximately **92% confidence** in that detection.

---

### `detections.class_id`

Contains the numeric class assigned to each detected object.

For example:

```text
0
5
11
```

The class ID can then be translated into a human-readable class name using the YOLO model's class mapping.

---

## 7. Creating Labels

Labels make detections easier to understand by displaying information such as the object's class and confidence score.

A label can be created using:

```python
labels = [
    f"{results.names[class_id]} {confidence:.0%}"
    for class_id, confidence in zip(
        detections.class_id,
        detections.confidence
    )
]
```

This can produce labels such as:

```text
person 92%
bus 87%
stop sign 81%
```

These labels can later be passed to `LabelAnnotator`.

---

## 8. Why Inspect Detections First?

Visualization should not be treated as an isolated step.

Before choosing an Annotator, we should understand the information available inside the detection results.

For example:

```text
detections
    │
    ├── xyxy ──────────► Where is the object?
    │
    ├── class_id ──────► What type of object is it?
    │
    └── confidence ────► How confident is the model?
```

Once this information is available, we can decide how we want to communicate it visually.

This leads to the next stage:

> **Choosing the appropriate Supervision Annotator for the visualization task.**

---

## 9. Supervision Annotator Catalog

Supervision provides multiple Annotators that allow us to visualize the same detections in different ways.

In this lesson, we explore five important Annotators:

| Annotator | Purpose |
|---|---|
| `BoxAnnotator` | Draws standard bounding boxes |
| `RoundBoxAnnotator` | Draws bounding boxes with rounded corners |
| `HaloAnnotator` | Adds a halo-style visual effect |
| `BlurAnnotator` | Blurs the detected regions |
| `BoxCornerAnnotator` | Draws only the corners of the bounding boxes |

Each Annotator receives the same `detections`, but represents them differently.

---

## 10. BoxAnnotator

`BoxAnnotator` is one of the most common visualization tools in Supervision.

It draws a rectangular bounding box around every detected object.

```python
box_annotator = sv.BoxAnnotator()

scene = box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)
```

Conceptually:

```text
Detection
    ↓
[x1, y1, x2, y2]
    ↓
BoxAnnotator
    ↓
┌─────────────────┐
│ Detected Object │
└─────────────────┘
```

Bounding boxes are useful because they clearly show the location and approximate size of each detected object.

---

## 11. RoundBoxAnnotator

`RoundBoxAnnotator` performs a similar task to `BoxAnnotator`, but uses rounded corners.

```python
round_box_annotator = sv.RoundBoxAnnotator()

scene = round_box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)
```

The difference is primarily visual:

```text
BoxAnnotator             RoundBoxAnnotator

┌──────────────┐          ╭──────────────╮
│    Object    │          │    Object    │
└──────────────┘          ╰──────────────╯
```

Both represent the same detection information.

The visualization style changes, but the underlying detection does not.

---

## 12. HaloAnnotator

`HaloAnnotator` provides another way to visually emphasize detected objects.

```python
halo_annotator = sv.HaloAnnotator()

scene = halo_annotator.annotate(
    scene=image.copy(),
    detections=detections
)
```

Instead of relying only on a traditional rectangular box, the halo effect helps highlight the detected region.

This demonstrates an important concept:

> The same detection data can be represented using different visual styles.

---

## 13. BlurAnnotator

`BlurAnnotator` applies a blur effect to detected regions.

```python
blur_annotator = sv.BlurAnnotator()

scene = blur_annotator.annotate(
    scene=image.copy(),
    detections=detections
)
```

Conceptually:

```text
Original Image
      ↓
Detections
      ↓
BlurAnnotator
      ↓
Detected regions are blurred
```

Unlike a bounding box, which adds information around an object, `BlurAnnotator` modifies the appearance of the detected region itself.

---

## 14. BoxCornerAnnotator

`BoxCornerAnnotator` draws only the corners of the detection bounding boxes.

```python
corner_annotator = sv.BoxCornerAnnotator()

scene = corner_annotator.annotate(
    scene=image.copy(),
    detections=detections
)
```

Instead of:

```text
┌───────────────────┐
│                   │
│      Object       │
│                   │
└───────────────────┘
```

the visualization focuses on the corners:

```text
┌─                 ─┐


        Object


└─                 ─┘
```

This provides another visual alternative while still showing the location of the detection.

---

## 15. Comparing Multiple Annotators

The notebook compares several Annotators side by side:

```python
configs = [
    ("BoxAnnotator",       sv.BoxAnnotator()),
    ("RoundBoxAnnotator",  sv.RoundBoxAnnotator()),
    ("HaloAnnotator",      sv.HaloAnnotator()),
    ("BlurAnnotator",      sv.BlurAnnotator()),
    ("BoxCornerAnnotator", sv.BoxCornerAnnotator()),
    ("Box + Label (combo)", None),
]
```

Each Annotator is applied to a fresh copy of the original image:

```python
scene = annotator.annotate(
    scene=image.copy(),
    detections=detections
)
```

This makes it possible to compare different visualization methods using exactly the same detections.

---

## 16. Combining Box and Label Annotators

Annotators become especially useful when they are combined.

For example:

```python
scene = sv.BoxAnnotator().annotate(
    scene=image.copy(),
    detections=detections
)

scene = sv.LabelAnnotator().annotate(
    scene=scene,
    detections=detections,
    labels=labels
)
```

The first Annotator adds the bounding boxes.

The second Annotator adds the labels.

```text
Original Image
      ↓
BoxAnnotator
      ↓
Bounding Boxes
      ↓
LabelAnnotator
      ↓
Bounding Boxes + Labels
```

This is an example of **Annotator composition**.

Instead of asking one Annotator to perform every visualization task, multiple Annotators can be combined as layers.

---

## Important Implementation Detail

In the notebook, the Annotators are instantiated before they are reused:

```python
configs = [
    ("BoxAnnotator", sv.BoxAnnotator()),
    ("RoundBoxAnnotator", sv.RoundBoxAnnotator()),
    ("HaloAnnotator", sv.HaloAnnotator()),
    ("BlurAnnotator", sv.BlurAnnotator()),
    ("BoxCornerAnnotator", sv.BoxCornerAnnotator()),
]
```

This follows the lesson's recommendation to instantiate an Annotator once and reuse it instead of repeatedly recreating it inside a loop.

---

### Key Takeaway

Different Annotators do **not** change what YOLO detected.

They change **how those detections are visually communicated**.

```text
             YOLO Detection
                  │
       ┌──────────┼───────────┐
       ↓          ↓           ↓
     Boxes       Blur        Halo
       │          │           │
       └──────────┼───────────┘
                  ↓
           Visualization
```

---

## 17. Customizing Annotation Colors

Supervision allows us to customize the colors used by Annotators.

For example:

```python
box_annotator = sv.BoxAnnotator(
    color=sv.Color.RED
)
```

This changes the visual appearance of the bounding boxes without changing the underlying detections.

Conceptually:

```text
YOLO Detection
      ↓
sv.Detections
      ↓
BoxAnnotator
      ↓
Custom Color
      ↓
Visualization
```

The color belongs to the visualization layer, not to the detection model.

---

## 18. Using Color Palettes

Instead of assigning a single color to every detection, Supervision can use a color palette.

For example:

```python
box_annotator = sv.BoxAnnotator(
    color=sv.ColorPalette.DEFAULT
)
```

A color palette provides multiple colors that can be used when visualizing detections.

This is useful when an image contains several objects or classes.

```text
Detections
    │
    ├── Object 1 → Color A
    ├── Object 2 → Color B
    ├── Object 3 → Color C
    └── Object 4 → Color D
```

Color variation can make a visualization easier to interpret.

---

## 19. Customizing Box Thickness

The thickness of bounding boxes can also be modified.

```python
box_annotator = sv.BoxAnnotator(
    thickness=3
)
```

The `thickness` parameter controls the width of the lines used to draw the bounding boxes.

For example:

```text
thickness=1
      ↓
Thin bounding boxes

thickness=3
      ↓
Medium bounding boxes

thickness=5
      ↓
Thicker bounding boxes
```

The appropriate thickness depends on factors such as:

- Image resolution
- Object size
- Number of detections
- Intended display size

---

## 20. Combining Color and Thickness

Multiple visualization settings can be configured together.

For example:

```python
box_annotator = sv.BoxAnnotator(
    color=sv.ColorPalette.DEFAULT,
    thickness=3
)
```

This creates a `BoxAnnotator` with:

```text
Color Palette
      +
Line Thickness
      ↓
Customized Bounding Boxes
```

The detection data remains unchanged.

Only the way the detections are displayed is modified.

---

## 21. LabelAnnotator

Bounding boxes show **where** objects are located, but they do not automatically explain what each object is.

For that, we can use:

```python
sv.LabelAnnotator()
```

Example:

```python
label_annotator = sv.LabelAnnotator()
```

Then apply it:

```python
scene = label_annotator.annotate(
    scene=scene,
    detections=detections,
    labels=labels
)
```

The labels can contain information such as:

```text
person 92%
car 87%
bus 81%
```

This makes the visualization much more informative.

---

## 22. Customizing Label Text Size

The size of the label text can be customized.

For example:

```python
label_annotator = sv.LabelAnnotator(
    text_scale=0.6
)
```

The `text_scale` parameter controls the size of the displayed text.

For example:

```text
text_scale=0.3
      ↓
Smaller Text

text_scale=0.6
      ↓
Medium Text

text_scale=1.0
      ↓
Larger Text
```

The goal is to make the labels readable without covering too much of the image.

---

## 23. Why Label Size Matters

Label size is especially important when working with:

- High-resolution images
- Small detected objects
- Large numbers of detections
- Dense scenes
- Screenshots used for documentation

If the text is too small:

```text
Detection
    ↓
Tiny Label
    ↓
Difficult to Read
```

If the text is too large:

```text
Detection
    ↓
Large Label
    ↓
Object May Be Covered
```

A good visualization balances readability with image clarity.

---

## 24. Creating Class and Confidence Labels

A useful label combines the detected class with its confidence score.

```python
labels = [
    f"{results.names[class_id]} {confidence:.0%}"
    for class_id, confidence in zip(
        detections.class_id,
        detections.confidence
    )
]
```

For example:

```text
person 94%
car 89%
dog 83%
```

This gives the viewer two important pieces of information:

```text
Class Name
    +
Confidence Score
```

---

## 25. Understanding `zip()`

The label-generation code uses:

```python
zip(
    detections.class_id,
    detections.confidence
)
```

`zip()` pairs corresponding values from the two collections.

Conceptually:

```text
class_id        confidence
    0      +       0.94
    2      +       0.89
   16      +       0.83
```

Each pair can then be used to create one label.

For example:

```python
results.names[class_id]
```

converts the numeric class ID into a readable class name.

---

## 26. Formatting Confidence as a Percentage

The expression:

```python
{confidence:.0%}
```

formats the confidence score as a percentage.

For example:

```text
0.94 → 94%
0.89 → 89%
0.83 → 83%
```

Without formatting, a label might look like:

```text
person 0.943728
```

With percentage formatting:

```text
person 94%
```

The second version is easier for a human to interpret.

---

## 27. Combining Boxes and Labels

A common visualization pipeline combines `BoxAnnotator` and `LabelAnnotator`.

First create the Annotators:

```python
box_annotator = sv.BoxAnnotator(
    color=sv.ColorPalette.DEFAULT,
    thickness=3
)

label_annotator = sv.LabelAnnotator(
    text_scale=0.6
)
```

Then apply the boxes:

```python
annotated_image = box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)
```

Then apply the labels:

```python
annotated_image = label_annotator.annotate(
    scene=annotated_image,
    detections=detections,
    labels=labels
)
```

The pipeline becomes:

```text
Original Image
      ↓
BoxAnnotator
      ↓
Bounding Boxes
      ↓
LabelAnnotator
      ↓
Bounding Boxes + Labels
```

---

## 28. Reusing the Annotated Scene

An important pattern is:

```python
scene=image.copy()
```

for the first annotation layer.

After that, we reuse the resulting image:

```python
scene=annotated_image
```

This allows each new Annotator to build on the previous visualization.

For example:

```text
image.copy()
      ↓
Box Layer
      ↓
annotated_image
      ↓
Label Layer
      ↓
annotated_image
```

This is the foundation of multi-layer annotation pipelines.

---

## 29. Why Use `image.copy()`?

Using:

```python
image.copy()
```

creates a copy of the original image before drawing annotations.

This is useful because it preserves the original image.

Conceptually:

```text
Original Image
     │
     ├──────────────► Remains unchanged
     │
     └── copy()
           ↓
       Annotation
           ↓
     Modified Copy
```

This is especially important when comparing multiple visualization styles.

---

## 30. Comparing Different Visualization Styles

Suppose we want to compare:

```text
BoxAnnotator
RoundBoxAnnotator
HaloAnnotator
BlurAnnotator
BoxCornerAnnotator
```

Each visualization should begin with:

```python
image.copy()
```

For example:

```python
scene = annotator.annotate(
    scene=image.copy(),
    detections=detections
)
```

This ensures that every Annotator starts from the same original image.

Otherwise, annotations from previous experiments could remain on the image and affect the comparison.

---

## 31. Annotation Composition

Multiple Annotators can be combined to create more complex visualizations.

For example:

```text
Original Image
      ↓
BoxAnnotator
      ↓
EllipseAnnotator
      ↓
DotAnnotator
      ↓
LabelAnnotator
      ↓
Final Visualization
```

Each Annotator represents another visual layer.

The important idea is:

> **The detections remain the same while the visualization becomes richer.**

---

## 32. EllipseAnnotator

`EllipseAnnotator` provides another way to represent detections.

```python
ellipse_annotator = sv.EllipseAnnotator()
```

It can then be applied using:

```python
scene = ellipse_annotator.annotate(
    scene=scene,
    detections=detections
)
```

Instead of only drawing rectangular bounding boxes, this Annotator adds ellipse-based visualization around detected objects.

Conceptually:

```text
Detection
    ↓
EllipseAnnotator
    ↓
Ellipse Visualization
```

---

## 33. DotAnnotator

`DotAnnotator` adds a point-based visual representation to detections.

Create it using:

```python
dot_annotator = sv.DotAnnotator()
```

Then apply it:

```python
scene = dot_annotator.annotate(
    scene=scene,
    detections=detections
)
```

This provides another visual layer that can be combined with other Annotators.

---

## 34. Multi-Annotator Visualization

A complete multi-Annotator pipeline can combine:

```python
box_annotator = sv.BoxAnnotator(
    color=sv.ColorPalette.DEFAULT,
    thickness=3
)

ellipse_annotator = sv.EllipseAnnotator()

dot_annotator = sv.DotAnnotator()

label_annotator = sv.LabelAnnotator(
    text_scale=0.6
)
```

The Annotators can then be applied sequentially.

---

## 35. Layer 1 — Bounding Boxes

Start with the original image:

```python
scene = box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)
```

The visualization now contains bounding boxes.

```text
Original Image
      ↓
Bounding Boxes
```

---

## 36. Layer 2 — Ellipses

Use the result from the previous layer:

```python
scene = ellipse_annotator.annotate(
    scene=scene,
    detections=detections
)
```

Now the visualization contains:

```text
Bounding Boxes
      +
Ellipses
```

---

## 37. Layer 3 — Dots

Apply the `DotAnnotator`:

```python
scene = dot_annotator.annotate(
    scene=scene,
    detections=detections
)
```

The visualization now contains:

```text
Bounding Boxes
      +
Ellipses
      +
Dots
```

---

## 38. Layer 4 — Labels

Finally, apply the labels:

```python
scene = label_annotator.annotate(
    scene=scene,
    detections=detections,
    labels=labels
)
```

The final visualization contains:

```text
Bounding Boxes
      +
Ellipses
      +
Dots
      +
Class Labels
      +
Confidence Scores
```

---

## 39. Why Apply Labels Last?

Labels contain important textual information.

For example:

```text
person 94%
car 89%
bus 82%
```

If other visual layers are drawn after the labels, they may overlap or cover the text.

Therefore, a useful layer order is:

```text
Box
 ↓
Ellipse
 ↓
Dot
 ↓
Label
```

This keeps the labels visible on top of the other annotation layers.

---

## 40. Understanding the `scene` Variable

In a multi-Annotator pipeline, the variable:

```python
scene
```

represents the current version of the visualization.

For example:

```text
image.copy()
     ↓
scene
     ↓
Add Boxes
     ↓
scene
     ↓
Add Ellipses
     ↓
scene
     ↓
Add Dots
     ↓
scene
     ↓
Add Labels
     ↓
Final scene
```

The same variable can be updated after each annotation layer.

This creates a simple and reusable visualization pipeline.

---

## 41. Detection and Visualization Are Separate

One of the most important concepts in this lesson is the separation between:

```text
Detection
```

and:

```text
Visualization
```

YOLO is responsible for detection.

Supervision Annotators are responsible for visualization.

```text
YOLO
  ↓
Predictions
  ↓
sv.Detections
  ↓
Supervision Annotators
  ↓
Visualization
```

Changing the Annotator does not require changing the YOLO model.

---

## 42. What YOLO Controls

YOLO determines:

```text
What object was detected?
Where is the object?
How confident is the model?
```

The detection result contains information such as:

```text
Bounding Box
Class ID
Confidence Score
```

These are model predictions.

---

## 43. What Supervision Controls

Supervision determines how those predictions are displayed.

For example:

```text
Bounding Box Style
Color
Thickness
Labels
Text Size
Ellipse
Dot
Halo
Blur
Corner Style
```

These are visualization decisions.

Therefore:

```text
YOLO Detection
      ↓
Same Detection Data
      ↓
Different Annotators
      ↓
Different Visualizations
```

---

## 44. Visualization Does Not Change the Prediction

Suppose YOLO detects:

```text
person 94%
```

Using:

```python
sv.BoxAnnotator()
```

does not change the prediction.

Using:

```python
sv.RoundBoxAnnotator()
```

also does not change it.

Using:

```python
sv.HaloAnnotator()
```

still does not change it.

The detection remains:

```text
person 94%
```

Only its visual representation changes.

---

## 45. Confidence Threshold vs. Visualization Settings

It is important to distinguish between a detection setting and a visualization setting.

For example:

```python
CONFIDENCE_THRESHOLD = 0.50
```

can affect which predictions are accepted.

But:

```python
thickness=3
```

only changes how a box looks.

Similarly:

```python
text_scale=0.6
```

only changes the appearance of the label.

Conceptually:

```text
Confidence Threshold
        ↓
Detection Filtering
        ↓
Accepted Predictions May Change
```

while:

```text
Color
Thickness
Text Scale
Annotator Type
        ↓
Visualization
        ↓
Predictions Do Not Change
```

---

## 46. Building a Reusable Visualization Pipeline

The concepts from this lesson can be organized into a reusable pipeline.

```text
Input Image
     ↓
Load with OpenCV
     ↓
YOLO Inference
     ↓
Detection Results
     ↓
sv.Detections
     ↓
Create Labels
     ↓
Create Annotators
     ↓
Apply Annotation Layers
     ↓
Save Final Image
```

This architecture separates the different responsibilities of the application.

---

## 47. Complete Multi-Annotator Example

```python
import cv2
import supervision as sv

from ultralytics import YOLO


MODEL_NAME = "yolov8n.pt"

INPUT_IMAGE = "input/image.jpg"
OUTPUT_IMAGE = "output/annotated_image.jpg"

CONFIDENCE_THRESHOLD = 0.50


# Load image
image = cv2.imread(INPUT_IMAGE)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {INPUT_IMAGE}"
    )


# Load YOLO model
model = YOLO(MODEL_NAME)


# Run object detection
results = model(
    image,
    conf=CONFIDENCE_THRESHOLD
)[0]


# Convert YOLO results to Supervision
detections = sv.Detections.from_ultralytics(
    results
)


# Create labels
labels = [
    f"{results.names[class_id]} {confidence:.0%}"
    for class_id, confidence in zip(
        detections.class_id,
        detections.confidence
    )
]


# Create Annotators
box_annotator = sv.BoxAnnotator(
    color=sv.ColorPalette.DEFAULT,
    thickness=3
)

ellipse_annotator = sv.EllipseAnnotator()

dot_annotator = sv.DotAnnotator()

label_annotator = sv.LabelAnnotator(
    text_scale=0.6
)


# Layer 1 — Boxes
scene = box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)


# Layer 2 — Ellipses
scene = ellipse_annotator.annotate(
    scene=scene,
    detections=detections
)


# Layer 3 — Dots
scene = dot_annotator.annotate(
    scene=scene,
    detections=detections
)


# Layer 4 — Labels
scene = label_annotator.annotate(
    scene=scene,
    detections=detections,
    labels=labels
)


# Save result
success = cv2.imwrite(
    OUTPUT_IMAGE,
    scene
)

if not success:
    raise RuntimeError(
        f"Could not save image: {OUTPUT_IMAGE}"
    )


print(
    f"Detected objects: {len(detections)}"
)

print(
    f"Annotated image saved to: {OUTPUT_IMAGE}"
)
```

---

## 48. Complete Pipeline Architecture

The complete architecture can be represented as:

```text
input/image.jpg
      ↓
OpenCV
      ↓
YOLOv8
      ↓
Detection Results
      ↓
sv.Detections
      ↓
Create Labels
      ↓
BoxAnnotator
      ↓
EllipseAnnotator
      ↓
DotAnnotator
      ↓
LabelAnnotator
      ↓
output/annotated_image.jpg
```

This architecture demonstrates how object detection and visualization can be combined while remaining logically separate.

---

## 49. Practical Applications

Annotation and visualization are useful in many computer vision applications.

Examples include:

- Object-detection debugging
- Dataset inspection
- Model evaluation
- Security-camera analysis
- Traffic monitoring
- Industrial inspection
- Retail analytics
- Robotics
- Autonomous systems
- Sports analysis
- Educational demonstrations

Visualization allows developers and users to understand what the model is detecting.

---

## 50. Class Recording

The class recording for this lesson is available on my YouTube channel:

[Watch Lesson 02 — Annotation and Visualization](https://youtu.be/71h6y-YQOUA)

The recording covers the practical concepts documented in this lesson, including YOLO detections, Supervision Annotators, annotation customization, label visualization, and multi-layer annotation pipelines.

---

## 51. Lesson Notebook

The original notebook used during this lesson is included in this directory:

[`01_b_anotacion_visualizacion.ipynb`](./01_b_anotacion_visualizacion.ipynb)

The notebook contains the practical experiments and demonstrations used to study annotation and visualization with Supervision.

It serves as the original experimental environment, while the concept notes and practical exercises provide a more structured explanation of the material.

---

## 52. Lesson Structure

The lesson is organized into several types of learning material:

```text
02-Annotation-and-Visualization/
│
├── concepts/
│   ├── README.md
│   ├── 01-supervision-annotators.md
│   ├── 02-annotation-customization.md
│   └── 03-annotation-layers.md
│
├── practical-exercises/
│   ├── README.md
│   ├── 01-basic-box-annotation.md
│   ├── 02-label-annotation.md
│   ├── 03-annotation-customization.md
│   └── 04-multi-annotator-challenge.md
│
├── 01_b_anotacion_visualizacion.ipynb
├── CLASS-RECORDING.md
└── README.md
```

Each section has a different purpose:

| Section | Purpose |
|---|---|
| `README.md` | Complete lesson documentation |
| `CLASS-RECORDING.md` | Link to the class recording |
| Notebook | Original practical lesson notebook |
| `concepts/` | Detailed explanations of the main concepts |
| `practical-exercises/` | Hands-on exercises based on the lesson |

---

## 53. Repository Structure

This lesson connects to several other sections of the SAM3 Learning Journey repository.

```text
SAM3-Learning-Journey/
│
├── 03-notebooks/
│   └── Original Google Colab / Jupyter notebooks
│
├── 04-examples/
│   └── 02-Annotation-and-Visualization/
│
├── 05-projects/
│   └── 02-Multi-Annotator-Visualization-Pipeline/
│
├── 08-course-notes/
│   └── 02-Annotation-and-Visualization/
│       │
│       ├── concepts/
│       │   ├── README.md
│       │   ├── 01-supervision-annotators.md
│       │   ├── 02-annotation-customization.md
│       │   └── 03-annotation-layers.md
│       │
│       ├── practical-exercises/
│       │   ├── README.md
│       │   ├── 01-basic-box-annotation.md
│       │   ├── 02-label-annotation.md
│       │   ├── 03-annotation-customization.md
│       │   └── 04-multi-annotator-challenge.md
│       │
│       ├── 01_b_anotacion_visualizacion.ipynb
│       ├── CLASS-RECORDING.md
│       └── README.md
│
└── 09-assets/
    └── banners/
```

This structure separates:

- Original notebooks
- Reusable code examples
- Complete practical projects
- Detailed course notes
- Concept explanations
- Practical exercises
- Class recordings
- Repository assets

This makes the learning journey easier to navigate and maintain as new SAM3 lessons are added.

---

## Related Material

### Class Recording

The complete class recording is documented here:

[`CLASS-RECORDING.md`](./CLASS-RECORDING.md)

You can also watch the lesson directly on YouTube:

[Watch Lesson 02 — Annotation and Visualization](https://youtu.be/71h6y-YQOUA)

---

### Course Notebook

The original lesson notebook is available here:

[`01_b_anotacion_visualizacion.ipynb`](./01_b_anotacion_visualizacion.ipynb)

---

### Concepts

Detailed concept explanations are available in:

[`concepts/`](./concepts/)

The concepts section contains:

1. [`01-supervision-annotators.md`](./concepts/01-supervision-annotators.md)
2. [`02-annotation-customization.md`](./concepts/02-annotation-customization.md)
3. [`03-annotation-layers.md`](./concepts/03-annotation-layers.md)

These files separate the main theoretical concepts from the larger lesson README.

---

### Practical Exercises

Hands-on exercises are available in:

[`practical-exercises/`](./practical-exercises/)

The exercises progress from basic visualization to a complete multi-Annotator workflow:

1. [`01-basic-box-annotation.md`](./practical-exercises/01-basic-box-annotation.md)
2. [`02-label-annotation.md`](./practical-exercises/02-label-annotation.md)
3. [`03-annotation-customization.md`](./practical-exercises/03-annotation-customization.md)
4. [`04-multi-annotator-challenge.md`](./practical-exercises/04-multi-annotator-challenge.md)

The learning progression is:

```text
Basic Bounding Boxes
        ↓
Boxes + Labels
        ↓
Annotation Customization
        ↓
Multi-Annotator Challenge
```

---

### Code Examples

Small reusable Python examples based on this lesson are available in:

[`../../04-examples/02-Annotation-and-Visualization/`](../../04-examples/02-Annotation-and-Visualization/)

These examples are designed to isolate individual concepts so they can be studied and executed independently.

---

### Practical Project

The lesson concepts were combined into a complete reusable project:

[`../../05-projects/02-Multi-Annotator-Visualization-Pipeline/`](../../05-projects/02-Multi-Annotator-Visualization-Pipeline/)

The project combines:

- YOLOv8
- Ultralytics
- OpenCV
- Supervision
- `sv.Detections`
- `BoxAnnotator`
- `EllipseAnnotator`
- `DotAnnotator`
- `LabelAnnotator`
- Confidence thresholds
- Class and confidence labels
- Multiple visualization layers
- Automatic output generation

The project pipeline is:

```text
Input Image
     ↓
OpenCV
     ↓
YOLOv8
     ↓
Detection Results
     ↓
sv.Detections
     ↓
BoxAnnotator
     ↓
EllipseAnnotator
     ↓
DotAnnotator
     ↓
LabelAnnotator
     ↓
Annotated Image
```

**Project status:** Completed and tested successfully in Google Colab.

---

## Learning Progression

The complete learning workflow for this lesson is:

```text
Class
  ↓
Class Recording
  ↓
Course Notebook
  ↓
Detailed Lesson Notes
  ↓
Concept Notes
  ↓
Code Examples
  ↓
Practical Exercises
  ↓
Complete Project
```

This structure transforms the original lesson into multiple forms of reusable learning material.

---

## 54. What I Learned

After completing this lesson, I learned how object-detection results can be transformed into useful visual representations.

I learned that object detection and visualization are separate responsibilities.

YOLO performs the detection:

```text
What object is present?
Where is the object?
How confident is the model?
```

Supervision controls the visualization:

```text
How should the bounding box look?
What color should be used?
Should a label be displayed?
How large should the text be?
Should another visual effect be added?
```

I also learned how to:

- Convert Ultralytics YOLO results into `sv.Detections`
- Inspect bounding-box coordinates
- Access class IDs
- Access confidence scores
- Create readable class and confidence labels
- Use `BoxAnnotator`
- Use `RoundBoxAnnotator`
- Use `HaloAnnotator`
- Use `BlurAnnotator`
- Use `BoxCornerAnnotator`
- Use `EllipseAnnotator`
- Use `DotAnnotator`
- Use `LabelAnnotator`
- Customize bounding-box thickness
- Customize colors and color palettes
- Customize label text size
- Combine multiple Annotators
- Reuse the same detection data across multiple visualization styles
- Build layered annotation pipelines
- Understand why annotation order matters
- Preserve the original image using `image.copy()`
- Save the final visualization with OpenCV

---

## Detection vs. Visualization

The most important architectural concept from this lesson is:

```text
Detection ≠ Visualization
```

Detection produces structured information:

```text
Bounding Boxes
Class IDs
Confidence Scores
```

Visualization converts that information into something humans can easily interpret:

```text
Boxes
Labels
Colors
Dots
Ellipses
Halos
Blur
Other Visual Layers
```

Therefore:

```text
YOLO
  ↓
Detection Data
  ↓
sv.Detections
  ↓
Supervision
  ↓
Visual Representation
```

---

## From Lesson to Project

This lesson also demonstrates how a course concept can evolve into a complete project.

```text
Learn Annotators
      ↓
Experiment in Notebook
      ↓
Document Concepts
      ↓
Create Small Examples
      ↓
Complete Practical Exercises
      ↓
Combine Annotation Layers
      ↓
Build Reusable Project
```

The final result is:

```text
02-Multi-Annotator-Visualization-Pipeline
```

This project demonstrates how the concepts studied during the lesson can be organized into a reusable computer vision application.

---

## 55. Key Takeaway

The central idea of this lesson is:

> **Object detection determines what the model sees, while annotation determines how those detections are communicated visually.**

A single set of detections can support many different visual representations.

```text
                    ┌── Bounding Boxes
                    │
                    ├── Rounded Boxes
                    │
                    ├── Labels
YOLO → Detections ──┼── Ellipses
                    │
                    ├── Dots
                    │
                    ├── Halos
                    │
                    └── Blur
```

This means the detection model does not need to change when we want to change the visualization.

The complete architecture is:

```text
Input Image
     ↓
YOLOv8
     ↓
Detection Results
     ↓
sv.Detections
     ↓
Visualization Configuration
     ↓
Supervision Annotators
     ↓
Annotation Layers
     ↓
Final Visualization
```

This separation makes computer vision applications more:

- Flexible
- Reusable
- Readable
- Customizable
- Easier to debug
- Easier to document

---

## Lesson Status

| Component | Status |
|---|---|
| Main Lesson Notes | Completed |
| Class Recording | Completed |
| Course Notebook | Added |
| Concepts | Completed |
| Practical Exercises | Completed |
| Code Examples | Completed |
| Multi-Annotator Project | Completed & Tested |

**Lesson 02 — Annotation and Visualization: Completed**

---

## Author

**Peyman Miyandashti**

SAM3 Computer Vision Learning Journey  
Computer Vision · Artificial Intelligence · Machine Learning

[LinkedIn](https://www.linkedin.com/in/peyman-mxli/) | [GitHub](https://github.com/Peyman-mxli)
