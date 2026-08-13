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
       ┌───────────┼───────────┐
       ↓           ↓           ↓
     Boxes        Blur        Halo
       │           │           │
       └───────────┼───────────┘
                   ↓
           Visualization
```

---

## 17. Customizing Annotation Colors

Supervision allows us to customize how bounding boxes are displayed.

One of the simplest options is changing the annotation color.

In this lesson, three different configurations are compared:

```python
anotador_rojo = sv.BoxAnnotator(
    color=sv.Color.RED,
    thickness=3
)

anotador_verde = sv.BoxAnnotator(
    color=sv.Color.GREEN,
    thickness=3
)

anotador_paleta = sv.BoxAnnotator(
    color=sv.ColorPalette.DEFAULT,
    thickness=3
)
```

This demonstrates two important approaches:

- Using a specific color
- Using an automatic color palette

---

## 18. Using Predefined Colors

Supervision provides predefined colors through `sv.Color`.

For example:

```python
sv.Color.RED
sv.Color.GREEN
```

These colors can be passed directly to an Annotator:

```python
box_annotator = sv.BoxAnnotator(
    color=sv.Color.RED,
    thickness=3
)
```

This creates bounding boxes using the selected color.

The same idea can be used with another color:

```python
box_annotator = sv.BoxAnnotator(
    color=sv.Color.GREEN,
    thickness=3
)
```

---

## 19. Using ColorPalette.DEFAULT

When an image contains several object classes, manually assigning a color to every class can become inconvenient.

Supervision provides:

```python
sv.ColorPalette.DEFAULT
```

It can be used like this:

```python
box_annotator = sv.BoxAnnotator(
    color=sv.ColorPalette.DEFAULT,
    thickness=3
)
```

According to the lesson, `ColorPalette.DEFAULT` automatically assigns different colors to different classes.

This is especially useful when working with many categories.

```text
Multiple Classes
      ↓
ColorPalette.DEFAULT
      ↓
Automatic Colors
      ↓
Easier Visual Differentiation
```

---

## 20. Comparing Color Configurations

The notebook compares the three configurations side by side:

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (titulo, ann) in zip(axes, [
    ("Color.RED",            anotador_rojo),
    ("Color.GREEN",          anotador_verde),
    ("ColorPalette.DEFAULT", anotador_paleta),
]):
    scene = ann.annotate(
        scene=image.copy(),
        detections=detections
    )

    ax.imshow(
        cv2.cvtColor(
            scene,
            cv2.COLOR_BGR2RGB
        )
    )

    ax.set_title(titulo)
    ax.axis("off")

plt.tight_layout()
plt.show()
```

The detection results remain the same.

Only their visual representation changes.

---

## 21. Customizing Bounding Box Thickness

Another important visualization parameter is:

```python
thickness=
```

This controls the thickness of the bounding-box lines.

The lesson compares:

```python
[1, 4, 10]
```

using:

```python
fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5)
)

for ax, thickness in zip(
    axes,
    [1, 4, 10]
):
    ann = sv.BoxAnnotator(
        thickness=thickness
    )

    scene = ann.annotate(
        scene=image.copy(),
        detections=detections
    )

    ax.imshow(
        cv2.cvtColor(
            scene,
            cv2.COLOR_BGR2RGB
        )
    )

    ax.set_title(
        f"thickness={thickness}"
    )

    ax.axis("off")

plt.tight_layout()
plt.show()
```

This produces three versions of the same detections:

```text
thickness=1      thickness=4      thickness=10
     │                │                 │
     ↓                ↓                 ↓
 Thin Box         Medium Box         Thick Box
```

---

## 22. Choosing the Appropriate Thickness

There is no single thickness value that is ideal for every image.

The lesson highlights two important considerations.

### High-Resolution Images

For high-resolution images, a larger thickness may be necessary.

```text
High Resolution
      ↓
More Pixels
      ↓
Thin Lines May Be Difficult to See
      ↓
Increase Thickness
```

### Small Displays

For smaller displays, excessive thickness can cover part of the detected object.

```text
Small Display
      ↓
Limited Visible Area
      ↓
Very Thick Boxes
      ↓
Object Can Become Obscured
```

The appropriate value therefore depends on the image and how the visualization will be displayed.

---

## 23. Visualization Is Part of the Pipeline

Changing color or thickness does not affect YOLO's predictions.

For example:

```python
sv.BoxAnnotator(
    color=sv.Color.RED,
    thickness=10
)
```

does **not** cause YOLO to detect objects differently.

Instead:

```text
YOLO
 │
 ├── Object Classes
 ├── Confidence Scores
 └── Bounding Box Coordinates
          │
          ↓
     Supervision
          │
          ├── Color
          ├── Thickness
          └── Visualization Style
```

YOLO is responsible for **detection**.

Supervision Annotators are responsible for **visualization**.

---

### Experiment Reflection

The notebook asks:

> **What thickness is the most readable?**

The answer depends on factors such as:

- Image resolution
- Object size
- Display size
- Number of detections
- Desired visual clarity

This experiment demonstrates why visualization parameters should be selected according to the context of the application.

---

## 24. LabelAnnotator

Bounding boxes show **where** an object is located, but they do not automatically explain what the detected object is.

For this reason, Supervision provides:

```python
sv.LabelAnnotator()
```

`LabelAnnotator` can display text associated with each detection.

For example, labels can contain:

```text
person 92%
bus 87%
stop sign 81%
```

This makes the visualization much easier to understand.

---

## 25. Combining BoxAnnotator and LabelAnnotator

A common visualization pipeline combines:

```python
sv.BoxAnnotator()
```

with:

```python
sv.LabelAnnotator()
```

The notebook uses:

```python
box_ann = sv.BoxAnnotator()
label_ann = sv.LabelAnnotator()
```

The bounding boxes are applied first:

```python
scene = box_ann.annotate(
    scene=image.copy(),
    detections=detections
)
```

Then the labels are added:

```python
scene = label_ann.annotate(
    scene=scene,
    detections=detections,
    labels=labels
)
```

The complete process becomes:

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

## 26. Customizing Label Text Size

Supervision allows us to control the size of label text using:

```python
text_scale=
```

The notebook experiments with three different values:

```python
[0.3, 0.6, 1.0]
```

The experiment uses:

```python
fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5)
)

for ax, scale in zip(
    axes,
    [0.3, 0.6, 1.0]
):
    box_ann = sv.BoxAnnotator()

    label_ann = sv.LabelAnnotator(
        text_scale=scale
    )

    scene = box_ann.annotate(
        scene=image.copy(),
        detections=detections
    )

    scene = label_ann.annotate(
        scene=scene,
        detections=detections,
        labels=labels
    )

    ax.imshow(
        cv2.cvtColor(
            scene,
            cv2.COLOR_BGR2RGB
        )
    )

    ax.set_title(
        f"text_scale={scale}"
    )

    ax.axis("off")

plt.tight_layout()
plt.show()
```

---

## 27. Comparing Text Scale

The three values produce different label sizes:

```text
text_scale=0.3
      ↓
 Small Text

text_scale=0.6
      ↓
 Medium Text

text_scale=1.0
      ↓
 Large Text
```

The goal is not simply to make the text as large as possible.

The goal is to make the information **readable without unnecessarily covering the image**.

---

## 28. Choosing an Appropriate Text Scale

The notebook asks an important question:

> **Is `text_scale=0.3` readable in this image?**

There is no universal answer.

The appropriate text size depends on factors such as:

- Image resolution
- Size of detected objects
- Number of detections
- Display size

For example:

```text
Very Small Text
      ↓
More Image Visible
      ↓
Labels May Be Difficult to Read
```

While:

```text
Very Large Text
      ↓
Easy to Read
      ↓
May Cover Important Parts of the Image
```

A good visualization requires balancing these two factors.

---

## 29. Labels Add Meaning to Bounding Boxes

Consider a bounding box by itself:

```text
┌────────────────────┐
│                    │
│       Object       │
│                    │
└────────────────────┘
```

We know something was detected, but we do not immediately know its class or confidence.

Adding a label provides more information:

```text
person 92%
┌────────────────────┐
│                    │
│       Person       │
│                    │
└────────────────────┘
```

This combines:

```text
Localization
     +
Classification
     +
Confidence
     ↓
More Informative Visualization
```

---

## 30. Detection Data vs. Visual Presentation

At this point, it is useful to separate the different responsibilities in the pipeline.

```text
YOLO
 │
 ├── Bounding Box Coordinates
 ├── Class IDs
 └── Confidence Scores
          │
          ↓
Supervision Detections
          │
          ↓
Annotators
 │
 ├── Bounding Boxes
 ├── Labels
 ├── Colors
 ├── Thickness
 └── Text Scale
          │
          ↓
Final Visualization
```

The detection model produces the prediction data.

Supervision controls how that information is presented visually.

---

### Key Takeaway

`LabelAnnotator` transforms detection information into readable text that can be displayed directly on the image.

The `text_scale` parameter allows the visualization to be adapted to different image resolutions and object sizes.

---
## 31. The Order of Annotation Layers Matters

One of the most important concepts in this lesson is that Annotators are applied sequentially.

Every Annotator receives the result produced by the previous Annotator.

Because of this, the order in which Annotators are applied changes the final visualization.

The notebook compares two different orders:

```text
Order A

Image
  ↓
BoxAnnotator
  ↓
LabelAnnotator
  ↓
Final Image
```

and:

```text
Order B

Image
  ↓
LabelAnnotator
  ↓
BoxAnnotator
  ↓
Final Image
```

The detections are exactly the same.

Only the **drawing order** changes.

---

## 32. Order A — Box → Label

The recommended order in the notebook is:

```text
Box → Label
```

First, the bounding boxes are drawn:

```python
orden_a = box_ann.annotate(
    scene=image.copy(),
    detections=detections
)
```

Then the labels are added:

```python
orden_a = label_ann.annotate(
    scene=orden_a,
    detections=detections,
    labels=labels
)
```

The complete code is:

```python
box_ann = sv.BoxAnnotator(thickness=3)
label_ann = sv.LabelAnnotator()

orden_a = box_ann.annotate(
    scene=image.copy(),
    detections=detections
)

orden_a = label_ann.annotate(
    scene=orden_a,
    detections=detections,
    labels=labels
)
```

The resulting layer structure is:

```text
┌─────────────────────────┐
│       Label Layer       │  ← Top
├─────────────────────────┤
│        Box Layer        │
├─────────────────────────┤
│     Original Image      │  ← Bottom
└─────────────────────────┘
```

Because the label is drawn last, it appears on top of the existing bounding box.

This makes the text easier to read.

---

## 33. Order B — Label → Box

The notebook also demonstrates the opposite order:

```text
Label → Box
```

First, the labels are drawn:

```python
orden_b = label_ann.annotate(
    scene=image.copy(),
    detections=detections,
    labels=labels
)
```

Then the bounding boxes are drawn:

```python
orden_b = box_ann.annotate(
    scene=orden_b,
    detections=detections
)
```

The complete structure becomes:

```text
┌─────────────────────────┐
│        Box Layer        │  ← Top
├─────────────────────────┤
│       Label Layer       │
├─────────────────────────┤
│     Original Image      │  ← Bottom
└─────────────────────────┘
```

Because the bounding box is drawn after the label, the box line can cover part of the text.

---

## 34. Comparing Both Orders

The notebook displays both results side by side:

```python
fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(14, 5)
)

ax1.imshow(
    cv2.cvtColor(
        orden_a,
        cv2.COLOR_BGR2RGB
    )
)

ax1.set_title(
    "Box → Label (recomendado)"
)

ax1.axis("off")

ax2.imshow(
    cv2.cvtColor(
        orden_b,
        cv2.COLOR_BGR2RGB
    )
)

ax2.set_title(
    "Label → Box"
)

ax2.axis("off")

plt.tight_layout()
plt.show()
```

This experiment demonstrates that the same Annotators can produce different visual results depending on their order.

---

## 35. Why Is Box → Label More Readable?

The notebook asks:

> **Why is "Box → Label" more readable?**

The reason is simple.

`LabelAnnotator` draws its text on top of everything that already exists in the image.

Therefore:

```text
Box First
    ↓
Label Second
    ↓
Label Appears on Top
    ↓
Better Readability
```

If the order is reversed:

```text
Label First
    ↓
Box Second
    ↓
Box Appears on Top
    ↓
Box Line May Cover Text
```

---

## 36. Thinking in Layers

This concept becomes even more important when three or more Annotators are combined.

For example:

```python
scene = image.copy()

scene = box_annotator.annotate(
    scene=scene,
    detections=detections
)

scene = halo_annotator.annotate(
    scene=scene,
    detections=detections
)

scene = label_annotator.annotate(
    scene=scene,
    detections=detections,
    labels=labels
)
```

Conceptually:

```text
Original Image
      ↓
Box
      ↓
Halo
      ↓
Label
      ↓
Final Visualization
```

The last Annotator is visually placed on top of the previous annotations.

---

## 37. Annotation Composition

Combining several Annotators is called **composition**.

Instead of producing only one visualization effect, we can build a custom visualization pipeline.

```text
Detection Results
       ↓
┌─────────────────┐
│ Annotator Layer │
│       #1        │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Annotator Layer │
│       #2        │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Annotator Layer │
│       #3        │
└────────┬────────┘
         ↓
 Final Visualization
```

Each layer can have a different responsibility.

For example:

```text
BoxAnnotator
     ↓
Shows object location

LabelAnnotator
     ↓
Shows class and confidence

HaloAnnotator
     ↓
Adds visual emphasis
```

Together, they create a richer visualization.

---

### Key Takeaway

> **The last Annotator applied is visually placed on top of the previous annotations.**

Therefore, Annotator order should be chosen intentionally.

For labels, the notebook recommends:

```text
Box → Label
```

rather than:

```text
Label → Box
```

because drawing the label last helps keep the text readable.

---
## 38. Extension Challenge

The final exercise of this lesson is to create a custom visualization using **at least three different Annotators**.

The goal is to move beyond the default bounding-box visualization and experiment with different ways of representing detections.

The notebook suggests exploring:

```python
sv.DotAnnotator()
sv.TriangleAnnotator()
sv.EllipseAnnotator()
```

These can be combined with the Annotators already explored earlier in the lesson.

---

## 39. DotAnnotator

`DotAnnotator` provides another visual representation for detections.

The notebook suggests experimenting with:

```python
sv.DotAnnotator()
```

It can be added to the annotation pipeline in the same way as the previous Annotators:

```python
scene = sv.DotAnnotator().annotate(
    scene=scene,
    detections=detections
)
```

This demonstrates that Supervision visualization is not limited to traditional bounding boxes.

---

## 40. TriangleAnnotator

Another Annotator suggested by the notebook is:

```python
sv.TriangleAnnotator()
```

It can be applied using:

```python
scene = sv.TriangleAnnotator().annotate(
    scene=scene,
    detections=detections
)
```

This provides another option for visually marking detected objects.

---

## 41. EllipseAnnotator

The third additional Annotator suggested in the challenge is:

```python
sv.EllipseAnnotator()
```

It can be used with:

```python
scene = sv.EllipseAnnotator().annotate(
    scene=scene,
    detections=detections
)
```

Again, the underlying detections remain unchanged.

Only their visual representation changes.

---

## 42. Building a Custom Combination

The notebook provides the following starting structure:

```python
scene = image.copy()

# scene = sv.??Annotator().annotate(
#     scene=scene,
#     detections=detections
# )

# scene = sv.??Annotator().annotate(
#     scene=scene,
#     detections=detections
# )

# scene = sv.??Annotator().annotate(
#     scene=scene,
#     detections=detections,
#     labels=labels
# )
```

The task is to replace the placeholders with a custom combination of Annotators.

For example, a visualization pipeline could conceptually follow:

```text
Original Image
      ↓
Annotator #1
      ↓
Annotator #2
      ↓
Annotator #3
      ↓
Final Custom Visualization
```

---

## 43. Example Custom Composition

One possible experiment is:

```python
scene = image.copy()

scene = sv.EllipseAnnotator().annotate(
    scene=scene,
    detections=detections
)

scene = sv.DotAnnotator().annotate(
    scene=scene,
    detections=detections
)

scene = sv.LabelAnnotator().annotate(
    scene=scene,
    detections=detections,
    labels=labels
)
```

The pipeline becomes:

```text
Original Image
      ↓
EllipseAnnotator
      ↓
DotAnnotator
      ↓
LabelAnnotator
      ↓
Final Visualization
```

This uses three different Annotators while keeping the label as the final layer.

---

## 44. Displaying the Final Visualization

The notebook displays the custom result using Matplotlib:

```python
plt.figure(figsize=(12, 7))

plt.imshow(
    cv2.cvtColor(
        scene,
        cv2.COLOR_BGR2RGB
    )
)

plt.axis("off")

plt.title(
    "Mi combinación personalizada"
)

plt.show()
```

This allows the final annotation composition to be inspected visually.

---

## 45. Parameters for Further Experimentation

The lesson also encourages experimenting with parameters such as:

```python
color=
```

and, depending on the Annotator:

```python
thickness=
```

or:

```python
radius=
```

This makes it possible to customize both the type and appearance of the annotations.

The experimentation process can therefore include:

```text
Choose Annotator
      ↓
Choose Color
      ↓
Adjust Thickness / Radius
      ↓
Choose Layer Order
      ↓
Compare Result
```

---

## 46. Why Experiment with Different Annotators?

There is no requirement that every computer vision application use exactly the same visualization.

Different applications may require different visual representations.

The important concept from this lesson is that visualization can be **composed and customized**.

```text
YOLO Predictions
       ↓
Supervision Detections
       ↓
Custom Annotator Pipeline
       ↓
Final Visualization
```

The detection model and visualization system therefore have different responsibilities.

---

## 47. Main Concepts Learned

This lesson introduced several important concepts:

- Supervision **Annotators**
- `BoxAnnotator`
- `RoundBoxAnnotator`
- `HaloAnnotator`
- `BlurAnnotator`
- `BoxCornerAnnotator`
- `LabelAnnotator`
- Annotation colors
- `ColorPalette.DEFAULT`
- Bounding-box thickness
- Label `text_scale`
- Annotator composition
- Annotation layer order
- Custom visualization pipelines
- `DotAnnotator`
- `TriangleAnnotator`
- `EllipseAnnotator`

---

## Lesson Summary

The complete workflow explored in this lesson can be summarized as:

```text
Input Image
     ↓
YOLO Model
     ↓
Detection Results
     ↓
sv.Detections
     ↓
┌──────────────────────┐
│ Supervision          │
│ Annotators           │
├──────────────────────┤
│ Boxes                │
│ Labels               │
│ Halos                │
│ Blur                 │
│ Corners              │
│ Dots                 │
│ Triangles            │
│ Ellipses             │
└──────────┬───────────┘
           ↓
Customized Visualization
```

The central idea is:

> **Object detection determines what was found and where it was found. Annotators determine how those detections are presented visually.**

By composing multiple Annotators in the correct order, we can build visualization pipelines that are clearer and better suited to the needs of a computer vision application.

---

## 48. Technologies Used

The concepts and experiments in this lesson use the following technologies:

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| YOLOv8 | Object detection model |
| Ultralytics | YOLO model interface |
| Supervision | Detection processing and visualization |
| OpenCV | Image processing |
| Matplotlib | Displaying and comparing images |
| Google Colab | Notebook execution environment |

---

## 49. Python Libraries

The main libraries used in this lesson are:

```python
import cv2
import supervision as sv
import matplotlib.pyplot as plt

from ultralytics import YOLO
```

The required packages can be installed with:

```bash
pip install supervision ultralytics
```

---

## 50. Lesson Notebook

The original practical lesson is contained in:

```text
01_b_anotacion_visualizacion.ipynb
```

Notebook topic:

```text
02 — Anotación y Visualización
```

The notebook contains the practical experiments used throughout this documentation.

---

## 51. Lesson Structure

The lesson is organized into three main stages:

```text
Introduction and Core Concepts
            ↓
Practical Development and Demonstration
            ↓
Analysis and Edge Cases
```

The notebook estimates approximately:

| Section | Estimated Time |
|---|---:|
| Introduction and Core Concepts | 15 min |
| Practical Development and Demonstration | 25 min |
| Analysis and Edge Cases | 20 min |

---

## 52. Repository Structure

This lesson is documented inside the SAM3 Learning Journey repository:

```text
SAM3-Learning-Journey/
│
├── 08-course-notes/
│   │
│   ├── 00-Agentic-AI-Programming/
│   │
│   ├── 01-Introduction-to-Supervision/
│   │
│   └── 02-Annotation-and-Visualization/
│       └── README.md
│
└── 09-assets/
    └── banners/
```

This structure keeps the course material organized by topic while separating documentation from reusable visual assets.

---

## 53. What I Learned

After completing this lesson, I can now explain the difference between **object detection data** and **object detection visualization**.

I learned that YOLO is responsible for detecting objects, while Supervision Annotators provide flexible tools for presenting those detections.

I also learned how to:

- Inspect detection information before visualization
- Draw bounding boxes
- Add labels
- Apply alternative visualization styles
- Customize colors
- Customize bounding-box thickness
- Adjust label text size
- Combine multiple Annotators
- Control visualization through layer order
- Build custom annotation pipelines

The most important concept is that Annotators can be treated as **visual layers**.

```text
Detection
    ↓
Visualization Layer
    ↓
Visualization Layer
    ↓
Visualization Layer
    ↓
Final Result
```

The order of these layers directly affects the final visualization.

---

## 54. Key Takeaway

```text
YOLO
  ↓
Detects Objects
  ↓
Supervision Detections
  ↓
Annotators
  ↓
Visualizes Results
```

**Detection answers:**

> What did the model find and where?

**Annotation answers:**

> How should those results be shown to the user?

---

## Author

**Peyman Miyandashti**

SAM3 Learning Journey  
Computer Vision · Artificial Intelligence · Machine Learning

[LinkedIn](https://www.linkedin.com/in/peyman-mxli/) | [GitHub](https://github.com/Peyman-mxli)

---
