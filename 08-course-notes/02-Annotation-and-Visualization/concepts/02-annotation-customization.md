# Annotation Customization

## Overview

Supervision Annotators can be customized to control how object detections appear in an image.

Instead of using only default visualization settings, developers can modify properties such as:

- Colors
- Color palettes
- Line thickness
- Text size
- Label content
- Annotation style

Customization makes detection results clearer and more suitable for different computer vision applications.

---

## Why Customize Annotations?

Default annotations are useful for quick visualization, but different projects may require different visual styles.

For example:

```text
Small Image
    ↓
Smaller Labels

High-Resolution Image
    ↓
Thicker Bounding Boxes

Many Object Classes
    ↓
Different Colors

Presentation / Documentation
    ↓
Clearer Labels and Visual Layers
```

The visualization should make the detection results easy to understand.

---

## Customizing BoxAnnotator

A basic `BoxAnnotator` can be created with:

```python
box_annotator = sv.BoxAnnotator()
```

The appearance can also be customized.

Example:

```python
box_annotator = sv.BoxAnnotator(
    color=sv.ColorPalette.DEFAULT,
    thickness=3
)
```

In this example:

```text
color
```

controls the colors used for the bounding boxes.

```text
thickness
```

controls the width of the bounding-box lines.

---

## Color Palettes

Supervision provides color palettes that can help visually distinguish detections.

Example:

```python
color=sv.ColorPalette.DEFAULT
```

Using a palette allows different detections or classes to appear with different colors.

Conceptually:

```text
Detection 1 → Color A
Detection 2 → Color B
Detection 3 → Color C
```

This can make images containing many detected objects easier to interpret.

---

## Bounding Box Thickness

The thickness of bounding boxes can be changed.

Example:

```python
box_annotator = sv.BoxAnnotator(
    thickness=3
)
```

A larger value creates thicker lines.

```text
Small thickness
      ↓
Thin Bounding Box

Large thickness
      ↓
Thick Bounding Box
```

The appropriate thickness depends on the image resolution and visualization requirements.

---

## Customizing LabelAnnotator

Labels can also be customized.

Example:

```python
label_annotator = sv.LabelAnnotator(
    text_scale=0.6
)
```

The `text_scale` parameter controls the size of the displayed label text.

This is useful when working with images of different sizes.

---

## Creating Custom Labels

Labels do not have to display only the object class.

They can combine multiple pieces of detection information.

Example:

```python
labels = [
    f"{results.names[class_id]} {confidence:.0%}"
    for class_id, confidence in zip(
        detections.class_id,
        detections.confidence
    )
]
```

This produces labels such as:

```text
person 89%
car 84%
bus 92%
dog 94%
```

Each label contains:

```text
Class Name
    +
Confidence Score
```

---

## Confidence Formatting

The expression:

```python
confidence:.0%
```

converts a confidence value into a percentage.

For example:

```text
0.89 → 89%
0.94 → 94%
0.72 → 72%
```

This makes model confidence easier to read in the final visualization.

---

## Combining Customization Options

Different customization options can be used together.

Example:

```python
box_annotator = sv.BoxAnnotator(
    color=sv.ColorPalette.DEFAULT,
    thickness=3
)

label_annotator = sv.LabelAnnotator(
    text_scale=0.6
)
```

The resulting pipeline becomes:

```text
Detection Data
      ↓
Custom Colors
      ↓
Custom Box Thickness
      ↓
Custom Labels
      ↓
Custom Text Size
      ↓
Final Visualization
```

---

## Visualization Readability

Customization should improve readability rather than simply add more visual elements.

Important considerations include:

- Avoid labels that are too large
- Avoid bounding boxes that hide important image details
- Use colors that make detections distinguishable
- Keep confidence information readable
- Avoid unnecessary visual clutter

The goal is to make model predictions easier to interpret.

---

## Detection vs. Appearance

The visualization settings do not change the actual detections.

For example:

```text
YOLO Detection
      ↓
Same Detection Data
      ↓
Different Annotation Settings
      ↓
Different Visual Appearance
```

Changing a bounding-box color does not change:

- The detected object
- Bounding-box coordinates
- Class ID
- Confidence score

It only changes how the detection is displayed.

---

## Key Takeaway

Annotation customization allows developers to control how computer vision predictions are presented without modifying the underlying detection results.

The main principle is:

```text
Detection Data
      ↓
Visualization Configuration
      ↓
Clear Visual Representation
```

Good annotation design makes computer vision results easier to analyze, debug, demonstrate, and communicate.
