# 10 — SAM3 Point Prompts

This session introduces **point-guided segmentation with SAM 3**.

A point prompt identifies a pixel that belongs to the target object. SAM then infers the complete object mask. Positive points include regions, while negative points exclude unwanted regions and refine ambiguous masks.

> **Numbering note:** The supplied source material contains inconsistent internal numbering. This repository records it as Session 10 to preserve the Learning Journey sequence. The original notebook is retained unchanged.

---

# Session Objective

The objective is to build and evaluate interactive segmentation workflows using positive and negative point prompts.

The notebook asks the learner to:

- Use `points=` and `labels=` with SAM 3
- Generate point coordinates from YOLO detections
- Segment an object using one positive point
- Inspect mask count, area, and confidence
- Compare one, two, and three positive points
- Add a negative point to exclude background
- Compare text, point, and bounding-box prompts
- Combine text discovery with point refinement
- Complete an extension challenge for the first three YOLO detections

---

# Current Structure

```text
10-SAM3-Point-Prompts/
├── README.md
├── CLASS-RECORDING.md
└── 05_a_sam3_prompts_puntos.ipynb
```

The practical implementation and output assets will be added only after the notebook is executed and the extension challenge is genuinely completed.

---

# Original Class Notebook

[05_a_sam3_prompts_puntos.ipynb](./05_a_sam3_prompts_puntos.ipynb)

The notebook is preserved as the source course artifact. Its original Spanish content and internal numbering are not silently rewritten.

---

# Point Labels

SAM uses labels to distinguish inclusion from exclusion:

| Label | Meaning | Purpose |
|---:|---|---|
| `1` | Positive point | Include the object containing this pixel |
| `0` | Negative point | Exclude the region containing this pixel |

Example:

```python
sam_model.predict(
    source=image,
    points=[positive_point, negative_point],
    labels=[1, 0]
)
```

---

# Lesson Workflow

```text
Load SAM 3 and bus.jpg
          ↓
Detect objects with YOLOv8
          ↓
Use a box center as a point
          ↓
Segment with one positive point
          ↓
Inspect masks, areas, and confidence
          ↓
Compare multiple positive points
          ↓
Add a negative point
          ↓
Compare text, point, and box prompts
          ↓
Text discovery → point refinement
          ↓
Complete the three-object challenge
```

---

# Required Environment

The notebook expects:

```text
Google Colab
Python
Ultralytics
SAM 3 checkpoint
YOLOv8n
Supervision
OpenCV
NumPy
Matplotlib
```

The source notebook contains this checkpoint path:

```text
/content/drive/MyDrive/RandD/Archive_Zero_Resolved/sam3.pt
```

For this Learning Journey, we will verify and use the actual configured checkpoint path before inference rather than assuming the source path exists.

---

# Input Images

The notebook downloads:

```text
bus.jpg
zidane.jpg
```

The primary experiments use `bus.jpg`. These inputs will be copied into the practical assets only after successful execution confirms which files are genuinely used.

---

# Experiments

## 1. YOLO-Guided Point Selection

YOLO detects objects and the notebook calculates the center of the first bounding box:

```python
point = [
    int((x1 + x2) / 2),
    int((y1 + y2) / 2)
]
```

## 2. One Positive Point

A positive label requests the object containing the selected pixel:

```python
labels=[1]
```

## 3. Multiple Positive Points

The notebook compares one, two, and three points inside the same detected object to determine whether additional guidance changes the mask.

## 4. Positive and Negative Points

A positive point selects the object and a negative point marks background to exclude:

```python
labels=[1, 0]
```

## 5. Prompt-Type Comparison

The same scene is analyzed with:

```text
Text prompt
Point prompt
YOLO bounding-box prompt
```

## 6. Text Discovery and Point Refinement

The semantic text predictor discovers people automatically. The center of one discovered instance becomes a point prompt for individual refinement.

---

# Required Extension Challenge

The notebook’s final task is incomplete by design.

The learner must:

1. Select the first three YOLO detections.
2. Calculate one center point for each object.
3. Run SAM point-prompt segmentation independently.
4. Annotate the generated masks.
5. Show the three results side by side.
6. Display object IDs and class IDs.
7. Save the completed visual evidence.

The supplied final cell currently displays the original image because the mask-annotation lines are commented out. We will complete and validate this cell in Colab before calling Lesson 10 finished.

---

# Validation Requirements

Lesson 10 will be considered complete only after verifying:

- Google Drive mounted
- Real `sam3.pt` path confirmed
- Required packages installed
- `bus.jpg` loaded
- YOLO detection completed
- First point coordinate calculated
- Positive-point segmentation completed
- Mask count, area, and confidence inspected
- One-, two-, and three-point comparison completed
- Negative-point experiment completed
- Text, point, and box comparison completed
- Text-to-point workflow completed
- Three-object extension challenge completed
- Final outputs saved and visually inspected
- Completed notebook downloaded
- Practical structure created from validated work

---

# Status

**IN PROGRESS**

Completed:

- Source notebook reviewed
- Professional Lesson 10 folder created
- Notebook preserved
- Tasks and validation criteria documented

Pending:

- Colab execution
- Checkpoint-path correction
- Extension-challenge completion
- Practical code
- Validated input/output assets
- Full class-recording URL
