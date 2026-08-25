# 08 — SAM3 Text Prompts

This session introduces **text-prompt segmentation with SAM 3**.

Instead of providing a bounding box that tells the model where an object is located, SAM 3 can receive a natural-language concept such as:

```text
person
bus
vehicle
wheel
```

The model searches the image for every matching instance and produces segmentation masks for the requested concept.

This lesson introduces `SAM3SemanticPredictor` and compares text prompts with the bounding-box prompting workflow used in the previous SAM sessions.

---

## Session Status

```text
Session:         08 — SAM3 Text Prompts
Course Notes:    🟡 In Progress
Class Notebook:  ✅ Preserved
Class Recording: ⏳ Pending
Practical:       ⏳ Pending
Colab Test:      ⏳ Pending
Outputs:         ⏳ Pending
```

---

# Session Objective

The objective of this lesson is to understand how SAM 3 uses natural-language prompts to find and segment objects without requiring a separate detector to provide their locations.

The session explores:

- Text-based object segmentation
- `SAM3SemanticPredictor`
- Natural-language concepts
- Generic and specific prompts
- Segmentation-confidence thresholds
- Multiple-instance segmentation
- Mask visualization with Supervision
- Text prompts versus bounding-box prompts
- YOLO + SAM comparison
- Mask-area analysis
- Reusable semantic-segmentation workflows

---

# Topics Covered

This session covers:

- SAM 3 semantic prediction
- `SAM3SemanticPredictor`
- Text prompts
- Natural-language object concepts
- `predictor.set_image()`
- `predictor(text=[...])`
- Confidence thresholds
- FP16 inference on GPU
- `sv.Detections.from_ultralytics()`
- `sv.MaskAnnotator`
- `sv.BoxAnnotator`
- Mask opacity
- Mask-area calculation
- Specific versus generic concepts
- Text prompts versus YOLO bounding boxes
- Multiple-object segmentation
- Google Colab execution
- Practical validation

---

# Planned Session Structure

```text
08-SAM3-Text-Prompts/
│
├── README.md
├── CLASS-RECORDING.md
├── 04_a_sam3_prompts_texto.ipynb
│
└── practical/
    │
    ├── README.md
    ├── sam3_text_prompts.py
    │
    └── assets/
        │
        ├── README.md
        │
        ├── input/
        │   ├── README.md
        │   ├── bus.jpg
        │   └── zidane.jpg
        │
        └── output/
            └── README.md
```

The practical files and validated outputs will be added progressively after completing the notebook execution in Google Colab.

---

# Original Class Notebook

The original class notebook is preserved as:

[04_a_sam3_prompts_texto.ipynb](./04_a_sam3_prompts_texto.ipynb)

The notebook is maintained as the original course artifact for this session.

Its internal lesson numbering and original Spanish content are preserved without modification.

---

# What Are Text Prompts?

In the previous segmentation workflow, SAM received bounding-box coordinates:

```python
sam_model(
    image,
    bboxes=[[100, 50, 300, 400]]
)
```

This tells the model:

```text
The object is inside this region.
Segment it.
```

With SAM 3 text prompting, the model receives a concept:

```python
predictor(
    text=["person"]
)
```

This tells the model:

```text
Search the image.
Find every object matching "person."
Segment those objects.
```

The model is therefore responsible for both:

```text
Finding the requested concept
              +
Generating its segmentation masks
```

---

# Bounding-Box Prompts versus Text Prompts

## Bounding-Box Prompt

A bounding-box prompt requires the object location to be known first.

```text
Image
  ↓
Object Detector
  ↓
Bounding Box
  ↓
SAM
  ↓
Segmentation Mask
```

Example:

```python
results = sam_model(
    image,
    bboxes=detections.xyxy.tolist()
)
```

This workflow is useful when:

- A detector is already available
- The target classes are known
- Precise object localization is required
- The application uses a specialized trained detector

## Text Prompt

A text prompt describes the desired object using natural language.

```text
Image
  ↓
Text Concept
  ↓
SAM3SemanticPredictor
  ↓
Matching Object Instances
  ↓
Segmentation Masks
```

Example:

```python
results = predictor(
    text=["person"]
)
```

This workflow is useful when:

- No detector is available
- The user wants to search for a concept directly
- The requested object can be described using natural language
- Multiple matching instances must be located
- The application requires flexible prompts

---

# Images Used in the Lesson

The notebook downloads two Ultralytics example images:

```text
bus.jpg
zidane.jpg
```

The files are downloaded from:

```text
https://ultralytics.com/images/bus.jpg
https://ultralytics.com/images/zidane.jpg
```

They are stored locally inside:

```text
assets/
```

The main text-prompt experiments use:

```text
assets/bus.jpg
```

---

# Google Drive and SAM 3 Model

The notebook mounts Google Drive:

```python
from google.colab import drive

drive.mount("/content/drive")
```

The original notebook expects the SAM 3 checkpoint at:

```python
sam_path = "/content/drive/MyDrive/RandD/Archive_Zero_Resolved/sam3.pt"
```

Before running the complete notebook, this path must point to the actual location of `sam3.pt` in Google Drive.

The model file should be verified before initialization:

```python
from pathlib import Path

print("SAM 3 exists:", Path(sam_path).exists())
print("SAM 3 path:", sam_path)
```

---

# Required Libraries

The notebook installs and imports:

```python
!pip install supervision ultralytics
```

Main libraries:

```python
import supervision as sv
from ultralytics.models.sam import SAM3SemanticPredictor
from ultralytics import YOLO, SAM
import torch
import cv2
import matplotlib.pyplot as plt
```

| Library | Purpose |
|---|---|
| Ultralytics | YOLO, SAM and `SAM3SemanticPredictor` |
| Supervision | Detection conversion and annotation |
| PyTorch | GPU availability and FP16 support |
| OpenCV | Image loading and color conversion |
| Matplotlib | Result visualization |

---

# SAM3SemanticPredictor

The central component introduced in this session is:

```python
SAM3SemanticPredictor
```

It is initialized using an `overrides` dictionary:

```python
overrides = {
    "conf": 0.25,
    "task": "segment",
    "mode": "predict",
    "model": sam_path
}
```

When a CUDA GPU is available:

```python
if torch.cuda.is_available():
    overrides["half"] = True
```

The predictor is then created:

```python
predictor = SAM3SemanticPredictor(
    overrides=overrides
)
```

---

# Predictor Configuration

| Parameter | Example | Purpose |
|---|---:|---|
| `conf` | `0.25` | Minimum confidence required for a result |
| `task` | `"segment"` | Configures the model for segmentation |
| `mode` | `"predict"` | Enables inference mode |
| `model` | `sam_path` | Specifies the SAM 3 checkpoint |
| `half` | `True` | Enables FP16 inference when using a compatible GPU |

A lower confidence threshold may return more objects but can increase false positives.

A higher confidence threshold may reduce false positives but can omit valid objects.

---

# Loading an Image into the Predictor

The image is loaded using OpenCV:

```python
image = cv2.imread(
    "assets/bus.jpg"
)
```

It is then assigned to the semantic predictor:

```python
predictor.set_image(image)
```

`set_image()` allows the image to be loaded once and reused with different text prompts.

```text
Load Image Once
       ↓
Set Predictor Image
       ↓
Prompt: "person"
       ↓
Prompt: "vehicle"
       ↓
Prompt: "bus"
       ↓
Reuse the Same Image
```

---

# Text-Prompt Inference

The first semantic prompt used in the notebook is:

```text
person
```

Inference:

```python
results = predictor(
    text=["person"]
)[0]
```

The Ultralytics result is converted into Supervision detections:

```python
detections = sv.Detections.from_ultralytics(
    results
)
```

The resulting object contains:

- Bounding boxes
- Segmentation masks
- Confidence values
- Detected instances

---

# Visualizing Text-Prompt Results

The notebook uses:

```python
box_annotator = sv.BoxAnnotator()
mask_annotator = sv.MaskAnnotator(
    opacity=0.6
)
```

The masks are drawn first and the boxes are then drawn over the masks.

```text
Original Image
      ↓
Text-Prompt Masks
      ↓
Bounding Boxes
      ↓
Final Visualization
```

---

# Confidence and Mask-Area Inspection

SAM 3 assigns a confidence value to each text-prompt result.

```python
for index, confidence in enumerate(
    detections.confidence
):
    area = int(
        detections.mask[index].sum()
    )

    print(
        f"Object {index}: "
        f"confidence={confidence:.3f} "
        f"area={area:,} px2"
    )
```

This provides two measurements:

```text
Confidence
    ↓
How strongly the object matches the prompt

Mask Area
    ↓
How many image pixels belong to the object
```

---

# Experiment 1 — Specific versus Generic Concepts

The notebook compares:

```text
vehicle
bus
person
```

The experiment asks whether broad concepts produce more results and whether specific concepts produce cleaner results.

---

# Experiment 2 — Confidence Threshold

The notebook compares:

```text
0.1
0.3
0.6
```

```text
Lower Threshold
      ↓
More Candidate Objects
      ↓
Potential False Positives
```

```text
Higher Threshold
      ↓
Fewer Candidate Objects
      ↓
Potential Omissions
```

After the experiment, the original value is restored:

```python
predictor.args.conf = 0.25
```

---

# Experiment 3 — Text versus Bounding Box

The notebook compares two segmentation workflows.

## Text Workflow

```text
Image
  ↓
SAM3SemanticPredictor
  ↓
Prompt: "person"
  ↓
Person Masks
```

## Bounding-Box Workflow

```text
Image
  ↓
YOLOv8
  ↓
Person Bounding Boxes
  ↓
SAM 3
  ↓
Person Masks
```

The comparison investigates:

- Which workflow finds more people
- Which workflow produces more precise masks
- Whether the text prompt finds objects omitted by YOLO
- Whether bounding boxes provide stronger spatial guidance
- Which method produces false positives
- Which method is more flexible

---

# Extension Challenge

The notebook includes an extension challenge using:

```text
person
bus
wheel
```

The objective is to find the largest mask generated for each concept.

```python
areas = detections.mask.sum(
    axis=(1, 2)
)

largest_index = int(
    areas.argmax()
)

largest_detection = detections[
    largest_index
]
```

The expected visualization is a `1 × 3` comparison containing the largest matching instance for each text prompt.

---

# Planned Practical Implementation

After validating the notebook, the lesson will include:

```text
practical/sam3_text_prompts.py
```

The practical is expected to include reusable functions for:

```text
Image Downloading
        ↓
Image Loading
        ↓
SAM 3 Initialization
        ↓
Text-Prompt Inference
        ↓
Detection Conversion
        ↓
Mask Visualization
        ↓
Confidence Comparison
        ↓
Text-vs-Bbox Comparison
        ↓
Output Saving
```

The exact implementation and output names will be determined from the successful Google Colab execution.

---

# Planned Practical Assets

Expected input images:

```text
bus.jpg
zidane.jpg
```

Potential output visualizations:

```text
text_prompt_person.png
concept_comparison.png
confidence_comparison.png
text_vs_bbox.png
largest_mask_by_concept.png
```

These names are provisional until the practical is implemented and validated.

No output will be documented as completed until it has been generated and visually verified.

---

# Validation Requirements

The lesson will be considered complete after verifying:

- Google Drive mounted successfully
- `sam3.pt` found successfully
- Required packages installed successfully
- GPU availability checked
- `SAM3SemanticPredictor` initialized successfully
- `bus.jpg` downloaded successfully
- Text-prompt inference completed successfully
- Segmentation masks returned successfully
- Confidence values inspected
- Generic and specific prompts compared
- Confidence thresholds compared
- Text and bounding-box workflows compared
- Extension challenge completed
- Practical Python implementation executed
- Expected output files generated
- Output images visually inspected
- Practical documentation completed
- Class recording documented

---

# Expected Learning Outcomes

After completing this lesson, the learner should be able to:

- Explain the difference between text and bounding-box prompts
- Initialize `SAM3SemanticPredictor`
- Load an image for repeated semantic prompting
- Segment objects using natural-language concepts
- Convert SAM 3 results into `sv.Detections`
- Visualize semantic masks with Supervision
- Inspect confidence and mask-area values
- Compare generic and specific prompts
- Adjust the semantic confidence threshold
- Compare text-guided and detector-guided segmentation
- Select the largest mask for a requested concept
- Build a reusable text-prompt segmentation workflow

---

# Relationship to Previous Sessions

```text
Session 06
Basic YOLO + SAM Segmentation
             ↓
Session 07
Advanced Mask Visualization
             ↓
Session 08
Natural-Language SAM 3 Segmentation
```

Session 08 asks:

```text
Can SAM 3 find and segment objects directly
from a natural-language description?
```

---

# Technologies

```text
Python
Google Colab
Ultralytics
SAM 3
SAM3SemanticPredictor
YOLOv8
Supervision
OpenCV
PyTorch
Matplotlib
NumPy
```

---

# Current Progress

Completed:

- Lesson 08 folder created
- Original class notebook preserved
- Notebook content reviewed
- Lesson objectives identified
- Initial session documentation prepared

Pending:

- Google Colab execution
- Runtime results
- Class-recording URL
- Practical implementation
- Input assets
- Output assets
- Practical documentation
- Final validation
- Course index update

---

# Next Step

Run the original class notebook in Google Colab and record:

- Installed library versions
- GPU information
- SAM 3 checkpoint path
- Number of objects returned for each prompt
- Confidence values
- Mask areas
- Results at each confidence threshold
- Text-prompt result count
- Bounding-box result count
- Generated output images
- Any warnings or runtime errors

The verified results will be used to build the practical implementation and complete the final documentation.
