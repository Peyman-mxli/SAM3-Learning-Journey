# 08 — SAM3 Text Prompts

This directory contains six focused Python examples for text-guided segmentation with `SAM3SemanticPredictor`.

The examples progress from a basic natural-language prompt to confidence analysis, filtering, specific object-part prompting, and a reusable multi-prompt workflow.

---

## Structure

```text
08-SAM3-Text-Prompts/
├── 01_basic_text_prompt.py
├── 02_concept_comparison.py
├── 03_confidence_and_mask_area.py
├── 04_filter_text_detections.py
├── 05_specific_wheel_prompt.py
├── 06_reusable_text_prompt_function.py
├── README.md
└── assets/
    ├── README.md
    ├── input/
    │   ├── README.md
    │   ├── bus.jpg
    │   └── zidane.jpg
    └── output/
        ├── README.md
        ├── 01_basic_text_prompt_output.png
        ├── 02_concept_comparison_output.png
        ├── 04_filter_text_detections_output.png
        ├── 05_specific_wheel_prompt_output.png
        └── 06_reusable_prompt_comparison_output.png
```

---

## Examples

### 01 — Basic Text Prompt

Segments all matching instances of `person` and visualizes their masks and boxes.

### 02 — Concept Comparison

Compares the prompts `vehicle`, `bus`, and `person` on the same image.

### 03 — Confidence and Mask Area

Prints the confidence score and pixel area of every `person` mask. This analytical example produces console output rather than an image.

### 04 — Filter Text Detections

Keeps masks satisfying both:

```text
confidence >= 0.50
mask area  >= 1,000 px²
```

### 05 — Specific Wheel Prompt

Demonstrates object-part segmentation using the specific concept `wheel`.

### 06 — Reusable Text-Prompt Function

Uses one reusable function to process `vehicle`, `bus`, `person`, and `wheel`, then saves a four-panel comparison.

---

## Validated Notebook Results

```text
vehicle: 1 object
bus:     1 object
person:  6 objects
wheel:   2 objects

Person detections before filtering: 6
Person detections after filtering:  5
```

Mask analysis:

```text
Object 0: confidence=0.972  area=32,208 px²
Object 1: confidence=0.970  area=45,738 px²
Object 2: confidence=0.966  area=21,145 px²
Object 3: confidence=0.943  area=11,440 px²
Object 4: confidence=0.762  area=2,803 px²
Object 5: confidence=0.301  area=547 px²
```

---

## Model Path

The scripts expect the SAM 3 checkpoint at:

```text
/content/drive/MyDrive/SAM3-Models/sam3.pt
```

The checkpoint is not stored in GitHub because of its size.

---

## Run

Run each example from this directory in Google Colab after mounting Google Drive and installing `ultralytics`, `supervision`, OpenCV, NumPy, Matplotlib, and PyTorch.

```bash
python 01_basic_text_prompt.py
python 02_concept_comparison.py
python 03_confidence_and_mask_area.py
python 04_filter_text_detections.py
python 05_specific_wheel_prompt.py
python 06_reusable_text_prompt_function.py
```

The stored outputs are evidence from the successfully completed Session 08 notebook. The standalone scripts reproduce those workflows using repository input assets.

---

## Learning Progression

```text
Basic Text Prompt
       ↓
Concept Comparison
       ↓
Confidence + Area Analysis
       ↓
Reliable-Mask Filtering
       ↓
Specific Object-Part Prompt
       ↓
Reusable Multi-Prompt Workflow
```

**Status: Example structure complete; validated notebook evidence preserved.**
