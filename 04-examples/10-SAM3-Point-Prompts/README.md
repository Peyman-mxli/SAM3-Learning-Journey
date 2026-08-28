# 10 — SAM3 Point Prompts

This directory contains six focused examples for guiding SAM 3 with positive and negative point prompts.

> **Course Session 11** — Repository lessons start at `00`, so Course Session 11 is stored as Example `10`.

## Structure

```text
10-SAM3-Point-Prompts/
├── common.py
├── 01_basic_positive_point.py
├── 02_multiple_positive_points.py
├── 03_positive_negative_refinement.py
├── 04_prompt_type_comparison.py
├── 05_text_to_point_refinement.py
├── 06_three_object_point_challenge.py
├── README.md
├── requirements.txt
└── assets/
    ├── README.md
    ├── input/
    │   ├── README.md
    │   └── bus.jpg
    └── output/
        └── README.md
```

## Examples

### 01 — Basic Positive Point

Uses the center of the first YOLO detection as a positive SAM point and reports mask area and confidence.

### 02 — Multiple Positive Points

Compares one, two, and three positive points inside the same detected object.

### 03 — Positive and Negative Refinement

Uses label `1` to include the object and label `0` to exclude an unwanted region.

### 04 — Prompt-Type Comparison

Compares text, point, and YOLO bounding-box guidance on the same image.

### 05 — Text-to-Point Refinement

Uses the text prompt `person` for discovery, then converts one discovered box center into a point for individual control.

### 06 — Three-Object Challenge

Completes the notebook challenge by segmenting and annotating the first three YOLO objects. The original unfinished cell displayed the source image because its annotation lines were commented out; this implementation fixes that behavior.

## Model Configuration

Default checkpoint:

```text
/content/drive/MyDrive/SAM3-Models/sam3.pt
```

Override it when necessary:

```bash
export SAM3_MODEL_PATH=/path/to/sam3.pt
```

## Run

```bash
python -m pip install -r requirements.txt
python 01_basic_positive_point.py
python 02_multiple_positive_points.py
python 03_positive_negative_refinement.py
python 04_prompt_type_comparison.py
python 05_text_to_point_refinement.py
python 06_three_object_point_challenge.py
```

## Validation Status

- Source structure: complete
- Python syntax: validated
- Required input: included
- Runtime: Google Colab, Tesla T4
- Standalone examples executed: 6 of 6
- Visual outputs generated: 6 of 6
- Runtime errors: 0

## Validated Runtime Results

```text
Example 01
Positive point:  [413, 494]
Masks generated: 1
Mask area:       27,808 px²
Confidence:      0.683

Example 04
Text prompt:     6 people
Point prompt:    1 selected object
YOLO boxes:      4 objects

Example 05
Text discovery:  6 people
Refined center:  [281, 628]

Example 06
Objects segmented: 3
```

Ultralytics automatically adjusted point-prompt inference size from `1024` to `1036` and semantic-prompt size from `640` to `644` to satisfy model stride requirements. These were non-fatal informational warnings.

The six stored PNG files are the original outputs from the validated Colab/Tesla T4 execution.

**Status: Complete — all standalone examples and outputs validated.**
