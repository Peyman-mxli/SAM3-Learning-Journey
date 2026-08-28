# Practical — SAM3 Point Prompts

This directory contains the reproducible implementation validated for Course Session 11.

## Setup

```bash
pip install -r requirements.txt
export SAM3_MODEL_PATH=/content/drive/MyDrive/SAM3-Models/sam3.pt
```

The environment variable is optional in the established Colab layout because `common.py` uses that path by default.

## Run

```bash
python 01_basic_positive_point.py
python 02_multiple_positive_points.py
python 03_positive_negative_refinement.py
python 04_prompt_type_comparison.py
python 05_text_to_point_refinement.py
python 06_three_object_point_challenge.py
```

## What each script demonstrates

| Script | Purpose |
|---|---|
| `01_basic_positive_point.py` | Segment one YOLO-selected object and report mask analytics |
| `02_multiple_positive_points.py` | Compare one, two, and three positive prompts |
| `03_positive_negative_refinement.py` | Exclude background with a negative point |
| `04_prompt_type_comparison.py` | Compare text, point, and bounding-box prompts |
| `05_text_to_point_refinement.py` | Discover instances with text and refine one with a point |
| `06_three_object_point_challenge.py` | Segment the first three YOLO detections |

## Validation

All scripts were executed successfully in Google Colab with a Tesla T4. The committed PNG files are the actual generated outputs, not illustrative placeholders.
