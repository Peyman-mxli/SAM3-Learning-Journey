# 10 — SAM3 Point Prompts

> **Course Session 11** — This repository starts at lesson `00`, so Course Session 11 is stored as folder `10`.

This lesson develops point-guided segmentation with SAM 3. Positive points select an object, negative points exclude unwanted regions, and detector- or text-derived coordinates enable reproducible refinement workflows.

## Learning objectives

- Segment an object with one positive point
- Inspect mask area and confidence
- Compare one, two, and three positive points
- Refine a mask with a negative point
- Compare text, point, and bounding-box prompts
- Use text discovery followed by point refinement
- Segment the first three YOLO detections with individual point prompts

## Repository structure

```text
10-SAM3-Point-Prompts/
├── README.md
├── CLASS-RECORDING.md
├── 05_a_sam3_prompts_puntos.ipynb
└── practical/
    ├── README.md
    ├── requirements.txt
    ├── common.py
    ├── 01_basic_positive_point.py
    ├── 02_multiple_positive_points.py
    ├── 03_positive_negative_refinement.py
    ├── 04_prompt_type_comparison.py
    ├── 05_text_to_point_refinement.py
    ├── 06_three_object_point_challenge.py
    └── assets/
        ├── README.md
        ├── input/
        │   ├── README.md
        │   └── bus.jpg
        └── output/
            ├── README.md
            └── 6 validated PNG results
```

Only `bus.jpg` is included because it is the input genuinely used by every validated experiment. No empty or unused input folder is preserved.

## Notebook

[Open the corrected lesson notebook](./05_a_sam3_prompts_puntos.ipynb)

The notebook now uses the verified checkpoint location:

```text
/content/drive/MyDrive/SAM3-Models/sam3.pt
```

Its extension challenge has also been completed: the first three YOLO detections are segmented, annotated, displayed, and saved.

## Point labels

| Label | Meaning | Purpose |
|---:|---|---|
| `1` | Positive point | Include the object containing the pixel |
| `0` | Negative point | Exclude the region containing the pixel |

## Validated results

The practical suite was executed in Google Colab using a Tesla T4 GPU.

| Experiment | Validated result |
|---|---|
| Basic positive point | Point `[413, 494]`; 1 mask |
| Mask analytics | 27,808 px²; confidence 0.683 |
| Positive/negative refinement | Positive `[413, 494]`; negative `[353, 414]` |
| Prompt comparison | Text: 6 people; Point: 1 object; YOLO boxes: 4 objects |
| Text-to-point refinement | 6 people discovered; refined center `[281, 628]` |
| Three-object challenge | 3 objects segmented |

All six visual results are available in [practical/assets/output](./practical/assets/output/).

## Run locally or in Colab

```bash
cd 08-course-notes/10-SAM3-Point-Prompts/practical
pip install -r requirements.txt
export SAM3_MODEL_PATH=/content/drive/MyDrive/SAM3-Models/sam3.pt
python 01_basic_positive_point.py
```

Run the remaining numbered scripts in order to reproduce the full lesson.

## Class recording

[Watch Course Session 11 — SAM3 Point Prompts](https://youtu.be/MmGHsYvRyVc)

## Status

**COMPLETE AND VALIDATED**

- Correct checkpoint path documented
- Extension challenge implemented
- Six practical scripts validated
- One genuine input preserved
- Six real outputs preserved
- Recording and reproducibility instructions included
