# Practical — SAM3 Text Prompts

This folder contains the validated practical implementation for **Session 08 — SAM3 Text Prompts**.

The practical uses `SAM3SemanticPredictor` to locate and segment objects directly from natural-language concepts. It compares generic and specific prompts, inspects confidence and mask area, filters unreliable masks, labels reliable detections, and saves reproducible visual evidence.

---

## Practical Objective

- Segment objects from text prompts without detector-provided boxes
- Reuse one image with multiple semantic concepts
- Convert Ultralytics results to `sv.Detections`
- Inspect confidence scores and mask areas
- Filter detections using confidence and minimum area
- Compare `vehicle`, `bus`, `person`, and `wheel`
- Save validated visual outputs

---

## Structure

```text
practical/
├── README.md
├── sam3_text_prompts.py
└── assets/
    ├── README.md
    ├── input/
    │   ├── README.md
    │   ├── bus.jpg
    │   └── zidane.jpg
    └── output/
        ├── README.md
        ├── sam3_person_text_prompt_filtered.png
        └── sam3_text_prompts_comparison.png
```

---

## Validated Results

The completed Colab execution returned:

```text
vehicle: 1 object
bus:     1 object
person:  6 objects
wheel:   2 objects
```

Person-mask inspection:

```text
Object 0: confidence=0.972  area=32,208 px²
Object 1: confidence=0.970  area=45,738 px²
Object 2: confidence=0.966  area=21,145 px²
Object 3: confidence=0.943  area=11,440 px²
Object 4: confidence=0.762  area=2,803 px²
Object 5: confidence=0.301  area=547 px²
```

Filtering rules:

```text
Minimum confidence: 0.50
Minimum mask area:   1,000 px²
Original persons:    6
Filtered persons:    5
```

The low-confidence, small-area detection was removed.

---

## Workflow

```text
Input Image
     ↓
SAM3SemanticPredictor
     ↓
Natural-Language Prompt
     ↓
Segmentation Masks
     ↓
sv.Detections
     ↓
Confidence + Area Filtering
     ↓
Mask and Label Annotation
     ↓
Saved Outputs
```

---

## Run in Google Colab

Install dependencies and mount Google Drive, then run:

```bash
python sam3_text_prompts.py \
  --model /content/drive/MyDrive/SAM3-Models/sam3.pt \
  --image assets/input/bus.jpg \
  --output assets/output
```

The SAM 3 checkpoint is stored externally and is not committed to GitHub.

---

## Generated Outputs

- `sam3_person_text_prompt_filtered.png` — five reliable person masks with confidence labels
- `sam3_text_prompts_comparison.png` — four-panel comparison of vehicle, bus, person, and wheel prompts

**Status: Notebook execution completed and validated successfully in Google Colab.**
