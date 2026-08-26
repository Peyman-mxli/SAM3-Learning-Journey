# Project 09 — SAM3 Semantic Prompt Analytics

## Overview

This project implements a professional semantic-segmentation pipeline that uses natural-language prompts to locate, segment, measure, filter, visualize, and export objects with SAM 3.

Unlike detector-guided projects, Project 09 does not require YOLO bounding boxes. The user supplies concepts such as `vehicle`, `bus`, `person`, or `wheel`, and `SAM3SemanticPredictor` performs both concept localization and mask generation.

---

## Project Goal

```text
Input Image
     ↓
Configurable Text Prompts
     ↓
SAM3SemanticPredictor
     ↓
Matching Object Masks
     ↓
Confidence + Mask-Area Analysis
     ↓
Reliability Filtering
     ↓
Visual Evidence + JSON + CSV
```

---

## Main Features

- Configurable prompt list in JSON
- Multi-prompt semantic segmentation
- Predictor image reuse with `set_image()`
- Per-object confidence, mask area, and bounding coordinates
- Reliability classification using confidence and mask area
- Filtered mask visualization with labels
- Multi-panel prompt comparison
- Automatic image discovery
- JSON analytical export
- Detection-level CSV export
- Prompt-summary CSV export
- Reusable functions and organized artifacts

---

## Project Structure

```text
09-SAM3-Semantic-Prompt-Analytics/
├── README.md
├── requirements.txt
├── config/
│   ├── README.md
│   └── prompts.json
├── src/
│   ├── README.md
│   └── semantic_prompt_analytics.py
├── data/
│   ├── README.md
│   ├── input/
│   │   ├── README.md
│   │   └── bus.jpg
│   └── output/
│       ├── README.md
│       ├── bus_filtered_person.png
│       └── bus_prompt_comparison.png
├── results/
│   ├── README.md
│   ├── json/
│   │   ├── README.md
│   │   └── bus_semantic_analysis.json
│   └── csv/
│       ├── README.md
│       ├── bus_detections.csv
│       └── bus_prompt_summary.csv
└── docs/
    ├── README.md
    ├── RESULTS.md
    └── LIMITATIONS.md
```

---

## Configuration

```json
{
  "prompts": ["vehicle", "bus", "person", "wheel"],
  "model_confidence": 0.25,
  "confidence_min": 0.5,
  "area_min": 1000,
  "filtered_prompt": "person"
}
```

The filtering rule is:

```text
Reliable = confidence >= 0.50 AND mask area >= 1,000 px²
```

---

## Validated Results

| Prompt | Objects |
|---|---:|
| vehicle | 1 |
| bus | 1 |
| person | 6 |
| wheel | 2 |

For `person`, five of six masks passed the reliability filter. The removed candidate had confidence `0.301` and mask area `547 px²`.

---

## Structured Analytics

Each object record can contain:

```text
image
prompt
object_id
confidence
mask_area
x1, y1, x2, y2
reliable
```

The prompt summary contains:

```text
image
prompt
objects
reliable_objects
```

---

## Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

The SAM 3 checkpoint must exist at:

```text
/content/drive/MyDrive/SAM3-Models/sam3.pt
```

Run:

```bash
python src/semantic_prompt_analytics.py
```

Or provide another checkpoint/configuration:

```bash
python src/semantic_prompt_analytics.py --model /path/to/sam3.pt --config config/prompts.json
```

---

## Relationship to Today’s Lesson

Session 08 introduced direct text-prompt segmentation, prompt specificity, confidence inspection, mask-area measurement, filtering, and reusable prompt comparison. Project 09 integrates those individual exercises into one configurable analytics system with persistent visual and structured outputs.

---

## Validation Status

- Project structure: complete
- Source syntax validation: complete
- Notebook-derived results: preserved
- Input and visual evidence: preserved
- JSON and CSV evidence: preserved from validated notebook values
- Full standalone Colab rerun: pending

The repository distinguishes between verified notebook evidence and the future standalone pipeline rerun; it does not invent unexecuted performance results.
