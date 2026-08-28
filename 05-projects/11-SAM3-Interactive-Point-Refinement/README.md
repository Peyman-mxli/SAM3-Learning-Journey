# Project 11 — SAM3 Interactive Point Refinement

This project converts the SAM3 Point Prompts lesson into an end-to-end analytical pipeline for object discovery, point-guided segmentation, and positive/negative refinement.

> Project `11` corresponds to course-notes lesson `10`. The projects directory is one number ahead because it contains one additional earlier project.

## Objective

Use YOLO to discover reproducible object coordinates, segment individual objects using SAM 3 positive points, measure every mask, and quantify how a negative point changes the first object mask.

## Structure

```text
11-SAM3-Interactive-Point-Refinement/
├── README.md
├── requirements.txt
├── config/
│   ├── README.md
│   └── pipeline.json
├── src/
│   ├── README.md
│   └── point_refinement_pipeline.py
├── data/
│   ├── README.md
│   ├── input/
│   │   ├── README.md
│   │   └── bus.jpg
│   └── output/
│       └── README.md
├── results/
│   ├── README.md
│   ├── csv/
│   │   └── README.md
│   └── json/
│       └── README.md
└── docs/
    ├── README.md
    ├── ARCHITECTURE.md
    ├── RESULTS.md
    └── LIMITATIONS.md
```

## Pipeline

1. Load and validate configuration.
2. Detect objects with YOLO.
3. Convert the first three box centers into positive point prompts.
4. Segment each object with SAM 3.
5. Record mask area and confidence.
6. Add a negative point to refine the first object.
7. Export two PNGs, two CSV reports, and one JSON report.

## Model Path

Default:

```text
/content/drive/MyDrive/SAM3-Models/sam3.pt
```

Override with the `SAM3_MODEL_PATH` environment variable.

## Run

```bash
python -m pip install -r requirements.txt
python src/point_refinement_pipeline.py
```

## Validation Status

- Source implementation: complete
- Python syntax: validated
- Required input: included
- Standalone Colab runtime: pending
- Runtime outputs and analytics: pending

No result is claimed before execution.

**Status: Implementation complete — runtime validation pending.**
