# Documentation

This directory contains the validation and result documentation for **Project 08 — Advanced Mask Analysis Pipeline**.

The documentation records the final behavior of the pipeline after execution and validation.

---

## Purpose

The Project 08 pipeline produces several types of evidence:

```text
Project 08
    ↓
├── Source Code
├── Annotated Images
├── JSON Results
├── CSV Results
└── Documentation
```

The `docs/` directory explains and summarizes those generated results.

---

## Planned Files

```text
docs/
├── README.md
└── RESULTS.md
```

---

## RESULTS.md

`RESULTS.md` will document the validated execution of the complete pipeline.

It will include information such as:

- Images processed
- YOLOv8 detections
- SAM 3 masks generated
- Object classes
- Detection confidence
- Bounding-box measurements
- Mask-area measurements
- Occupancy ratios
- Generated annotated images
- JSON outputs
- CSV outputs
- Multi-image validation

The values recorded in `RESULTS.md` will come from the actual executed pipeline rather than estimated or placeholder values.

---

## Validation Evidence

Project 08 produces three primary forms of validation evidence.

### Visual Evidence

Stored in:

```text
data/output/
```

These files show:

```text
Original Image
      +
SAM 3 Segmentation Masks
      +
YOLO Bounding Boxes
```

### JSON Evidence

Stored in:

```text
results/json/
```

These files preserve structured detection and segmentation information for each processed image.

### CSV Evidence

Stored in:

```text
results/csv/
```

These files provide the same object-level analytical information in tabular form.

---

## Analysis Metrics

The documentation will summarize metrics generated for each segmented object:

```text
Object
├── Class ID
├── Class Name
├── Confidence
├── Bounding Box
├── Bounding-Box Area
├── Mask Area
└── Occupancy Ratio
```

The occupancy ratio is calculated as:

```text
Occupancy Ratio = Mask Area / Bounding-Box Area
```

This provides a quantitative relationship between YOLOv8 object localization and SAM 3 pixel-level segmentation.

---

## Documentation Workflow

```text
Run Project 08 Pipeline
        ↓
Generate Annotated Images
        ↓
Generate JSON Results
        ↓
Generate CSV Results
        ↓
Validate Outputs
        ↓
Document Actual Results
        ↓
docs/RESULTS.md
```

---

## Reproducibility

The documentation should describe results produced directly by:

```text
src/mask_analysis_pipeline.py
```

using the images stored in:

```text
data/input/
```

This keeps the documented results connected to reproducible project artifacts.

No performance values, detection counts, mask measurements, or other execution results should be added until they have been produced and verified by the actual pipeline.
