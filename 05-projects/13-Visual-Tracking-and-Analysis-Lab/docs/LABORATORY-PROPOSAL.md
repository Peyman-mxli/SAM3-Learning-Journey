# Original Laboratory Proposal

## SAM 3 — Computer Vision Laboratory

**Proposal title:** Sistema de Seguimiento y Análisis Visual  
**Run:** `45ef3e05`

## Project

Computer-vision tool for detecting, segmenting, and tracking objects or people in images and video while preserving historical results for later analysis.

The proposed implementation is a Python video-processing pipeline integrating tracking models and a database for querying sessions.

## Expected result

- Digital application
- Analysis tool
- Tracking dashboard

## Why this proposal fits the profile

The proposal connects skills in:

- Python
- computer vision
- databases
- programming

with the goal of processing recorded video and preserving structured evidence.

## Objective

Build an application to detect, track, and analyze the position and movement of entities in video sequences while exporting structured evidence.

## Learning objectives

- failure documentation
- model evaluation
- temporal identity
- IoU and Dice
- Supervision post-processing
- Precision and Recall
- tracking
- data visualization

## Evidence of mastery

- Python source code for the processing and tracking pipeline.
- Analysis tool and results-query interface.
- Performance-evaluation report under uncontrolled environmental conditions.

## Portfolio artifact

Documented test cases, results explorer, confusion matrix, and metrics report.

## MVP contract

### Includes

- Load a recorded image or video and retain date and source.
- Store identifier, timestamp, medium, result, confidence, and notes.
- Query previous observations and compare sessions from the same unit.
- Display evolution and evidence in a digital application, analysis tool, and tracking dashboard.
- Document errors related to illumination, scale, occlusion, and out-of-sample data.

### Not included yet

- Threshold alerts before validating metrics and false positives.
- Live capture, except as a later extension.

### Done when

- Recorded videos can be processed and tracking history is stored in a database.
- Exportable performance reports can be generated.

## Realization path

### 01 — Prepare

Technical feasibility test:

- 20–50 images or 2–5 short videos.
- 100–200 observations under varied conditions for MVP validation.
- Custom training is not required for the first test; document gaps if the base model cannot cover the task.

**Result:** reproducible technical test.

### 02 — Build

1. Load a recorded image or video and preserve date/source.
2. Store identifier, timestamp, medium, result, confidence, and notes.
3. Query previous observations and compare sessions.
4. Show evolution and evidence through a digital application, analysis tool, and tracking dashboard.
5. Document failures caused by illumination, scale, occlusion, and out-of-distribution data.

**Result:** the MVP produces and preserves evidence.

### 03 — Validate

- Evaluate false positives, omissions, and stability under varied conditions.
- Train or tune categories/regions only when the base model is insufficient.
- Verify capture, inference, and storage using 20–50 images or 2–5 short videos.

**Result:** documented success criteria and failure cases.

### 04 — Present

- Python source code.
- Analysis interface.
- Performance report for uncontrolled conditions.

**Result:** understandable and reproducible demo.

## Technologies associated with the proposal

- Supervision
- YOLO
- Ultralytics
- SAM 3

## Responsible-use principles

The system must not treat detections, identities, masks, or measurements as perfect ground truth. Results should be reported with confidence, limitations, and reproducible evaluation evidence.
