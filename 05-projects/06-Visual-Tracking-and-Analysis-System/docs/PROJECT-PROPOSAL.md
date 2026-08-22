# Project Proposal — Visual Tracking and Analysis System

## Origin

This project proposal was generated through the **Computer Vision Laboratory** provided as part of the **SAM3 Computer Vision Learning Program**.

The original generated project was titled:

**Sistema de Seguimiento y Análisis Visual**

This document preserves the objectives, scope, expected results, learning goals, validation strategy, MVP requirements, and portfolio evidence defined in the original proposal.

---

# Project

## Visual Tracking and Analysis System

A computer vision tool designed to detect, segment, and track objects or people in images and recorded video while storing and analyzing historical results.

The project consists of developing a Python-based video-processing pipeline integrating tracking models and databases for querying processing sessions.

---

# Expected Result

The expected result is an:

**Exploration and Evaluation Tool**

The proposal defines the expected result as a combination of:

- Digital application
- Analysis tool
- Tracking dashboard

---

# Guiding Question

The project is intended to explore how computer vision can be used to detect, track, and analyze entities over time while preserving structured evidence that can later be reviewed and compared.

---

# Why This Project Fits the Learning Profile

The proposal aligns skills in:

- Python
- Computer vision
- Databases
- Programming

with the objective of processing recorded videos and structuring the resulting evidence.

---

# Project Objective

Build an application to detect, track, and analyze the position and movement of entities in video sequences while exporting structured evidence.

---

# Learning and Evidence

## Learning Objectives

The proposal identifies the following learning objectives:

- Failure documentation
- Model evaluation
- Temporal identity
- IoU
- Dice coefficient
- Post-processing with Supervision
- Precision
- Recall
- Tracking
- Data visualization

---

# Evidence of Mastery

The project should produce:

- Python source code for the processing and tracking pipeline
- Analysis tool and interface for querying results
- Performance evaluation report under uncontrolled environmental conditions

---

# Portfolio Evidence

The proposed portfolio evidence includes:

- Documented test cases
- Results explorer
- Confusion matrix
- Metrics report

---

# MVP Contract

## Included in the MVP

The MVP should:

- Load an image or recorded video and preserve its date and source
- Store identifier, timestamp, media, result, confidence, and notes
- Query previous observations
- Compare sessions of the same unit
- Display evolution and evidence in a digital application
- Provide an analysis tool
- Provide a tracking dashboard
- Document errors caused by lighting
- Document errors caused by scale
- Document errors caused by occlusion
- Document errors involving out-of-sample data

---

# Not Included Initially

The initial MVP does not include:

- Threshold-based alerts before metrics and false positives have been validated
- Live capture as part of the initial MVP

Live capture may be evaluated later as an extension.

---

# Definition of Done

The project is considered complete when it demonstrates:

- The ability to process recorded videos
- The ability to register tracking history in the database
- Generation of exportable reports containing performance metrics for tracked objects

---

# Development Path

## 01 — Prepare

### Verify Project Feasibility

Begin with a technical test using:

- 20–50 images

or

- 2–5 short videos

### MVP Validation

Use approximately:

- 100–200 observations under varied conditions

### Custom Training

Custom training is:

**Not required for the first technical test.**

If the base model does not adequately cover the task, this limitation should be documented.

### Expected Result

A reproducible technical test.

---

# 02 — Build

## Implement the End-to-End Workflow

The system should:

1. Load an image or recorded video and preserve its date and source.
2. Store identifier, timestamp, media, result, confidence, and notes.
3. Query previous observations.
4. Compare sessions of the same unit.
5. Display evolution and evidence in a digital application.
6. Provide an analysis tool.
7. Provide a tracking dashboard.
8. Document errors involving lighting, scale, occlusion, and out-of-sample data.

### Expected Result

The MVP produces and preserves evidence.

---

# 03 — Validate

## Demonstrate Where the System Works and Where It Fails

### MVP Validation

Evaluate:

- False positives
- Omissions
- Stability under varied conditions

Recommended validation size:

**100–200 observations under varied conditions.**

### Custom Training

Train or adjust categories and regions not adequately covered by the base model only if necessary.

Custom training is:

**Not required for the first technical test.**

If the base model does not cover the task adequately, this limitation should be documented.

### Technical Test

Verify that the following work correctly:

- Capture
- Inference
- Storage

Recommended technical test:

- 20–50 images

or

- 2–5 short videos

### Expected Result

Documented success criteria and documented errors.

---

# 04 — Present

## Convert the Work into Professional Evidence

The final project presentation should include:

- Python source code for the processing and tracking pipeline
- Analysis tool and interface for querying results
- Performance evaluation report under uncontrolled environmental conditions

### Expected Result

A clear, understandable, and reproducible demonstration.

---

# Resources, Requirements, and Tools

The proposal identifies the following core technologies:

- Supervision
- YOLO
- Ultralytics
- SAM 3

---

# Responsible Use

The project should be developed within the limits and responsible-use considerations defined by the learning program, particularly when computer vision is used to observe or track people or other entities.

---

# Possible Future Development

After completing and validating the MVP, future development may include:

- Threshold-based alerts after validating metrics and false positives
- Live capture as a later extension

These capabilities are not part of the initial MVP.

---

# Program Context

This project proposal was generated as part of the **SAM3 Computer Vision Learning Program** involving:

- Meta
- CENTRO
- INFOTEC

---

## Repository

This project is part of the:

[SAM3 Learning Journey](../../../README.md)

Main project documentation:

[Visual Tracking and Analysis System](../README.md)

---

## Author

**Peyman Miyandashti**

GitHub: [Peyman-mxli](https://github.com/Peyman-mxli)

LinkedIn: [Peyman Miyandashti](https://www.linkedin.com/in/peyman-mxli/)
