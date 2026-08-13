# SAM3 — Projects

This directory contains practical computer vision projects developed throughout my **SAM3 Computer Vision Learning Journey**.

The purpose of this section is to transform the concepts studied during the course into complete, reusable projects that combine AI models, computer vision libraries, visualization tools, and structured project workflows.

Unlike the smaller examples in `04-examples/`, projects in this directory combine multiple concepts into complete applications.

---

## Available Projects

### 01 — YOLO + Supervision Object Detector

[`01-YOLO-Supervision-Object-Detector/`](./01-YOLO-Supervision-Object-Detector/)

A complete object-detection pipeline using **YOLOv8**, **OpenCV**, and **Supervision**.

The project:

- Loads an input image
- Runs YOLOv8 object detection
- Converts YOLO predictions into `sv.Detections`
- Creates class and confidence labels
- Draws bounding boxes
- Adds labels to detected objects
- Saves the annotated image
- Exports predictions to JSON

### Pipeline

```text
Input Image
    ↓
OpenCV
    ↓
YOLOv8
    ↓
Predictions
    ↓
sv.Detections
    ↓
BoxAnnotator
    ↓
LabelAnnotator
    ↓
Annotated Image
    ↓
JSON Predictions
```

**Status:** Completed and tested successfully in Google Colab.

---

### 02 — Multi-Annotator Visualization Pipeline

[`02-Multi-Annotator-Visualization-Pipeline/`](./02-Multi-Annotator-Visualization-Pipeline/)

A computer vision visualization pipeline demonstrating how multiple **Supervision annotators** can be composed as visual layers.

The project:

- Loads an input image
- Runs YOLOv8 object detection
- Converts predictions into `sv.Detections`
- Generates class and confidence labels
- Applies multiple annotators
- Demonstrates layered annotation
- Saves the final visualization

The project uses:

- `BoxAnnotator`
- `EllipseAnnotator`
- `DotAnnotator`
- `LabelAnnotator`

### Pipeline

```text
Input Image
    ↓
YOLOv8
    ↓
Object Detection
    ↓
sv.Detections
    ↓
BoxAnnotator
    ↓
EllipseAnnotator
    ↓
DotAnnotator
    ↓
LabelAnnotator
    ↓
Annotated Image
```

### Test Result

The project was successfully tested in **Google Colab**.

During the test:

```text
Detected objects: 13
Annotated image saved to: output/annotated_image.jpg
```

The final visualization successfully displayed multiple annotation layers around the detected objects.

**Status:** Completed and tested successfully.

---

## Project Organization

Each project is designed to be self-contained.

A typical project may contain:

```text
project-name/
│
├── assets/
│   ├── input/
│   ├── output/
│   └── screenshots/
│
├── project_script.py
├── requirements.txt
└── README.md
```

### `assets/input/`

Contains example input images used to test the project.

### `assets/output/`

Contains generated results such as annotated images.

### `assets/screenshots/`

Contains screenshots documenting successful executions, experiments, and results.

---

## Projects vs Examples

The repository separates small examples from larger projects.

```text
04-examples/
    Small runnable demonstrations of individual concepts

05-projects/
    Complete applications combining multiple concepts
```

Examples are designed for learning individual techniques.

Projects demonstrate how those techniques can be combined into a complete computer vision workflow.

---

## Technologies

Projects in this directory may use:

- Python
- OpenCV
- NumPy
- Matplotlib
- Ultralytics YOLO
- Supervision
- JSON
- Google Colab
- Git
- GitHub

Future projects will expand this stack as the course progresses toward more advanced computer vision and **SAM3** workflows.

---

## Current Progress

| # | Project | Status |
|---|---|---|
| 01 | YOLO + Supervision Object Detector | Completed & Tested |
| 02 | Multi-Annotator Visualization Pipeline | Completed & Tested |

**Total completed projects: 2**

---

## Repository Structure

These projects are part of the larger SAM3 Learning Journey:

```text
SAM3-Learning-Journey/
│
├── 03-notebooks/
│   └── Original course notebooks
│
├── 04-examples/
│   └── Small runnable examples
│
├── 05-projects/
│   └── Complete practical projects
│
├── 08-course-notes/
│   └── Detailed concepts and class notes
│
└── 09-assets/
    └── Repository-wide banners and supporting assets
```

---

## Goal

The goal of this directory is not only to store code, but to document the progression from individual computer vision concepts to complete AI applications.

Each project provides practical experience with:

- Designing computer vision pipelines
- Working with AI model predictions
- Processing detection data
- Visualizing model results
- Organizing reusable Python applications
- Testing projects in Google Colab
- Documenting experiments and results
- Maintaining projects with Git and GitHub

More projects will be added as the **SAM3 Learning Journey** progresses.

---

## Author

**Peyman Miyandashti**

SAM3 Computer Vision Learning Journey

LinkedIn:  
https://www.linkedin.com/in/peyman-mxli/

GitHub:  
https://github.com/Peyman-mxli
