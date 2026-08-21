# Notebooks

This directory contains Google Colab and Jupyter Notebook workflows used to develop, test, validate, and document the **Visual Tracking and Analysis System**.

The notebooks are intended to preserve the complete experimental workflow used during development.

---

## Purpose

Google Colab is used as the main execution and testing environment for this project.

The notebooks may include:

- Repository cloning
- Dependency installation
- Environment verification
- GPU checks
- Library imports
- Model loading
- Input preparation
- Object detection tests
- Segmentation tests
- Object tracking tests
- Database tests
- Metrics evaluation
- Visualization tests
- Full video processing
- Output generation
- Debugging
- Validation

---

## Important Project Rule

Important code used in Google Colab should not remain only inside the notebook.

Reusable project logic should also be stored in the appropriate Python modules inside:

```text
../src/
```

The notebook is used for:

- Execution
- Experimentation
- Integration testing
- Demonstration
- Documentation

The `src/` directory remains the main location for reusable application code.

---

## Planned Notebook

The main development notebook will be:

```text
Visual-Tracking-and-Analysis-Colab.ipynb
```

It will document the complete Colab workflow used to build and test Project 06.

---

## Planned Workflow

The notebook will be developed incrementally.

### 01 — Clone Repository

Clone the SAM3 Learning Journey repository into Google Colab.

### 02 — Enter Project Directory

Navigate to:

```text
05-projects/06-Visual-Tracking-and-Analysis-System/
```

### 03 — Install Dependencies

Install the dependencies defined in:

```text
../requirements.txt
```

### 04 — Environment Verification

Verify imports for:

- Python
- OpenCV
- NumPy
- Pandas
- Ultralytics
- Supervision

GPU availability may also be checked when required.

### 05 — Test Object Detection

Test:

```text
../src/detector.py
```

with sample media.

### 06 — Test Object Tracking

Test:

```text
../src/tracker.py
```

using ByteTrack.

### 07 — Test Visualization

Test:

```text
../src/visualization.py
```

and verify annotated output.

### 08 — Test Database

Test:

```text
../src/database.py
```

and confirm that sessions and observations can be stored successfully.

### 09 — Test Metrics

Test:

```text
../src/metrics.py
```

with controlled examples.

### 10 — Segmentation Integration

Configure and validate the SAM3 segmentation workflow before replacing the current placeholder implementation in:

```text
../src/segmenter.py
```

### 11 — Full Pipeline

Combine:

```text
Detection
    ↓
Segmentation
    ↓
Tracking
    ↓
Data Storage
    ↓
Metrics
    ↓
Visualization
```

### 12 — Output Validation

Verify generated results inside:

```text
../assets/output/
```

### 13 — Document Results

Record:

- Successful tests
- Errors
- Fixes
- Model limitations
- Failure cases
- Performance observations

---

## Notebook and Source-Code Synchronization

Whenever important code is introduced or changed in Colab:

1. Test it in Colab.
2. Confirm that it works.
3. Add or update the corresponding GitHub source file.
4. Document the change when necessary.
5. Continue to the next experiment.

This helps prevent the Colab notebook and GitHub repository from becoming inconsistent.

---

## Reproducibility

The final notebook should allow another user to understand and reproduce the development workflow.

It should clearly show:

- Required dependencies
- Required input files
- Commands executed
- Expected outputs
- Errors encountered
- Solutions applied
- Final results

---

## Project

This directory belongs to:

[Visual Tracking and Analysis System](../README.md)

Part of the [SAM3 Learning Journey](../../../README.md).

---

## Author

**Peyman Miyandashti**

GitHub: [Peyman-mxli](https://github.com/Peyman-mxli)

LinkedIn: [Peyman Miyandashti](https://www.linkedin.com/in/peyman-mxli/)
