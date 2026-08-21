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
