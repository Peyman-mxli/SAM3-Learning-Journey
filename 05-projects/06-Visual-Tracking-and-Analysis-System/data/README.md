# Data

This directory contains data generated and used by the **Visual Tracking and Analysis System**.

Unlike the `assets/` directory, which stores visual media, this directory is intended for structured information produced during processing and analysis.

---

## Purpose

The system will generate structured data while processing images and recorded videos.

This data may include:

- Processing sessions
- Frame numbers
- Timestamps
- Tracker IDs
- Object classes
- Confidence scores
- Bounding-box coordinates
- Segmentation information
- Position information
- Movement information
- Analysis results
- Notes about observations

---

## Database

The project uses SQLite for local historical storage.

The default database location is:

```text
data/analysis.db
