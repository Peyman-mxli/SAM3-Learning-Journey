# Screenshots

This directory contains screenshots documenting the development, experiments, practical exercises, results, and progress of the **SAM3 Learning Journey**.

The `screenshots` directory is located inside `07-assets` so that all visual documentation and repository assets remain organized in one central location.

---

## Directory Location

```text
SAM3-Learning-Journey/
│
└── 07-assets/
    │
    ├── banners/
    │
    ├── screenshots/
    │   └── README.md
    │
    └── README.md
```

---

## Purpose

The purpose of this directory is to preserve visual evidence of the learning and development process throughout the SAM3 Computer Vision course.

Screenshots may document:

- Environment configuration
- Google Colab execution
- NVIDIA GPU configuration
- Package installation
- Python environment verification
- Computer Vision experiments
- Object detection
- Detection filtering
- Image annotation
- Video processing
- Object tracking
- SAM3 segmentation
- Project execution
- Analytics results
- Course exercises
- Troubleshooting
- Successful development milestones

These screenshots complement the technical documentation contained throughout the repository.

---

## Environment Setup

Screenshots may document development environment configuration such as:

- Google Colab
- Jupyter Notebook
- Python
- NVIDIA T4 GPU
- CUDA
- Hugging Face
- Roboflow
- OpenCV
- PyTorch
- Ultralytics
- Supervision

Example filenames:

```text
google-colab-runtime.png
nvidia-t4-gpu.png
python-environment-test.png
cuda-verification.png
hugging-face-setup.png
roboflow-setup.png
```

---

## Computer Vision Experiments

Screenshots can document Computer Vision concepts and experiments performed throughout the course.

Examples include:

- YOLO object detection
- Bounding boxes
- Confidence scores
- Class labels
- Detection filtering
- Non-Maximum Suppression (NMS)
- Supervision annotations
- Video detection
- Object tracking
- ByteTrack
- Trajectory visualization
- Image segmentation
- Video segmentation
- SAM3 masks

Example filenames:

```text
yolo-object-detection.png
supervision-annotations.png
detection-filtering.png
nms-result.png
video-detection.png
bytetrack-object-tracking.png
trajectory-visualization.png
sam3-image-segmentation.png
sam3-video-segmentation.png
```

---

## Project Evidence

Screenshots may also be used as visual evidence for projects located in:

```text
05-projects/
```

For example, a project may generate:

```text
05-projects/06-Visual-Tracking-and-Analysis-System/
│
├── tracker_summary.csv
├── trajectory_summary.csv
├── trajectory_visualization.png
├── tracker_duration_chart.png
├── class_observation_chart.png
├── movement_distance_chart.png
├── confidence_chart.png
└── sam3_tracking_output_01.mp4
```

The original generated files should remain inside the project directory.

Screenshots demonstrating those results may be stored here:

```text
07-assets/screenshots/
```

This keeps project outputs and documentation evidence clearly separated.

---

## What Belongs Here

Files appropriate for this directory include screenshots showing:

- Successful notebook execution
- Terminal output
- Environment verification
- GPU information
- Model loading
- Detection results
- Segmentation results
- Tracking results
- Database results
- Analytics results
- Charts displayed during development
- GitHub progress
- Course exercises
- Important errors and their solutions
- Completed project milestones

---

## What Does Not Belong Here

Reusable repository graphics should not be stored in this directory.

For example:

```text
Main-Banner.png
SAM3.png
Python-Banner.png
Google-Colab-Banner.png
Hugging-Face-Banner.png
Roboflow-Banner.png
```

These belong in:

```text
07-assets/banners/
```

Likewise, original project-generated outputs should remain inside their respective project directories whenever possible.

---

## Naming Convention

Screenshots should use descriptive filenames that clearly explain what the image contains.

### Recommended

```text
google-colab-t4-gpu.png
python-environment-success.png
yolo-detection-result.png
supervision-box-annotation.png
detection-filtering-result.png
bytetrack-video-tracking.png
sam3-segmentation-result.png
sqlite-tracking-database.png
trajectory-analysis-result.png
```

### Avoid

```text
Screenshot1.png
Screenshot2.png
image.png
capture.png
test.png
new.png
final.png
final-final.png
```

Descriptive filenames make screenshots much easier to locate and reference from documentation.

---

## Recommended Naming Format

When useful, screenshots can follow this format:

```text
<topic>-<description>.png
```

Examples:

```text
colab-gpu-configuration.png
yolo-person-detection.png
bytetrack-tracker-ids.png
sam3-video-mask.png
sqlite-observations-table.png
analytics-confidence-chart.png
```

For multiple screenshots covering the same process:

```text
sam3-video-setup-01.png
sam3-video-setup-02.png
sam3-video-result-01.png
```

---

## Using Screenshots in Documentation

From the repository root:

```markdown
![SAM3 Segmentation](07-assets/screenshots/sam3-segmentation-result.png)
```

From a first-level directory:

```markdown
![SAM3 Segmentation](../07-assets/screenshots/sam3-segmentation-result.png)
```

From a second-level directory:

```markdown
![SAM3 Segmentation](../../07-assets/screenshots/sam3-segmentation-result.png)
```

HTML can also be used for better control over image size and alignment:

```html
<p align="center">
  <img
    src="../../07-assets/screenshots/sam3-segmentation-result.png"
    alt="SAM3 Segmentation Result"
    width="850"
  >
</p>
```

Always verify the relative path from the Markdown file where the screenshot is referenced.

---

## Screenshot Documentation

When a screenshot is included in a README or technical document, it should ideally include a short explanation.

For example:

```markdown
### NVIDIA T4 GPU Verification

The following screenshot confirms that the Google Colab runtime successfully detected an NVIDIA T4 GPU.

![NVIDIA T4 GPU](../../07-assets/screenshots/nvidia-t4-gpu.png)
```

This provides context instead of displaying an unexplained image.

---

## Organization Principle

The repository separates visual resources according to their purpose:

```text
Reusable repository graphics
        ↓
07-assets/banners/

Development and learning screenshots
        ↓
07-assets/screenshots/

Original project outputs
        ↓
05-projects/<project-name>/
```

This structure keeps the repository clean and prevents unnecessary duplication.

---

## Documentation Guidelines

When adding screenshots:

1. Use a descriptive filename.
2. Store the screenshot inside `07-assets/screenshots/`.
3. Avoid unnecessary duplicate screenshots.
4. Keep original project outputs inside their project directories.
5. Reference screenshots using relative paths.
6. Add descriptive `alt` text when embedding images.
7. Explain important screenshots in the related documentation.
8. Keep screenshots relevant to the SAM3 Learning Journey.
9. Prefer `.png` for interface screenshots and technical evidence.
10. Update documentation when screenshots represent important new milestones.

---

## Relationship to Other Directories

```text
07-assets/
│
├── banners/
│   └── Reusable repository graphics
│
└── screenshots/
    └── Visual development and learning evidence
```

The screenshots stored here may support documentation located in:

```text
01-setup/
02-docs/
03-notebooks/
04-examples/
05-projects/
06-resources/
08-course-notes/
```

---

## Repository Goal

The screenshots in this directory create a visual history of the **SAM3 Learning Journey**.

Together with the source code, notebooks, course notes, projects, reports, and documentation, they demonstrate the progression from basic Computer Vision concepts toward more advanced workflows involving:

- Object Detection
- Supervision
- Detection Filtering
- Video Processing
- Object Tracking
- ByteTrack
- SAM3
- Image Segmentation
- Video Segmentation
- Analytics
- Data Persistence
- Computer Vision Pipelines

---

## Author

**Peyman Miyandashti**

**Information Technology Engineering & Digital Innovation Student**  
**Universidad Politécnica de Baja California (UPBC)**

**Specializations:** Artificial Intelligence • Computer Vision • Machine Learning • Deep Learning • Python • Software Engineering • Cybersecurity • Data Science

[GitHub — Peyman-mxli](https://github.com/Peyman-mxli)

[LinkedIn — Peyman Miyandashti](https://www.linkedin.com/in/Peyman-mxli/)

---

**SAM3 Learning Journey — 2026**
