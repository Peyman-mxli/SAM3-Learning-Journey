# Assets

This directory contains the visual resources used throughout the **SAM3 Learning Journey** repository.

The purpose of `07-assets` is to keep repository images organized in one central location instead of storing them across multiple unrelated folders.

It includes reusable repository graphics as well as visual documentation generated during the learning process.

---

## Directory Structure

```text
07-assets/
│
├── banners/
│   └── Repository and technology banners
│
├── screenshots/
│   └── Screenshots documenting setup, experiments, and results
│
└── README.md
```

More asset categories can be added later as the repository continues to grow.

---

## Purpose

The `07-assets` directory is used to organize visual content such as:

- Repository banners
- Technology banners
- Documentation images
- Screenshots
- Setup evidence
- Experiment results
- Computer Vision outputs
- Course progress images
- Visual project references
- Supporting graphics used in Markdown documentation

Keeping these resources together makes the repository easier to maintain and keeps the main project folders focused on code, notebooks, reports, and technical documentation.

---

## Banners

The [`banners/`](./banners/) directory contains reusable visual banners used across the repository.

Examples include:

```text
Main-Banner.png
SAM3.png
Python-Banner.png
Google-Colab-Banner.png
Hugging-Face-Banner.png
Roboflow-Banner.png
```

These banners can be used at the top of README files, setup guides, course notes, examples, and projects.

Example:

```html
<p align="center">
  <img src="banners/Main-Banner.png"
       alt="SAM3 Learning Journey Banner"
       width="100%">
</p>
```

---

## Screenshots

The [`screenshots/`](./screenshots/) directory contains visual evidence documenting the learning and development process.

Examples include:

- Google Colab configuration
- NVIDIA T4 GPU activation
- Package installation
- Python environment testing
- YOLO detections
- Supervision annotations
- Detection filtering
- Non-Maximum Suppression (NMS)
- Object tracking
- ByteTrack results
- SAM3 segmentation
- Video segmentation
- Analytics results
- SQLite persistence
- Charts and visualizations
- Successful project execution
- Course exercises and milestones

The `screenshots/` directory has its own `README.md` for documenting screenshot organization and naming conventions.

---

## Assets vs. Project Outputs

The `07-assets` directory is intended for **visual documentation and reusable graphics**.

Actual generated project files should normally remain inside their corresponding project directories.

For example:

```text
05-projects/06-Visual-Tracking-and-Analysis-System/
```

may contain generated files such as:

```text
tracker_summary.csv
trajectory_summary.csv
trajectory_visualization.png
tracker_duration_chart.png
class_observation_chart.png
movement_distance_chart.png
confidence_chart.png
sam3_tracking_output_01.mp4
```

Those files are project outputs and should remain with the project.

Screenshots showing or documenting those results can instead be stored in:

```text
07-assets/screenshots/
```

---

## Naming Convention

Visual assets should use clear and descriptive filenames.

Recommended examples:

```text
Main-Banner.png
SAM3.png
Python-Banner.png
google-colab-t4-gpu.png
yolo-detection-result.png
sam3-video-segmentation.png
bytetrack-tracking-result.png
sqlite-persistence-result.png
trajectory-analysis.png
```

Avoid generic filenames such as:

```text
image.png
screenshot1.png
capture.png
final.png
test.png
```

Descriptive filenames make resources easier to locate, reference, and maintain.

---

## Using Assets in Markdown

From the repository root:

```markdown
![SAM3 Banner](07-assets/banners/SAM3.png)
```

Using HTML:

```html
<p align="center">
  <img src="07-assets/banners/SAM3.png"
       alt="SAM3 Banner"
       width="100%">
</p>
```

From a directory one level below the repository root:

```markdown
![SAM3 Banner](../07-assets/banners/SAM3.png)
```

From a directory two levels below:

```markdown
![SAM3 Banner](../../07-assets/banners/SAM3.png)
```

Always verify the relative path based on the location of the Markdown file.

---

## Repository Organization

The updated repository structure uses `07-assets` as the central location for visual resources.

```text
SAM3-Learning-Journey/
│
├── 01-setup/
├── 02-docs/
├── 03-notebooks/
├── 04-examples/
├── 05-projects/
├── 06-resources/
│
├── 07-assets/
│   ├── banners/
│   ├── screenshots/
│   └── README.md
│
└── 08-course-notes/
```

This structure keeps all repository visual resources under a single organized assets directory.

---

## Organization Principle

The repository follows this organization:

```text
Reusable visual resources
        ↓
07-assets/banners/

Visual learning evidence
        ↓
07-assets/screenshots/

Project-generated outputs
        ↓
05-projects/<project-name>/
```

This prevents duplicated files and keeps each type of resource in the correct location.

---

## Future Expansion

Additional directories may be created inside `07-assets` when needed.

For example:

```text
07-assets/
│
├── banners/
├── screenshots/
├── diagrams/
├── icons/
├── thumbnails/
└── README.md
```

New directories should only be added when they serve a clear documentation purpose.

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
