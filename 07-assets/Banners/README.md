# Banners

This directory contains the visual banners used throughout the **SAM3 Learning Journey** repository.

The `banners` directory is located inside `07-assets` and provides a centralized location for reusable repository graphics used in README files, setup guides, course notes, examples, projects, and technical documentation.

---

## Directory Location

```text
SAM3-Learning-Journey/
│
└── 07-assets/
    │
    ├── banners/
    │   └── README.md
    │
    ├── screenshots/
    │   └── README.md
    │
    └── README.md
```

---

## Purpose

The purpose of this directory is to maintain a consistent visual identity throughout the repository.

Banners can be used to:

- Introduce the main repository.
- Identify specific technologies.
- Improve README presentation.
- Separate major documentation sections visually.
- Provide consistent branding across course notes.
- Improve the presentation of projects and examples.
- Make technical documentation easier to recognize and navigate.

---

## Available Banners

The banner collection may include:

```text
Main-Banner.png
SAM3.png
Python-Banner.png
Google-Colab-Banner.png
Hugging-Face-Banner.png
Roboflow-Banner.png
```

Each banner has a specific purpose within the repository.

---

## Main Repository Banner

### `Main-Banner.png`

The primary visual banner for the entire **SAM3 Learning Journey** repository.

It is intended to appear at the top of the main repository `README.md`.

Example:

```html
<p align="center">
  <img
    src="07-assets/banners/Main-Banner.png"
    alt="SAM3 Learning Journey Banner"
    width="100%"
  >
</p>
```

The main banner represents the overall focus of the repository:

- SAM3
- Computer Vision
- Artificial Intelligence
- Image Segmentation
- Video Segmentation
- Object Detection
- Object Tracking
- Python
- Deep Learning

---

## SAM3 Banner

### `SAM3.png`

Used for documentation directly related to **Segment Anything Model 3 (SAM3)**.

Typical uses include:

- SAM3 concepts
- Image segmentation
- Video segmentation
- Promptable segmentation
- Point prompts
- Mask generation
- SAM3 experiments
- SAM3 projects
- Model inference

Example:

```html
<p align="center">
  <img
    src="../../07-assets/banners/SAM3.png"
    alt="SAM3 Banner"
    width="100%"
  >
</p>
```

---

## Python Banner

### `Python-Banner.png`

Used for Python-related documentation.

Typical uses include:

- Python setup
- Python fundamentals
- Computer Vision scripts
- Utility scripts
- AI pipelines
- Model inference
- Data processing
- Automation

Example:

```html
<p align="center">
  <img
    src="../07-assets/banners/Python-Banner.png"
    alt="Python Banner"
    width="100%"
  >
</p>
```

---

## Google Colab Banner

### `Google-Colab-Banner.png`

Used for documentation related to **Google Colab**.

Typical uses include:

- Colab setup
- Runtime configuration
- GPU activation
- NVIDIA T4 GPU
- Notebook execution
- Package installation
- Cloud-based AI experiments

Example:

```html
<p align="center">
  <img
    src="../07-assets/banners/Google-Colab-Banner.png"
    alt="Google Colab Banner"
    width="100%"
  >
</p>
```

---

## Hugging Face Banner

### `Hugging-Face-Banner.png`

Used for documentation related to **Hugging Face**.

Typical uses include:

- Hugging Face account setup
- Authentication
- Model repositories
- Model downloads
- Pretrained models
- Model configuration
- AI model management

Example:

```html
<p align="center">
  <img
    src="../07-assets/banners/Hugging-Face-Banner.png"
    alt="Hugging Face Banner"
    width="100%"
  >
</p>
```

---

## Roboflow Banner

### `Roboflow-Banner.png`

Used for documentation involving **Roboflow** and related Computer Vision workflows.

Typical uses include:

- Dataset management
- Dataset annotation
- Dataset preprocessing
- Computer Vision datasets
- Supervision
- Inference
- Detection visualization
- Model workflows

Example:

```html
<p align="center">
  <img
    src="../07-assets/banners/Roboflow-Banner.png"
    alt="Roboflow Banner"
    width="100%"
  >
</p>
```

---

## Using Banners

Banner paths depend on the location of the Markdown document referencing them.

### From the Repository Root

```html
<p align="center">
  <img
    src="07-assets/banners/Main-Banner.png"
    alt="SAM3 Learning Journey Banner"
    width="100%"
  >
</p>
```

### From a First-Level Directory

```html
<p align="center">
  <img
    src="../07-assets/banners/SAM3.png"
    alt="SAM3 Banner"
    width="100%"
  >
</p>
```

### From a Second-Level Directory

```html
<p align="center">
  <img
    src="../../07-assets/banners/SAM3.png"
    alt="SAM3 Banner"
    width="100%"
  >
</p>
```

### From a Third-Level Directory

```html
<p align="center">
  <img
    src="../../../07-assets/banners/SAM3.png"
    alt="SAM3 Banner"
    width="100%"
  >
</p>
```

Always verify the relative path based on the location of the README or Markdown document.

---

## Banner Naming Convention

Banner filenames should remain descriptive and consistent.

### Recommended

```text
Main-Banner.png
Python-Banner.png
Google-Colab-Banner.png
Hugging-Face-Banner.png
Roboflow-Banner.png
SAM3.png
```

### Avoid

```text
banner1.png
banner2.png
image.png
new-banner.png
final-banner.png
final-final-banner.png
```

Clear filenames make assets easier to identify and maintain.

---

## Adding New Banners

When creating a new banner:

1. Determine whether the banner will be reused across the repository.
2. Give it a clear and descriptive filename.
3. Store it inside `07-assets/banners/`.
4. Use the correct relative path when referencing it.
5. Add meaningful `alt` text.
6. Maintain a consistent visual style with the existing repository banners.
7. Update this README when adding an important new banner.

---

## Recommended Banner Format

For GitHub README files, banners should preferably:

- Use a wide horizontal layout.
- Have readable text.
- Maintain good contrast.
- Avoid excessive small details.
- Work well on desktop and mobile displays.
- Use high-quality PNG images.
- Maintain consistent repository branding.

A typical implementation is:

```html
<p align="center">
  <img
    src="PATH-TO-BANNER.png"
    alt="Descriptive Banner Name"
    width="100%"
  >
</p>
```

---

## Banners vs. Screenshots

The repository separates reusable graphics from development evidence.

### Banners

Stored in:

```text
07-assets/banners/
```

Used for:

```text
Repository branding
Technology branding
README headers
Course documentation
Project documentation
Reusable visual presentation
```

### Screenshots

Stored in:

```text
07-assets/screenshots/
```

Used for:

```text
Development evidence
Google Colab results
GPU configuration
Model execution
Detection results
Tracking results
Segmentation results
Analytics results
Course progress
Troubleshooting evidence
```

---

## Asset Organization

The overall visual asset structure is:

```text
07-assets/
│
├── banners/
│   ├── Main-Banner.png
│   ├── SAM3.png
│   ├── Python-Banner.png
│   ├── Google-Colab-Banner.png
│   ├── Hugging-Face-Banner.png
│   ├── Roboflow-Banner.png
│   └── README.md
│
├── screenshots/
│   └── README.md
│
└── README.md
```

This keeps permanent visual resources and development evidence clearly organized while maintaining a single central assets directory.

---

## Repository Branding

The banners help create a consistent identity for documentation covering technologies and concepts such as:

- Python
- Google Colab
- Jupyter Notebook
- NVIDIA T4 GPU
- Hugging Face
- Roboflow
- OpenCV
- PyTorch
- YOLO
- Supervision
- ByteTrack
- Object Detection
- Object Tracking
- Image Segmentation
- Video Segmentation
- Segment Anything Model 3
- Computer Vision
- Artificial Intelligence
- Deep Learning

As the learning journey continues, additional banners can be added when new technologies or major course topics require dedicated visual branding.

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
