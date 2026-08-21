# Visual Tracking and Analysis System

An integrated computer vision project combining object detection, multi-object tracking, segmentation, visualization, evaluation, and structured data storage.

This project was developed as part of my **SAM3 Computer Vision Learning Journey** and demonstrates how multiple computer vision components can be combined into a modular analysis pipeline.

The current image-based MVP integrates:

- Ultralytics YOLO object detection
- Supervision detections
- ByteTrack multi-object tracking
- Meta SAM 3 text-prompt segmentation
- SAM 3 segmentation masks
- Bounding-box visualization
- Persistent tracker IDs
- Confidence labels
- Object trajectories
- SQLite-based structured storage
- Evaluation metrics
- Google Colab GPU execution

The integrated image pipeline has been successfully tested in **Google Colab using an NVIDIA Tesla T4 GPU**.

---

## Project Status

**Status: Image-Based Integrated MVP Completed**

The core image-processing pipeline is operational and has been tested end-to-end.

Current verified capabilities:

| Component | Status |
|---|---|
| YOLO Object Detection | Completed |
| Supervision Integration | Completed |
| ByteTrack Object Tracking | Completed |
| Persistent Tracker IDs | Completed |
| SAM 3 Integration | Completed |
| SAM 3 Text-Prompt Segmentation | Completed |
| SAM 3 Mask Generation | Completed |
| Combined Visualization | Completed |
| SQLite Database Module | Completed |
| Evaluation Metrics Module | Completed |
| Google Colab Workflow | Completed |
| Final Integrated Image Test | Completed |
| Recorded Video Pipeline | Planned |
| Full Tracking Analytics | Planned |

---

## System Architecture

The project follows a modular architecture in which each major computer vision task is implemented independently.

```text
Input Image / Video
        |
        v
+-----------------------+
|   YOLO Detection      |
|   ObjectDetector      |
+-----------------------+
        |
        v
+-----------------------+
|   ByteTrack           |
|   ObjectTracker       |
+-----------------------+
        |
        v
+-----------------------+
|   Meta SAM 3          |
|   ObjectSegmenter     |
+-----------------------+
        |
        v
+-----------------------+
| Combined Visualization|
| TrackingVisualizer    |
+-----------------------+
        |
        +----------------------+
        |                      |
        v                      v
+----------------+     +----------------+
| Evaluation     |     | SQLite Storage |
| Metrics        |     | Database       |
+----------------+     +----------------+
        |
        v
Final Analysis Output
```

---

## Integrated Image Pipeline

The current working image pipeline follows this sequence:

```text
Input Image
    |
    v
YOLO Object Detection
    |
    v
Supervision Detections
    |
    v
ByteTrack Object Tracking
    |
    v
SAM 3 Text-Prompt Segmentation
    |
    v
SAM 3 Segmentation Masks
    |
    v
Combined Visualization
    |
    v
Final Annotated Image
```

The final visualization can contain:

- YOLO bounding boxes
- detected object classes
- confidence scores
- ByteTrack tracker IDs
- SAM 3 segmentation masks
- tracking visualization information

---

## Technologies

### Computer Vision

- Python
- OpenCV
- Ultralytics YOLO
- Supervision
- ByteTrack
- Meta SAM 3

### Machine Learning

- PyTorch
- TorchVision
- Hugging Face Hub

### Data and Analysis

- NumPy
- Pandas
- SQLite
- Matplotlib

### Development Environment

- Google Colab
- NVIDIA Tesla T4 GPU
- Git
- GitHub

---

## Project Structure

```text
06-Visual-Tracking-and-Analysis-System/
│
├── README.md
├── requirements.txt
│
├── assets/
│   ├── README.md
│   │
│   ├── input/
│   │   ├── README.md
│   │   └── yolo_bus_test.jpg
│   │
│   └── output/
│       ├── README.md
│       └── final_integrated_pipeline.jpg
│
├── notebooks/
│   ├── README.md
│   └── COLAB-WORKFLOW.md
│
└── src/
    ├── database.py
    ├── detector.py
    ├── metrics.py
    ├── pipeline.py
    ├── segmenter.py
    ├── tracker.py
    └── visualization.py
```

---

## Core Modules

### `detector.py`

Implements object detection using Ultralytics YOLO.

Responsibilities include:

- loading the YOLO model
- processing input images
- applying confidence thresholds
- converting YOLO predictions into `supervision.Detections`
- providing class-name information to other modules

---

### `tracker.py`

Implements multi-object tracking using ByteTrack.

The tracker receives detections from YOLO and assigns persistent tracker IDs to detected objects.

This makes it possible to maintain object identity across sequential frames when the project is extended to video processing.

---

### `segmenter.py`

Provides the integration layer between the project and the official Meta SAM 3 implementation.

The module:

- loads the SAM 3 image model
- loads the SAM 3 checkpoint
- creates a `Sam3Processor`
- accepts PIL images
- performs text-prompt segmentation
- returns SAM 3 segmentation results

Example text prompt:

```python
prompt="person"
```

SAM 3 can then identify and segment matching objects in the image.

---

### `visualization.py`

Handles the visual representation of detection, tracking, and segmentation results.

The visualizer supports:

- bounding boxes
- class labels
- confidence scores
- tracker IDs
- object traces
- SAM 3 segmentation-mask overlays

The segmentation masks are applied before the tracking annotations so that bounding boxes and labels remain visible in the final output.

---

### `pipeline.py`

Acts as the central integration layer.

The `VisualAnalysisPipeline` coordinates:

1. YOLO detection
2. ByteTrack tracking
3. SAM 3 segmentation
4. combined visualization

Example:

```python
final_result = pipeline.process_image(
    image_bgr=image,
    segmentation_prompt="person"
)
```

The returned result contains:

```python
{
    "detections": detections,
    "tracked_detections": tracked_detections,
    "segmentation": segmentation_output,
    "annotated_image": annotated_image
}
```

This provides both machine-readable results and the final visualization.

---

### `database.py`

Provides structured storage capabilities for analysis results.

SQLite is used as a lightweight local database that can later store information such as:

- tracker IDs
- object classes
- confidence scores
- timestamps
- frame numbers
- detection information
- tracking statistics

This module prepares the project for more advanced video analytics.

---

### `metrics.py`

Provides evaluation utilities for measuring system performance.

The metrics layer is designed to support evaluation of:

- detections
- tracking results
- segmentation results
- object counts
- future video-processing performance

---

## SAM 3 Integration

Meta SAM 3 is used as the segmentation component of the system.

SAM 3 provides promptable segmentation capabilities and can identify objects using text prompts.

For example:

```text
person
```

can be used to request segmentation of people in an image.

The system converts the OpenCV BGR image into RGB format and then creates a PIL image before sending it to SAM 3.

```python
image_rgb = image_bgr[:, :, ::-1]

pil_image = Image.fromarray(
    image_rgb
)
```

The segmentation result is then passed into the visualization module.

---

## Hugging Face Authentication

The official SAM 3 checkpoint is distributed through a gated Hugging Face repository.

Access therefore requires:

1. a Hugging Face account
2. approved access to the SAM 3 repository
3. a Hugging Face access token
4. authentication inside the execution environment

For Google Colab, the token can be stored securely using **Colab Secrets**.

The project uses:

```text
HF_TOKEN
```

as the secret name.

Tokens should never be committed directly to GitHub.

---

## SAM 3 Checkpoint

The SAM 3 checkpoint is downloaded from the official gated model repository and cached by Hugging Face.

The checkpoint file used during development was approximately:

```text
3.21 GB
```

Because model checkpoints are large and access-controlled, they are **not stored in this GitHub repository**.

The checkpoint must be downloaded separately by an authorized user.

---

## GPU Environment

The integrated pipeline was tested in Google Colab with:

```text
GPU: NVIDIA Tesla T4
CUDA: Available
```

SAM 3 successfully loaded onto the GPU.

Observed SAM 3 GPU memory usage after model loading was approximately:

```text
Allocated: ~3.33 GB
Reserved:  ~3.43 GB
```

The Tesla T4 does not support the Ampere Flash Attention path used by newer NVIDIA architectures.

The SAM 3 implementation therefore reports a warning indicating that Flash Attention is disabled.

This warning does not prevent the model from running successfully.

---

## Dependency Compatibility

During development, a binary compatibility issue occurred between NumPy and compiled Python packages.

The environment was stabilized using:

```text
NumPy 1.26.4
```

After restarting the Colab runtime, the environment successfully loaded:

- NumPy
- PyTorch
- Ultralytics
- Supervision
- SAM 3

This highlights an important machine-learning engineering lesson:

> Installing model repositories can modify shared dependencies, so dependency compatibility should always be verified after installation.

---

## Verified Integrated Test

The final image-based pipeline was tested using a street image containing a bus and several people.

The integrated test produced:

```text
YOLO detections: 4
Tracked objects: 4
SAM 3 masks: 4

UPDATED PIPELINE END-TO-END: SUCCESS
```

This confirms that the complete pipeline can execute:

```text
YOLO
  +
ByteTrack
  +
SAM 3
  +
Supervision
```

inside one integrated workflow.

---

## Final Integrated Output

The final evidence image is stored at:

```text
assets/output/final_integrated_pipeline.jpg
```

It demonstrates the combined output of the system, including:

- YOLO object detection
- object bounding boxes
- detected classes
- confidence scores
- ByteTrack tracker IDs
- SAM 3 segmentation masks
- combined visualization

![Final Integrated Pipeline](./assets/output/final_integrated_pipeline.jpg)

---

## Google Colab Workflow

The complete development and testing process is documented in:

[`notebooks/COLAB-WORKFLOW.md`](./notebooks/COLAB-WORKFLOW.md)

The workflow covers:

- GPU verification
- repository setup
- dependency installation
- Hugging Face authentication
- gated SAM 3 access
- checkpoint download
- SAM 3 model loading
- dependency troubleshooting
- YOLO integration
- ByteTrack integration
- segmentation testing
- combined visualization testing
- final pipeline validation

---

## Installation

Install the standard Python dependencies with:

```bash
pip install -r requirements.txt
```

The official Meta SAM 3 repository must be installed separately.

Example development workflow:

```bash
git clone https://github.com/facebookresearch/sam3.git /content/sam3_repo

pip install -e /content/sam3_repo
```

The SAM 3 checkpoint must also be downloaded separately using an authorized Hugging Face account.

---

## Current Limitations

The current implementation represents the first integrated MVP.

Current limitations include:

- primary validation is image-based
- ByteTrack has been integrated but long-duration tracking still requires video testing
- SAM 3 segmentation currently runs independently from YOLO class selection
- segmentation masks and YOLO detections are not yet geometrically matched object-by-object
- database integration is modular but full frame-by-frame persistence is still planned
- advanced tracking analytics are not yet implemented
- real-time processing performance has not yet been benchmarked

These limitations define the next development phase.

---

## Next Development Phase

The next major milestone is **recorded-video processing**.

The planned pipeline is:

```text
Video
  |
  v
Frame Extraction
  |
  v
YOLO Detection
  |
  v
ByteTrack Tracking
  |
  v
SAM 3 Segmentation
  |
  v
Combined Visualization
  |
  v
Tracking Metrics
  |
  v
SQLite Storage
  |
  v
Annotated Output Video
```

Future improvements include:

- full recorded-video processing
- persistent tracking across hundreds of frames
- SAM 3 segmentation during video analysis
- trajectory visualization
- per-object statistics
- object appearance duration
- object counting
- class-based analytics
- database persistence
- tracking performance evaluation
- segmentation evaluation
- annotated video export
- performance benchmarking

---

## Learning Outcomes

This project demonstrates several important computer vision engineering concepts.

### Modular Architecture

Detection, tracking, segmentation, visualization, metrics, and storage are implemented as separate modules.

### Model Integration

YOLO and SAM 3 perform different tasks but can operate together inside one pipeline.

### Detection vs. Segmentation

YOLO identifies objects primarily using bounding boxes.

SAM 3 provides pixel-level object masks.

Combining them provides richer visual information.

### Tracking

ByteTrack introduces persistent object identities, creating the foundation for temporal video analysis.

### Environment Management

Large machine-learning systems frequently introduce dependency conflicts.

Managing package versions and runtime state is an important part of practical AI development.

### GPU Computing

Large vision foundation models such as SAM 3 benefit significantly from GPU acceleration.

### Reproducibility

Documenting the Colab workflow, dependencies, model access process, and test outputs makes the project easier to reproduce and extend.

---

## Project Goal

The long-term goal of this project is to create a reusable visual-analysis system capable of:

```text
Detecting
    +
Tracking
    +
Segmenting
    +
Visualizing
    +
Measuring
    +
Storing
```

objects across images and videos.

The current image-based MVP establishes the foundation for that larger system.

---

## Author

**Peyman Miyandashti**

Computer Vision, Artificial Intelligence, and Software Development

[GitHub](https://github.com/Peyman-mxli)

[LinkedIn](https://www.linkedin.com/in/peyman-mxli/)
