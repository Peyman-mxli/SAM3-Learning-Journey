# Notebooks

This directory contains the Google Colab and Jupyter Notebook workflows used to develop, test, validate, troubleshoot, and document the **Visual Tracking and Analysis System**.

Google Colab is the primary execution environment used for the current integrated MVP because the project requires GPU acceleration for Meta SAM 3.

The current workflow has successfully validated the integration of:

- Ultralytics YOLO
- Supervision
- ByteTrack
- Meta SAM 3
- PyTorch
- OpenCV
- Combined segmentation and tracking visualization

---

## Current Status

**Status: Image-Based Colab Integration Completed**

The Google Colab development workflow has successfully produced a working end-to-end image pipeline.

Verified processing sequence:

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
Final Integrated Output
```

The final validation produced:

```text
YOLO detections: 4
Tracked objects: 4
SAM 3 masks: 4

UPDATED PIPELINE END-TO-END: SUCCESS
```

---

## Purpose

Google Colab is used as the main development and testing environment for Project 06.

The Colab workflow covers:

- repository cloning
- dependency installation
- environment verification
- GPU verification
- library imports
- Hugging Face authentication
- SAM 3 gated-model access
- SAM 3 checkpoint download
- model loading
- input preparation
- YOLO object detection
- ByteTrack object tracking
- SAM 3 segmentation
- segmentation-mask visualization
- combined visualization
- dependency troubleshooting
- integration testing
- output generation
- final validation

Future Colab work will extend the system into recorded-video processing and tracking analytics.

---

## Current Documentation

The main Colab development workflow is documented in:

[`COLAB-WORKFLOW.md`](./COLAB-WORKFLOW.md)

This file preserves the important setup, installation, authentication, troubleshooting, and testing steps used during development.

---

## Important Project Rule

Important code developed or tested in Google Colab should not remain only inside a notebook.

Reusable project logic belongs in:

```text
../src/
```

The notebook environment is used for:

- execution
- experimentation
- integration testing
- debugging
- GPU testing
- demonstration
- validation
- documentation

The `src/` directory remains the primary location for reusable application code.

---

## Source Modules

The Colab workflow currently tests and integrates the following modules:

```text
../src/
├── database.py
├── detector.py
├── metrics.py
├── pipeline.py
├── segmenter.py
├── tracker.py
└── visualization.py
```

Each module has a specific responsibility inside the project architecture.

---

## `detector.py`

The detection module uses Ultralytics YOLO.

It is responsible for:

- loading the YOLO model
- processing images
- applying confidence thresholds
- generating detections
- converting results into `supervision.Detections`
- providing class-name information

The detector was successfully tested in Google Colab.

---

## `tracker.py`

The tracking module uses ByteTrack through Supervision.

It is responsible for:

- receiving detections
- assigning tracker IDs
- maintaining the foundation for persistent object identities
- preparing the project for temporal video tracking

ByteTrack successfully assigned tracker IDs during the integrated image test.

---

## `segmenter.py`

The segmentation module integrates the official Meta SAM 3 implementation.

It is responsible for:

- loading the SAM 3 model
- loading the SAM 3 checkpoint
- creating the SAM 3 processor
- accepting PIL images
- processing text prompts
- returning segmentation masks and related results

SAM 3 was successfully integrated and tested in Google Colab.

---

## `visualization.py`

The visualization module combines information from multiple components.

It supports:

- SAM 3 segmentation masks
- YOLO bounding boxes
- class labels
- confidence scores
- ByteTrack tracker IDs
- object traces

The visualization system was successfully tested with SAM 3 masks and tracking annotations displayed together.

---

## `pipeline.py`

The pipeline module coordinates the complete image workflow.

The processing order is:

```text
YOLO
  |
  v
ByteTrack
  |
  v
SAM 3
  |
  v
Combined Visualization
```

Example:

```python
final_result = pipeline.process_image(
    image_bgr=image,
    segmentation_prompt="person"
)
```

The pipeline returns:

```python
{
    "detections": detections,
    "tracked_detections": tracked_detections,
    "segmentation": segmentation_output,
    "annotated_image": annotated_image
}
```

The complete pipeline has been successfully tested end-to-end.

---

## Google Colab GPU Environment

The project was tested using:

```text
NVIDIA Tesla T4
```

CUDA was successfully detected and used by PyTorch and SAM 3.

Example verification:

```python
import torch

print(
    "CUDA available:",
    torch.cuda.is_available()
)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )
```

The tested environment reported:

```text
CUDA available: True
GPU: Tesla T4
```

---

## SAM 3 Model Access

The official SAM 3 checkpoint is distributed through a gated Hugging Face repository.

The development workflow therefore requires:

1. a Hugging Face account
2. approved access to the SAM 3 model
3. a Hugging Face access token
4. authentication inside Google Colab

The token is stored securely using **Google Colab Secrets**.

The secret name used by the project is:

```text
HF_TOKEN
```

Authentication tokens must never be committed to GitHub.

---

## SAM 3 Checkpoint

The SAM 3 checkpoint used during development was approximately:

```text
3.21 GB
```

The checkpoint is downloaded and stored in the Hugging Face cache inside the Colab runtime.

The model checkpoint is not committed to this repository.

This keeps the repository lightweight and avoids distributing gated model files.

---

## Dependency Compatibility

One of the most important troubleshooting steps during the Colab workflow involved NumPy compatibility.

Installing and configuring the SAM 3 environment caused dependency changes that conflicted with other packages.

The working environment was stabilized using:

```text
NumPy 1.26.4
```

After restarting the Colab runtime, the verified environment included:

```text
NumPy: 1.26.4
CUDA available: True
GPU: Tesla T4
Ultralytics: 8.4.124
Supervision: 0.30.0
```

This allowed YOLO, Supervision, PyTorch, and SAM 3 to coexist successfully.

---

## Flash Attention Warning

During SAM 3 initialization, the Tesla T4 environment produced a warning indicating that Flash Attention was disabled.

This is expected because the Tesla T4 is based on an older NVIDIA architecture than the GPUs targeted by the optimized Flash Attention path.

The warning did not prevent SAM 3 from running.

SAM 3 successfully loaded and generated segmentation masks.

---

## Image Integration Test

The current integrated test uses:

```text
../assets/input/yolo_bus_test.jpg
```

The image contains:

- a bus
- multiple people
- partially visible people
- objects at different scales and positions

This makes it useful for testing detection, tracking, and segmentation together.

---

## YOLO Test

YOLO successfully processed the test image.

The final integrated run produced:

```text
YOLO detections: 4
```

The detected objects included:

```text
1 bus
3 people
```

---

## ByteTrack Test

ByteTrack successfully processed the YOLO detections.

The final test produced:

```text
Tracked objects: 4
```

Tracker IDs were assigned to the detected objects.

Examples included:

```text
#1 bus
#2 person
#3 person
#4 person
```

---

## SAM 3 Test

SAM 3 was tested using the text prompt:

```text
person
```

Example:

```python
final_result = pipeline.process_image(
    image_bgr=image,
    segmentation_prompt="person"
)
```

The final integrated test produced:

```text
SAM 3 masks: 4
```

The masks included people located near the image boundaries.

---

## Combined Visualization Test

The visualization module was tested by combining:

```text
SAM 3 Masks
      +
YOLO Bounding Boxes
      +
ByteTrack IDs
      +
Class Labels
      +
Confidence Scores
```

The combined visualization test completed successfully.

The final result is stored in:

[`../assets/output/final_integrated_pipeline.jpg`](../assets/output/final_integrated_pipeline.jpg)

---

## Final Output Preview

![Final Integrated Pipeline](../assets/output/final_integrated_pipeline.jpg)

The final image provides visual evidence that the individual modules can operate together inside the same pipeline.

---

## Verified End-to-End Test

The final Colab validation executed the actual:

```python
pipeline.process_image()
```

method rather than manually calling each component.

The result was:

```text
YOLO detections: 4
Tracked objects: 4
SAM 3 masks: 4

UPDATED PIPELINE END-TO-END: SUCCESS
```

This confirms that the source code currently stored in the repository can execute the complete image workflow.

---

## Colab and GitHub Synchronization

During development, source-code changes are synchronized between Google Colab and GitHub.

The development rule is:

```text
Develop / Modify
      |
      v
Test in Colab
      |
      v
Validate Result
      |
      v
Update GitHub
      |
      v
Document Result
```

This helps prevent the Colab environment and repository source code from becoming inconsistent.

---

## Repository Cloning

A clean GitHub clone can be created in Colab using:

```bash
git clone https://github.com/Peyman-mxli/SAM3-Learning-Journey.git
```

The Project 06 directory is:

```text
SAM3-Learning-Journey/
└── 05-projects/
    └── 06-Visual-Tracking-and-Analysis-System/
```

---

## SAM 3 Repository

The official SAM 3 implementation is installed separately from the project repository.

Example:

```bash
git clone https://github.com/facebookresearch/sam3.git /content/sam3_repo
```

Then:

```bash
pip install -e /content/sam3_repo
```

The SAM 3 repository is not copied into the SAM3 Learning Journey repository.

---

## Environment Verification

After installation or a Colab runtime restart, the environment should be verified before running the complete pipeline.

Example:

```python
import numpy as np
import torch
import ultralytics
import supervision as sv

print(
    "NumPy:",
    np.__version__
)

print(
    "CUDA available:",
    torch.cuda.is_available()
)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print(
    "Ultralytics:",
    ultralytics.__version__
)

print(
    "Supervision:",
    sv.__version__
)
```

This helps identify dependency problems before loading large models.

---

## Reproducibility

The Colab documentation should allow another developer to understand how the project was built and tested.

Important reproducibility information includes:

- required dependencies
- GPU requirements
- Hugging Face authentication
- SAM 3 access requirements
- checkpoint handling
- input assets
- model initialization
- expected outputs
- dependency conflicts
- troubleshooting steps
- final test results

---

## Security

Secrets must never be written directly into notebooks that will be committed to GitHub.

Do not commit:

```text
HF_TOKEN
GITHUB_TOKEN
API keys
Passwords
Access tokens
```

Use secure secret-management mechanisms such as **Google Colab Secrets** instead.

---

## Current Notebook Strategy

The project currently uses:

```text
COLAB-WORKFLOW.md
```

to preserve the tested Colab workflow and development process.

A dedicated `.ipynb` notebook may be added later if maintaining an executable notebook provides additional value.

Reusable application logic should continue to remain in:

```text
../src/
```

rather than being duplicated unnecessarily inside notebooks.

---

## Next Development Phase

The next major Colab milestone is **recorded-video processing**.

The planned temporal workflow is:

```text
Recorded Video
      |
      v
Frame Extraction
      |
      v
YOLO Detection
      |
      v
ByteTrack Persistent Tracking
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

This phase will allow the system to test capabilities that cannot be fully validated with a single image.

---

## Planned Video Tests

The recorded-video workflow should evaluate:

- persistent tracker IDs across frames
- object trajectories
- objects entering the scene
- objects leaving the scene
- temporary occlusion
- tracker-ID consistency
- object appearance duration
- frame-by-frame detections
- SAM 3 segmentation during temporal processing
- processing performance
- annotated video generation

---

## Notebook and Source-Code Synchronization

Whenever important code is introduced or changed in Colab:

1. Test it in Colab.
2. Confirm that it works.
3. Update the corresponding source file under `src/`.
4. Commit the tested implementation to GitHub.
5. Update documentation when necessary.
6. Continue to the next experiment.

This keeps the experimental environment and reusable project code synchronized.

---

## Project

This directory belongs to:

[Visual Tracking and Analysis System](../README.md)

Part of the:

[SAM3 Learning Journey](../../../README.md)

---

## Related Documentation

Main Project 06 documentation:

[`../README.md`](../README.md)

Complete Colab workflow:

[`COLAB-WORKFLOW.md`](./COLAB-WORKFLOW.md)

Asset documentation:

[`../assets/README.md`](../assets/README.md)

Input documentation:

[`../assets/input/README.md`](../assets/input/README.md)

Output documentation:

[`../assets/output/README.md`](../assets/output/README.md)

---

## Author

**Peyman Miyandashti**

Computer Vision, Artificial Intelligence, and Software Development

[GitHub](https://github.com/Peyman-mxli)

[LinkedIn](https://www.linkedin.com/in/peyman-mxli/)
