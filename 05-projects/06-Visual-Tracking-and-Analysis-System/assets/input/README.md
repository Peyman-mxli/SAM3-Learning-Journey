# Input Assets

This directory contains the input media used by the **Visual Tracking and Analysis System**.

The files stored here serve as the original visual data processed by the computer vision pipeline.

The current image-based MVP has been successfully tested using a real input image through the complete:

**YOLO → ByteTrack → SAM 3 → Visualization**

workflow.

---

## Current Input Asset

### `yolo_bus_test.jpg`

[`yolo_bus_test.jpg`](./yolo_bus_test.jpg)

This image is used as the primary integration-test image for the current version of the project.

It contains multiple visible objects, including:

- people
- a bus
- partially visible people near the image boundaries
- background objects
- objects at different positions and scales

These characteristics make the image useful for testing several computer vision components simultaneously.

---

## Purpose of the Test Image

The test image is used to verify the complete image-processing pipeline:

```text
yolo_bus_test.jpg
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
Combined Visualization
        |
        v
Final Integrated Output
```

The final generated result is stored in:

[`../output/final_integrated_pipeline.jpg`](../output/final_integrated_pipeline.jpg)

---

## Verified Test Results

The final integrated test successfully produced:

```text
YOLO detections: 4
Tracked objects: 4
SAM 3 masks: 4

UPDATED PIPELINE END-TO-END: SUCCESS
```

The YOLO detector identified:

```text
1 bus
3 people
```

ByteTrack assigned tracker IDs to all four YOLO detections.

SAM 3 was independently prompted with:

```text
person
```

and generated four segmentation masks for people visible in the image.

This includes partially visible people near the image boundaries.

---

## SAM 3 Test Prompt

The text prompt used during the final segmentation test was:

```text
person
```

The prompt was passed to SAM 3 through the project's segmentation module.

Example:

```python
final_result = pipeline.process_image(
    image_bgr=image,
    segmentation_prompt="person"
)
```

SAM 3 then returned segmentation information that was passed to the visualization module.

---

## Why This Image Is Useful

The current test image provides several useful conditions for validating the system.

### Multiple Objects

The image contains several detectable objects, allowing the system to test multiple detections in a single scene.

### Different Object Classes

YOLO identifies more than one object class, including:

```text
bus
person
```

This verifies that the detection stage is not limited to a single class.

### Multiple People

Several people are visible in different areas of the image.

This provides a useful test for the SAM 3 text prompt:

```text
person
```

### Partial Visibility

Some people are only partially visible near the image boundaries.

This provides a more realistic segmentation challenge than an image containing only centered, fully visible objects.

### Different Object Sizes

Objects appear at different scales, helping evaluate how detection and segmentation behave across the scene.

---

## Supported Input Types

The project architecture is designed to support:

- images
- recorded videos
- short computer vision test sequences

The current validated implementation focuses on **image processing**.

Recorded-video processing is the next major development phase.

---

## Image Processing

For image input, the system currently supports:

- YOLO object detection
- conversion to `supervision.Detections`
- ByteTrack ID assignment
- SAM 3 text-prompt segmentation
- segmentation-mask generation
- bounding-box visualization
- class labels
- confidence scores
- tracker IDs
- combined output visualization

---

## Future Video Input

The next phase of the project will introduce recorded-video input.

Example future files may include:

```text
tracking_test_01.mp4
pedestrian_sequence_01.mp4
occlusion_test_01.mp4
movement_test_01.mp4
```

Video input will allow the project to evaluate capabilities that cannot be fully demonstrated with a single image, including:

- persistent tracking across frames
- object trajectories
- object movement
- objects entering the scene
- objects leaving the scene
- temporary occlusion
- tracker-ID consistency
- appearance duration
- frame-by-frame analytics

---

## Future Testing Conditions

Additional input media should eventually include different visual conditions such as:

- different lighting conditions
- different object sizes
- object movement
- partial occlusion
- different backgrounds
- different camera perspectives
- motion blur
- crowded scenes
- objects entering or leaving the scene
- low-light environments

These conditions will help evaluate where the system performs well and where it fails.

---

## File Organization

Input files should use descriptive filenames whenever possible.

Examples:

```text
yolo_bus_test.jpg
tracking_test_01.mp4
tracking_test_02.mp4
pedestrian_sequence_01.mp4
object_detection_test_01.jpg
low_light_test_01.mp4
occlusion_test_01.mp4
```

Avoid generic filenames such as:

```text
image.jpg
test.jpg
video.mp4
```

Descriptive filenames make experiments easier to identify and reproduce.

---

## Input and Output Separation

Original media should remain inside:

```text
assets/input/
```

Generated files should be stored inside:

```text
assets/output/
```

The pipeline should never overwrite the original input asset.

This separation makes experiments reproducible and preserves the original test data.

---

## Current Asset Flow

```text
assets/input/
│
└── yolo_bus_test.jpg
        |
        v
VisualAnalysisPipeline
        |
        +-- YOLO
        |
        +-- ByteTrack
        |
        +-- SAM 3
        |
        +-- Supervision
        |
        v
assets/output/
│
└── final_integrated_pipeline.jpg
```

---

## Related Source Code

The input image is processed through the central pipeline:

[`../../src/pipeline.py`](../../src/pipeline.py)

Detection is implemented in:

[`../../src/detector.py`](../../src/detector.py)

Tracking is implemented in:

[`../../src/tracker.py`](../../src/tracker.py)

SAM 3 segmentation is implemented in:

[`../../src/segmenter.py`](../../src/segmenter.py)

Visualization is implemented in:

[`../../src/visualization.py`](../../src/visualization.py)

---

## Related Documentation

The generated output is documented in:

[`../output/README.md`](../output/README.md)

The complete Google Colab setup and testing process is documented in:

[`../../notebooks/COLAB-WORKFLOW.md`](../../notebooks/COLAB-WORKFLOW.md)

The main project documentation is available at:

[`../../README.md`](../../README.md)

---

## Important Notes

Input assets should not contain:

- API keys
- Hugging Face tokens
- GitHub tokens
- credentials
- private information
- large model checkpoints

Model files such as:

```text
sam3.pt
```

should not be committed to the repository.

Authentication information should be stored securely using environment variables or services such as **Google Colab Secrets**.

---

## Next Input Milestone

The next major input milestone is to add a short recorded video suitable for testing:

```text
YOLO Detection
      +
ByteTrack Persistent Tracking
      +
SAM 3 Segmentation
      +
Object Trajectories
      +
Tracking Analytics
      +
Annotated Video Output
```

This will extend the current image-based MVP into a temporal computer vision pipeline.

---

## Author

**Peyman Miyandashti**

Computer Vision, Artificial Intelligence, and Software Development

[GitHub](https://github.com/Peyman-mxli)

[LinkedIn](https://www.linkedin.com/in/peyman-mxli/)
