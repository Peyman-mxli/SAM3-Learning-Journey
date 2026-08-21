# Input Assets

This directory contains the input media used by the **Visual Tracking and Analysis System**.

The files stored here serve as the original visual data processed by the computer vision pipeline.

The project currently contains validated input assets for both:

- image-based computer vision
- recorded-video temporal tracking

The current development milestones include:

```text
Image-Based Integration       Completed
Recorded-Video Tracking       Completed
SAM 3 Video Integration       Next Phase
```

---

## Current Input Assets

```text
assets/input/
│
├── README.md
├── yolo_bus_test.jpg
└── tracking_test_01.mp4
```

The two current input assets serve different purposes.

| Input | Purpose | Status |
|---|---|---|
| `yolo_bus_test.jpg` | YOLO + ByteTrack + SAM 3 image integration | Completed |
| `tracking_test_01.mp4` | YOLO + ByteTrack temporal tracking | Completed |

---

# 1. Image Input

## `yolo_bus_test.jpg`

[`yolo_bus_test.jpg`](./yolo_bus_test.jpg)

This image is used as the primary integration-test image for the image-based version of the project.

It contains multiple visible objects, including:

- people
- a bus
- partially visible people near the image boundaries
- background objects
- objects at different positions and scales

These characteristics make the image useful for testing several computer vision components simultaneously.

---

## Image Processing Workflow

The image is used to verify the complete image-processing pipeline:

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
SAM 3 Segmentation Masks
        |
        v
Combined Visualization
        |
        v
final_integrated_pipeline.jpg
```

The final generated result is stored in:

[`../output/final_integrated_pipeline.jpg`](../output/final_integrated_pipeline.jpg)

---

## Verified Image Test Results

The final integrated image test successfully produced:

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

This included partially visible people near the image boundaries.

---

## SAM 3 Image Prompt

The text prompt used during the final image segmentation test was:

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

SAM 3 returned segmentation information that was then passed to the visualization module.

---

## Why the Image Is Useful

The image provides several useful conditions for validating the system.

### Multiple Objects

The image contains several detectable objects, allowing the system to process multiple detections in the same scene.

### Different Object Classes

YOLO identifies more than one object class, including:

```text
bus
person
```

This verifies that the detection stage is not limited to a single class.

### Multiple People

Several people are visible in different areas of the image.

This provides a useful test for the SAM 3 prompt:

```text
person
```

### Partial Visibility

Some people are partially visible near the image boundaries.

This creates a more realistic segmentation challenge than an image containing only centered, fully visible objects.

### Different Object Sizes

Objects appear at different scales, helping evaluate how detection and segmentation behave across the scene.

---

# 2. Recorded-Video Input

## `tracking_test_01.mp4`

[`tracking_test_01.mp4`](./tracking_test_01.mp4)

This is the first recorded-video input used for temporal object-tracking validation in Project 06.

The video was created specifically for testing the transition from single-image processing to sequential frame processing.

It uses the same real-world bus scene as the image test while introducing controlled visual movement across multiple frames.

---

## Video Specifications

The validated video contains:

```text
Resolution: 640 × 360
Frame rate: 15 FPS
Frames: 75
Duration: 5 seconds
Codec: H.264
Container: MP4
```

The short 5-second duration was intentionally selected for the first temporal test.

It provides enough sequential frames to validate ByteTrack behavior while keeping processing time manageable.

---

## How the Test Video Was Created

The video was generated programmatically in Google Colab using:

- OpenCV
- NumPy
- the existing real-world test image
- controlled image movement
- sequential frame generation

The source image was resized and shifted slightly between frames to simulate camera movement.

The basic concept was:

```text
Real Test Image
      |
      v
Resize Image
      |
      v
Apply Controlled Offset
      |
      v
Generate Sequential Frames
      |
      v
Encode Temporary Video
      |
      v
Convert to H.264
      |
      v
tracking_test_01.mp4
```

This created a reproducible test sequence without requiring an external video dataset.

---

## Why Controlled Motion Was Used

A static image repeated across every frame would provide very little temporal variation.

The controlled movement introduces changes between frames while keeping the visible scene predictable.

This allows the project to test:

- frame-by-frame YOLO detection
- ByteTrack association
- persistent tracker IDs
- detection changes between frames
- partial object visibility
- tracker behavior near frame boundaries
- trajectory visualization

---

## Video Processing Workflow

The current validated video workflow is:

```text
tracking_test_01.mp4
        |
        v
Open Video
        |
        v
Read Frame
        |
        v
YOLO Object Detection
        |
        v
Supervision Detections
        |
        v
ByteTrack Update
        |
        v
Persistent Tracker IDs
        |
        v
Tracking Visualization
        |
        v
Write Annotated Frame
        |
        v
H.264 Output Conversion
        |
        v
tracking_output_01.mp4
```

The generated annotated result is stored in:

[`../output/tracking_output_01.mp4`](../output/tracking_output_01.mp4)

---

## Verified Video Test Results

The complete 75-frame sequence was successfully processed.

The test produced:

```text
VIDEO TRACKING TEST: SUCCESS

Frames processed: 75
Total detections: 253
Unique tracker IDs: [1, 2, 3, 4, 5, 6]
Unique tracked objects: 6
```

Sample tracker states during the sequence were:

```text
Frame 015 | Detections: 4 | Tracker IDs: [2, 1, 4, 3]

Frame 030 | Detections: 4 | Tracker IDs: [2, 1, 5, 4]

Frame 045 | Detections: 3 | Tracker IDs: [2, 1, 3]

Frame 060 | Detections: 4 | Tracker IDs: [2, 1, 3, 6]

Frame 075 | Detections: 3 | Tracker IDs: [2, 1, 3]
```

---

## Temporal Tracking Result

This test is different from the earlier single-image ByteTrack test.

With a single image, tracker IDs can be assigned, but there is no temporal sequence across which identity persistence can be evaluated.

With:

```text
tracking_test_01.mp4
```

the same ByteTrack instance remains active while all 75 frames are processed sequentially.

This allows the system to test actual temporal tracking behavior.

Tracker IDs:

```text
1
2
```

remained present throughout the sampled portions of the video.

Other tracker IDs changed when objects were temporarily lost or re-associated.

---

## Understanding the Tracker IDs

The video produced six unique tracker IDs:

```text
[1, 2, 3, 4, 5, 6]
```

This does not necessarily represent six continuously visible physical objects.

A new tracker ID may appear when:

- YOLO temporarily loses an object
- detection confidence changes
- an object becomes partially visible
- an object moves near the image boundary
- frame-to-frame association is interrupted
- an object is detected again after being lost

These behaviors provide useful data for future tracking evaluation.

---

## Input Video and Output Video

The relationship between the two files is:

```text
assets/input/
│
└── tracking_test_01.mp4
        |
        v
YOLO + ByteTrack
        |
        v
assets/output/
│
└── tracking_output_01.mp4
```

The input video contains the original generated sequence.

The output video contains:

- YOLO bounding boxes
- class labels
- confidence scores
- ByteTrack tracker IDs
- object traces

---

## Browser Compatibility

The first OpenCV-generated version of the video used the `mp4v` codec.

Although OpenCV could read the file, browser playback inside Google Colab was not reliable.

The video was therefore converted using FFmpeg to:

```text
Codec: H.264
Pixel format: yuv420p
Container: MP4
```

After conversion, the video played successfully inside the Google Colab browser player.

This ensures that the test asset is easier to preview across different environments.

---

# Supported Input Types

The project architecture currently supports or is designed to support:

- images
- recorded videos
- short computer vision test sequences

Both image and recorded-video processing have now been validated.

---

# Current Image Capabilities

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

# Current Video Capabilities

For recorded-video input, the system currently supports:

- OpenCV video loading
- sequential frame processing
- YOLO detection on every frame
- conversion to `supervision.Detections`
- ByteTrack updates across frames
- persistent tracker state
- tracker-ID visualization
- bounding boxes
- class labels
- confidence scores
- object traces
- annotated video generation
- H.264 output conversion
- browser-compatible playback

---

# Current Asset Flow

The project now has two validated input/output workflows.

## Image Workflow

```text
assets/input/
│
└── yolo_bus_test.jpg
        |
        v
YOLO
        |
        v
ByteTrack
        |
        v
SAM 3
        |
        v
Supervision Visualization
        |
        v
assets/output/
│
└── final_integrated_pipeline.jpg
```

## Video Workflow

```text
assets/input/
│
└── tracking_test_01.mp4
        |
        v
Frame-by-Frame YOLO
        |
        v
ByteTrack
        |
        v
Tracking Visualization
        |
        v
H.264 Encoding
        |
        v
assets/output/
│
└── tracking_output_01.mp4
```

---

# Why Separate Image and Video Tests

The image and video tests validate different capabilities.

The image test focuses on:

```text
Detection
    +
Segmentation
    +
Integration
```

The video test focuses on:

```text
Detection
    +
Temporal Tracking
    +
Tracker Persistence
```

The next phase will combine both:

```text
Detection
    +
Temporal Tracking
    +
SAM 3 Segmentation
```

---

# Future Testing Conditions

Additional input media should eventually include different visual conditions such as:

- different lighting conditions
- different object sizes
- natural object movement
- partial occlusion
- different backgrounds
- different camera perspectives
- motion blur
- crowded scenes
- objects entering the scene
- objects leaving the scene
- low-light environments
- longer sequences

These conditions will help evaluate where the system performs well and where it fails.

---

# Future Video Inputs

Additional video tests may include:

```text
tracking_test_02.mp4
pedestrian_sequence_01.mp4
occlusion_test_01.mp4
movement_test_01.mp4
low_light_test_01.mp4
```

These videos can be used to evaluate increasingly difficult tracking scenarios.

---

# File Organization

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

# Input and Output Separation

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

# Related Source Code

The integrated image pipeline is implemented in:

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

# Related Outputs

The image-based integrated output is:

[`../output/final_integrated_pipeline.jpg`](../output/final_integrated_pipeline.jpg)

The temporal tracking output is:

[`../output/tracking_output_01.mp4`](../output/tracking_output_01.mp4)

Full output documentation is available in:

[`../output/README.md`](../output/README.md)

---

# Related Documentation

The complete Google Colab setup and testing process is documented in:

[`../../notebooks/COLAB-WORKFLOW.md`](../../notebooks/COLAB-WORKFLOW.md)

The main project documentation is available at:

[`../../README.md`](../../README.md)

---

# Important Notes

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

# Current Milestones

```text
MILESTONE 1

Image Input
    |
    v
YOLO + ByteTrack + SAM 3
    |
    v
COMPLETED


MILESTONE 2

Recorded Video Input
    |
    v
YOLO + ByteTrack
    |
    v
COMPLETED


MILESTONE 3

Recorded Video Input
    |
    v
YOLO + ByteTrack + SAM 3
    |
    v
NEXT
```

---

# Next Input Milestone

The next phase does **not** require creating another test video.

The existing:

```text
tracking_test_01.mp4
```

can be reused to test SAM 3 video integration.

The next processing goal is:

```text
tracking_test_01.mp4
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
Annotated SAM 3 Tracking Video
```

This will combine the two previously validated capabilities into one temporal computer vision pipeline.

---

## Author

**Peyman Miyandashti**

Computer Vision, Artificial Intelligence, and Software Development

[GitHub](https://github.com/Peyman-mxli)

[LinkedIn](https://www.linkedin.com/in/peyman-mxli/)
