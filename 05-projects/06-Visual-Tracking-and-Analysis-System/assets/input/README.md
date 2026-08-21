# Input Assets

This directory contains the input media used by the **Visual Tracking and Analysis System**.

The files stored here serve as the original visual data processed by the computer vision pipeline.

---

## Supported Input

The project is designed initially to process:

- Images
- Recorded videos
- Short computer vision test sequences

These inputs can be used for:

- Object detection
- Object segmentation
- Object tracking
- Position analysis
- Movement analysis
- Model evaluation
- Failure analysis

---

## Recommended Initial Tests

For the first technical tests, the project may use:

- 20–50 images

or

- 2–5 short recorded videos

The goal of the initial test is to verify that the complete processing pipeline works correctly before performing larger evaluations.

---

## Testing Conditions

When possible, input media should include different visual conditions such as:

- Different lighting conditions
- Different object sizes
- Object movement
- Partial occlusion
- Different backgrounds
- Different camera perspectives
- Motion blur
- Objects entering or leaving the scene

This will help evaluate where the system performs well and where it fails.

---

## File Organization

Use descriptive filenames whenever possible.

Examples:

```text
tracking_test_01.mp4
tracking_test_02.mp4
pedestrian_sequence_01.mp4
object_detection_test_01.jpg
low_light_test_01.mp4
occlusion_test_01.mp4
