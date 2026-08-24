# Session 06 — Class Recording

## Segmentation with SAM 3

This file contains the class recording for **Session 06 — Segmentation with SAM 3** from my **SAM3 Computer Vision Learning Journey**.

---

## Class Recording

**YouTube:**

[▶ Watch Session 06 — Segmentation with SAM 3](https://youtu.be/1EYfpSsmHO0)

---

## Session Topic

**Segmentation with Segment Anything Model 3 (SAM 3)**

This session introduces image segmentation and demonstrates how object detection and segmentation can be combined to move from rectangular bounding boxes to precise pixel-level object masks.

The main workflow covered during the session is:

    Input Image
         ↓
       YOLOv8
         ↓
    Object Detection
         ↓
    Bounding Boxes
         ↓
    SAM 3 Prompts
         ↓
    Segmentation Masks
         ↓
    Supervision Detections
         ↓
    Mask Analysis

---

## Main Concepts Covered

The class introduces and explores:

- Object detection vs. image segmentation
- Bounding boxes vs. segmentation masks
- SAM 3 segmentation
- Boolean segmentation masks
- NumPy mask operations
- YOLOv8 object detection
- YOLO bounding boxes as SAM 3 prompts
- Conversion to `sv.Detections`
- Pixel-level object extraction
- Mask-area calculation
- Bounding-box-area comparison
- SAM 3 text prompts
- Mask serialization
- NumPy `packbits`
- Base64 encoding
- JSON storage
- Segmentation-mask reconstruction

---

## YOLO + SAM 3

The session demonstrates how YOLO and SAM 3 can work together.

YOLO first detects the objects:

    Image
      ↓
    YOLOv8
      ↓
    Bounding Boxes

The detected bounding boxes are then used as prompts for SAM 3:

    Bounding Boxes
         ↓
       SAM 3
         ↓
    Segmentation Masks

This allows YOLO to provide fast object localization while SAM 3 provides a much more precise pixel-level representation of each detected object.

---

## Segmentation Masks

A segmentation mask identifies exactly which pixels belong to an object.

The masks used during the session are represented as NumPy boolean arrays.

Each pixel contains:

    True  → pixel belongs to the object
    False → pixel belongs to the background

This representation makes it possible to directly analyze segmentation results using NumPy.

---

## Mask Analysis

The session demonstrates several operations that can be performed directly on segmentation masks.

Examples include:

    mask.sum()

to calculate the number of pixels belonging to an object.

The fraction of the complete image occupied by an object can be calculated using:

    fraction = mask.sum() / mask.size

Masks can also be used to remove the background:

    object_crop = image.copy()
    object_crop[~mask] = 0

This isolates the segmented object with pixel-level precision.

---

## Mask Area vs. Bounding-Box Area

Another concept explored during the session is the difference between the area of a segmentation mask and the area of its corresponding bounding box.

A bounding box includes rectangular background space surrounding an object.

A segmentation mask represents only the pixels belonging to the object itself.

Conceptually:

    Bounding Box Area
           ↓
    Rectangle Around Object

while:

    Segmentation Mask Area
           ↓
    Actual Object Pixels

This demonstrates why segmentation provides more precise geometric information than object detection alone.

---

## SAM 3 Text Prompts

The session also introduces the ability of SAM 3 to work with text prompts.

Instead of using a bounding box as the prompt, SAM 3 can work with semantic concepts such as:

    person

Conceptually:

    Image
      ↓
    Text Prompt
      ↓
    SAM 3
      ↓
    Segmentation Masks

This capability provides another way to identify and segment objects based on semantic descriptions.

---

## Saving Segmentation Masks

Segmentation masks are NumPy boolean arrays and cannot be stored directly inside standard JSON.

The session demonstrates a compact serialization process:

    Boolean Mask
         ↓
    Flatten Array
         ↓
    np.packbits
         ↓
    Binary Data
         ↓
    Base64 Encoding
         ↓
    JSON Storage

This allows segmentation masks to be preserved together with detection metadata.

---

## Recovering Stored Masks

The encoded mask can later be reconstructed by reversing the serialization process.

The workflow is:

    Base64 String
         ↓
    Decode Bytes
         ↓
    NumPy Array
         ↓
    np.unpackbits
         ↓
    Reshape
         ↓
    Boolean Segmentation Mask

This makes it possible to store segmentation results and recover them later for additional analysis.

---

## Practical Implementation

The concepts from this class were implemented in the practical located at:

    08-course-notes/06-Segmentation-with-SAM/practical/

Main implementation:

    segmentation_with_sam.py

Input image:

    assets/input/bus.jpg

Generated results:

    assets/output/raw_mask.png
    assets/output/segmented_object.png
    assets/output/segmentation_results.json

---

## Validated Practical Results

The practical implementation associated with this session was successfully executed and validated using **Google Colab with an NVIDIA T4 GPU**.

YOLOv8 detected:

    5 objects

consisting of:

    4 persons
    1 bus

SAM 3 generated:

    5 segmentation masks

Mask array shape:

    (5, 1080, 810)

The first segmentation mask contained:

    Object pixels: 263695
    Total pixels: 874800
    Image coverage: 30.14%

The serialized mask was successfully reconstructed.

Validation result:

    Decoded mask matches original: True

Final execution status:

    Segmentation practical completed successfully.

---

## Learning Outcome

After completing this session and its practical implementation, I understand the fundamental difference between object detection and segmentation.

YOLOv8 provides efficient object localization using bounding boxes, while SAM 3 converts those approximate locations into precise pixel-level masks.

I also practiced how to:

- Inspect segmentation masks
- Measure mask areas
- Extract objects from images
- Compare masks with bounding boxes
- Serialize segmentation masks
- Store masks in JSON
- Reconstruct encoded masks
- Validate recovered segmentation data

These concepts provide the foundation for more advanced SAM 3 workflows involving prompts, tracking, video segmentation, object analysis, and mask-based computer vision pipelines.

---

## Related Files

Main session documentation:

    README.md

Practical documentation:

    practical/README.md

Practical implementation:

    practical/segmentation_with_sam.py

Generated output documentation:

    practical/assets/output/README.md

---

## Recording

[▶ Session 06 — Segmentation with SAM 3 — YouTube](https://youtu.be/1EYfpSsmHO0)
