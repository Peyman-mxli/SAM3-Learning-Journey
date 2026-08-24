# 06 — Segmentation with SAM

This directory contains small, focused Python examples based on **Session 06 — Segmentation with SAM 3** from my SAM3 Computer Vision Learning Journey.

The examples in this folder demonstrate how to combine **YOLOv8**, **SAM 3**, **Supervision**, and **NumPy** to move from object detection with bounding boxes to precise pixel-level segmentation masks.

Unlike the larger practical implementation, these examples are intentionally separated into small reusable scripts so that each concept can be studied independently.

---

## Topics Covered

The examples in this folder cover:

- YOLOv8 object detection
- Extracting bounding boxes
- Using YOLO bounding boxes as SAM 3 prompts
- Generating segmentation masks
- Converting SAM 3 results into `sv.Detections`
- Inspecting boolean segmentation masks
- Measuring mask area
- Extracting objects using masks
- Comparing mask area with bounding-box area
- Encoding segmentation masks using Base64
- Decoding and reconstructing stored masks

---

## Example Structure

    06-Segmentation-with-SAM/
    ├── README.md
    ├── 01_yolo_detection.py
    ├── 02_sam_bbox_segmentation.py
    ├── 03_mask_inspection.py
    ├── 04_object_extraction.py
    ├── 05_mask_area_comparison.py
    └── 06_mask_serialization.py

---

## 01 — YOLO Detection

File:

    01_yolo_detection.py

This example demonstrates the first stage of the pipeline.

It:

- Loads an image
- Runs YOLOv8 object detection
- Converts the results into `sv.Detections`
- Prints the detected bounding boxes
- Prints class IDs and confidence scores

The bounding boxes generated here are later used as prompts for SAM 3.

---

## 02 — SAM Bounding-Box Segmentation

File:

    02_sam_bbox_segmentation.py

This example demonstrates how YOLO bounding boxes can be used as prompts for SAM 3.

Pipeline:

    Image
      ↓
    YOLOv8
      ↓
    Bounding Boxes
      ↓
    SAM 3
      ↓
    Segmentation Masks

The example converts both YOLO and SAM outputs into Supervision detections.

---

## 03 — Mask Inspection

File:

    03_mask_inspection.py

This example focuses on the segmentation mask as a NumPy data structure.

It demonstrates how to inspect:

- Mask type
- Mask shape
- Unique values
- Number of object pixels
- Total number of pixels
- Percentage of the image occupied by the object

A SAM 3 segmentation mask is represented as a boolean NumPy array:

    True  → pixel belongs to the object
    False → pixel belongs to the background

---

## 04 — Object Extraction

File:

    04_object_extraction.py

This example demonstrates how a segmentation mask can be used to isolate an object from the original image.

The main operation is:

    object_image[~mask] = 0

This removes pixels that do not belong to the segmented object.

The result is a pixel-level object extraction instead of a rectangular crop.

---

## 05 — Mask Area Comparison

File:

    05_mask_area_comparison.py

This example compares:

    Segmentation Mask Area

with:

    Bounding Box Area

The comparison shows how much of each bounding box is actually occupied by the segmented object.

The calculation is:

    percentage = (
        mask_area
        / bounding_box_area
    ) * 100

This demonstrates why segmentation provides more precise geometric information than bounding boxes alone.

---

## 06 — Mask Serialization

File:

    06_mask_serialization.py

This example demonstrates how segmentation masks can be stored in a JSON-compatible format.

The mask is:

1. Flattened
2. Packed using `np.packbits`
3. Converted to bytes
4. Encoded using Base64

The example also reconstructs the original mask using:

- Base64 decoding
- `np.frombuffer`
- `np.unpackbits`
- NumPy reshape

Finally, it validates the reconstructed mask using:

    np.array_equal(
        original_mask,
        decoded_mask
    )

---

## Technologies Used

- Python
- Ultralytics
- YOLOv8
- SAM 3
- Supervision
- NumPy
- OpenCV
- Base64
- JSON

---

## SAM 3 Model

The SAM 3 model weights are not stored inside this GitHub repository because the model file is very large.

The examples expect the SAM 3 model to be available externally.

For my validated Google Colab environment, the model is stored at:

    /content/drive/MyDrive/SAM3-Models/sam3.pt

The examples that require SAM 3 use this path.

---

## Input Image

The examples use the Ultralytics sample image:

    bus.jpg

This image is useful because YOLOv8 detects several objects, including people and a bus, allowing multiple segmentation masks to be generated and analyzed.

---

## Learning Goal

The purpose of these examples is to break the complete segmentation workflow into smaller reusable pieces.

The main progression is:

    Detection
       ↓
    Bounding Boxes
       ↓
    SAM 3 Segmentation
       ↓
    Mask Inspection
       ↓
    Object Extraction
       ↓
    Geometric Analysis
       ↓
    Mask Serialization

These examples provide a foundation for more advanced SAM 3 workflows involving segmentation, tracking, video analysis, and structured computer vision pipelines.
