# 06 — Segmentation with SAM

This directory contains small, focused Python examples based on **Session 06 — Segmentation with SAM 3** from my **SAM3 Computer Vision Learning Journey**.

The examples demonstrate how to combine **YOLOv8**, **SAM 3**, **Supervision**, and **NumPy** to move from object detection with bounding boxes to precise pixel-level segmentation masks.

Unlike the larger practical implementation, these examples are intentionally separated into small reusable scripts so that each concept can be studied independently.

All six examples were successfully executed and validated in **Google Colab using an NVIDIA T4 GPU**.

---

## Topics Covered

The examples in this folder cover:

- YOLOv8 object detection
- Extracting bounding boxes
- Using YOLO bounding boxes as SAM 3 prompts
- Generating segmentation masks
- Converting YOLO and SAM 3 results into `sv.Detections`
- Inspecting boolean segmentation masks
- Measuring mask area
- Calculating image coverage
- Extracting objects using segmentation masks
- Comparing mask area with bounding-box area
- Encoding segmentation masks using Base64
- Storing masks in JSON
- Decoding and reconstructing stored masks
- Validating reconstructed masks

---

## Example Structure

    06-Segmentation-with-SAM/
    ├── README.md
    ├── bus.jpg
    ├── 01_yolo_detection.py
    ├── 02_sam_bbox_segmentation.py
    ├── 03_mask_inspection.py
    ├── 04_object_extraction.py
    ├── 05_mask_area_comparison.py
    └── 06_mask_serialization.py

The Python examples may generate temporary output files when executed, such as:

    extracted_object.png
    mask_serialization_example.json

These generated files are execution artifacts and are not required as source files for the examples.

---

# 01 — YOLO Detection

File:

    01_yolo_detection.py

This example demonstrates the first stage of the segmentation pipeline.

It:

- Loads `bus.jpg`
- Runs YOLOv8 object detection
- Converts the results into `sv.Detections`
- Retrieves class IDs
- Retrieves confidence scores
- Extracts bounding boxes
- Converts bounding boxes into a Python list
- Prepares the boxes for use as SAM 3 prompts

Pipeline:

    Image
      ↓
    YOLOv8
      ↓
    Object Detections
      ↓
    Bounding Boxes

### Validated Result

The example successfully detected:

    6 objects

The detections were:

    1 bus
    4 persons
    1 stop sign

The six bounding boxes were successfully converted into a format ready for SAM 3.

Final execution status:

    YOLO detection example completed.

---

# 02 — SAM Bounding-Box Segmentation

File:

    02_sam_bbox_segmentation.py

This example demonstrates how YOLO bounding boxes can be used as spatial prompts for SAM 3.

Pipeline:

    Image
      ↓
    YOLOv8
      ↓
    6 Object Detections
      ↓
    Bounding Boxes
      ↓
    SAM 3
      ↓
    6 Segmentation Masks

Both YOLO and SAM 3 outputs are converted into Supervision detections using:

    sv.Detections.from_ultralytics(...)

### Validated Result

YOLO produced:

    YOLO detections: 6

SAM 3 produced:

    SAM detections: 6
    SAM masks generated: 6
    Mask array shape: (6, 1080, 810)

The number of YOLO detections and SAM 3 masks matched exactly.

Validation:

    Each YOLO bounding box produced a SAM 3 mask.

Final execution status:

    SAM bounding-box segmentation example completed.

---

# 03 — Mask Inspection

File:

    03_mask_inspection.py

This example focuses on understanding a SAM 3 segmentation mask as a NumPy data structure.

A segmentation mask is represented as a boolean array:

    True  → pixel belongs to the object
    False → pixel belongs to the background

The example inspects:

- Mask type
- Mask dimensions
- NumPy data type
- Unique values
- Object pixel count
- Background pixel count
- Total pixel count
- Image coverage
- Complete SAM mask-array dimensions

### Validated Result

SAM 3 generated:

    6 masks

The first mask contained:

    Type: numpy.ndarray
    Shape: (1080, 810)
    Data type: bool
    Unique values: [False True]

Pixel analysis:

    Object pixels: 265686
    Background pixels: 609114
    Total pixels: 874800

Image coverage:

    Fraction: 0.3037
    Percentage: 30.37%

The complete SAM mask array had the shape:

    (6, 1080, 810)

which represents:

    (number_of_objects, image_height, image_width)

Final execution status:

    Mask inspection example completed.

---

# 04 — Object Extraction

File:

    04_object_extraction.py

This example demonstrates how a SAM 3 segmentation mask can be used to isolate an object from the original image with pixel-level precision.

The main operation is:

    object_image[~mask] = 0

The inverted mask selects pixels that do not belong to the object and sets them to black.

Pixels inside the segmentation mask remain unchanged.

Pipeline:

    Original Image
         ↓
    Segmentation Mask
         ↓
    Invert Mask
         ↓
    Remove Background Pixels
         ↓
    Extracted Object

### Validated Result

YOLO detected:

    6 objects

SAM 3 generated:

    6 masks

The selected first mask had:

    Shape: (1080, 810)
    Object pixels: 265686

The example successfully generated:

    extracted_object.png

Final execution status:

    Object extraction example completed.

---

# 05 — Mask Area Comparison

File:

    05_mask_area_comparison.py

This example compares the area of each SAM 3 segmentation mask with the area of its corresponding YOLO bounding box.

The calculation is:

    percentage = (
        mask_area
        / bounding_box_area
    ) * 100

A bounding box contains a rectangular region around an object.

A segmentation mask contains only the pixels identified as belonging to the object.

This comparison therefore demonstrates how much of the bounding box is actually occupied by the segmented object.

### Validated Results

Six objects were available for comparison:

| Object | Class | Mask Area | Bounding Box Area | Mask / Box |
|---|---|---:|---:|---:|
| 0 | bus | 265,686 px | 411,059.31 px | 64.63% |
| 1 | person | 46,648 px | 99,214.33 px | 47.02% |
| 2 | person | 20,935 px | 67,998.79 px | 30.79% |
| 3 | person | 32,911 px | 55,768.55 px | 59.01% |
| 4 | person | 10,715 px | 20,346.07 px | 52.66% |
| 5 | stop sign | 1,878 px | 2,288.43 px | 82.07% |

The stop sign produced the highest mask-to-box percentage:

    82.07%

The third detected person produced the lowest:

    30.79%

This illustrates how the amount of background contained inside a bounding box can vary significantly between detections.

Final execution status:

    Mask area comparison example completed.

---

# 06 — Mask Serialization

File:

    06_mask_serialization.py

This example demonstrates how a boolean SAM 3 segmentation mask can be converted into a compact JSON-compatible representation.

Encoding pipeline:

    Boolean Mask
         ↓
    Flatten
         ↓
    np.packbits
         ↓
    Packed Bytes
         ↓
    Base64 Encoding
         ↓
    JSON

The example then performs the reverse process:

    JSON
      ↓
    Base64 Decode
      ↓
    np.frombuffer
      ↓
    np.unpackbits
      ↓
    Reshape
      ↓
    Boolean Mask

The reconstructed mask is compared with the original using:

    np.array_equal(
        original_mask,
        decoded_mask
    )

### Validated Result

Original mask:

    Shape: (1080, 810)
    Data type: bool
    Object pixels: 265686

Serialization:

    Boolean pixels: 874800
    Packed bytes: 109350
    Base64 characters: 145800

The example successfully generated:

    mask_serialization_example.json

Decoded mask:

    Shape: (1080, 810)
    Data type: bool

Final validation:

    Decoded mask matches original: True

This confirms that the serialization and reconstruction process preserves the original segmentation mask exactly.

Final execution status:

    Mask serialization example completed.

---

# Input Image

All examples use:

    bus.jpg

Image dimensions:

    1080 × 810 pixels

Image array shape:

    (1080, 810, 3)

During the validated YOLOv8 execution, the image produced:

    4 persons
    1 bus
    1 stop sign

for a total of:

    6 detections

The multiple detected objects make the image useful for testing bounding-box prompting, multi-object segmentation, mask analysis, and geometric comparisons.

---

# SAM 3 Model

The SAM 3 model weights are not stored inside this GitHub repository because the model checkpoint is very large.

The examples that require SAM 3 expect the model to be available externally.

For the validated Google Colab environment, the model was stored at:

    /content/drive/MyDrive/SAM3-Models/sam3.pt

Validated model size:

    3.21 GB

Examples requiring SAM 3:

    02_sam_bbox_segmentation.py
    03_mask_inspection.py
    04_object_extraction.py
    05_mask_area_comparison.py
    06_mask_serialization.py

`01_yolo_detection.py` only requires YOLOv8 and does not load the SAM 3 checkpoint.

---

# Validated Environment

The examples were tested using:

- Google Colab
- NVIDIA T4 GPU
- Python
- Ultralytics
- YOLOv8
- SAM 3
- Supervision
- NumPy
- OpenCV
- Base64
- JSON

Google Drive was mounted to provide access to the SAM 3 checkpoint.

---

# SAM 3 Image-Size Warning

During SAM 3 inference, Ultralytics displayed:

    imgsz=[1024] must be multiple of max stride 14,
    updating to [1036]

This is an automatic adjustment performed by Ultralytics.

SAM 3 inference continued normally using:

    1036 × 1036

and all six segmentation masks were generated successfully.

---

# Complete Learning Progression

The six examples form the following progression:

    01 — YOLO Detection
             ↓
    Detect Objects
             ↓
    Extract Bounding Boxes
             ↓
    02 — SAM Bounding-Box Segmentation
             ↓
    Generate Pixel-Level Masks
             ↓
    03 — Mask Inspection
             ↓
    Understand Boolean Mask Data
             ↓
    04 — Object Extraction
             ↓
    Apply Mask to Original Image
             ↓
    05 — Mask Area Comparison
             ↓
    Analyze Mask Geometry
             ↓
    06 — Mask Serialization
             ↓
    Store and Recover Mask Data
             ↓
    Validate Reconstruction

---

# Validation Summary

All six examples were successfully executed.

| Example | Result |
|---|---|
| `01_yolo_detection.py` | ✅ Passed |
| `02_sam_bbox_segmentation.py` | ✅ Passed |
| `03_mask_inspection.py` | ✅ Passed |
| `04_object_extraction.py` | ✅ Passed |
| `05_mask_area_comparison.py` | ✅ Passed |
| `06_mask_serialization.py` | ✅ Passed |

Key validated results:

    Input image: bus.jpg
    Image shape: (1080, 810, 3)

    YOLO detections: 6
    SAM masks generated: 6
    Mask array shape: (6, 1080, 810)

    First mask object pixels: 265686
    First mask image coverage: 30.37%

    Decoded mask matches original: True

---

# Learning Goal

The purpose of these examples is to break the complete segmentation workflow into small, reusable components.

The main lesson is that object detection and segmentation provide complementary information.

**YOLOv8** identifies where objects are located using bounding boxes.

**SAM 3** uses those spatial prompts to identify the precise pixels belonging to each object.

**Supervision** provides a consistent representation for handling outputs from both models.

**NumPy** makes it possible to directly inspect, measure, manipulate, serialize, and reconstruct segmentation masks.

Together, these examples provide a practical foundation for more advanced workflows involving:

- Instance segmentation
- Prompt-based segmentation
- Object extraction
- Mask visualization
- Pixel-level measurements
- Tracking with segmentation
- Video segmentation
- Structured computer vision data
- Persistent segmentation results
