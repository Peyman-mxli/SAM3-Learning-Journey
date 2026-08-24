# Practical — Segmentation with SAM 3

This folder contains the practical implementation for **Session 06 — Segmentation with SAM 3**.

The practical demonstrates how to combine **YOLOv8**, **SAM 3**, and **Supervision** to move from object detection with rectangular bounding boxes to precise pixel-level segmentation masks.

The complete pipeline was implemented, executed, and validated in **Google Colab using a T4 GPU**.

---

## Practical Objective

The objective of this practical is to build and understand the following computer vision pipeline:

    Input Image
         ↓
       YOLOv8
         ↓
    Object Detections
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
         ↓
    Object Extraction
         ↓
    JSON Export
         ↓
    Validation

The practical focuses on understanding how segmentation masks are generated, represented, analyzed, serialized, and validated.

---

## Main Steps

The practical workflow includes:

1. Load an input image.
2. Run YOLOv8 object detection.
3. Convert YOLO results into `sv.Detections`.
4. Extract detected bounding boxes.
5. Use the YOLO bounding boxes as prompts for SAM 3.
6. Generate pixel-level segmentation masks.
7. Convert SAM 3 results into `sv.Detections`.
8. Inspect individual segmentation masks.
9. Calculate mask pixel areas.
10. Measure the percentage of the image occupied by a mask.
11. Extract an object using its segmentation mask.
12. Compare mask area with bounding-box area.
13. Encode segmentation masks using NumPy packbits and Base64.
14. Store detection and segmentation information in JSON.
15. Decode a stored mask.
16. Verify that the reconstructed mask matches the original mask.

---

## Technologies Used

- Python
- Ultralytics
- YOLOv8
- SAM 3
- Supervision
- NumPy
- OpenCV
- Matplotlib
- JSON
- Base64
- Google Colab
- NVIDIA T4 GPU

---

## Input Image

The validated practical uses:

    assets/input/bus.jpg

Image dimensions:

    1080 × 810 pixels

Image array shape:

    (1080, 810, 3)

The image contains multiple people and a bus, making it useful for testing detection followed by segmentation.

---

## YOLO + SAM 3 Pipeline

YOLOv8 is responsible for detecting objects and generating bounding boxes.

SAM 3 receives those bounding boxes as spatial prompts and generates precise segmentation masks.

Conceptually:

    bus.jpg
       ↓
    YOLOv8
       ↓
    Object Detections
       ↓
    Bounding Boxes
       ↓
    SAM 3
       ↓
    Pixel-Level Masks

This combines the strengths of object detection and segmentation in a single workflow.

YOLO provides the approximate spatial location of each object, while SAM 3 refines that information into a pixel-level representation of the object's shape.

---

## YOLO Detection Results

During the validated execution, YOLO detected:

    5 objects

The detections consisted of:

    4 persons
    1 bus

The five detected bounding boxes were then passed to SAM 3 as prompts.

---

## SAM 3 Model

The practical uses the official SAM 3 model checkpoint:

    sam3.pt

Because the model file is approximately **3.21 GB**, it is intentionally not stored inside this GitHub repository.

For the validated Google Colab execution, the model was stored permanently in Google Drive at:

    /content/drive/MyDrive/SAM3-Models/sam3.pt

The Python practical verifies that the model exists before attempting to load it.

If Google Drive has not been mounted in a new Colab runtime, it must first be mounted before running the practical.

---

## SAM 3 Segmentation Results

SAM 3 successfully generated:

    5 segmentation masks

The resulting mask array had the shape:

    (5, 1080, 810)

This means that one segmentation mask was generated for each YOLO bounding-box prompt.

---

## Segmentation Masks

The generated masks are represented as boolean NumPy arrays.

Each pixel contains:

- `True` if the pixel belongs to the segmented object
- `False` if the pixel belongs to the background

This makes it possible to perform operations such as:

- Measuring object area
- Removing the background
- Extracting individual objects
- Measuring image coverage
- Comparing mask area with bounding-box area
- Saving segmentation results for later analysis

---

## First Mask Analysis

The first generated mask produced the following validated results:

    Mask shape: (1080, 810)
    Object pixels: 263695
    Total pixels: 874800
    Image occupied by first segmented object: 30.14%

The mask contains boolean values:

    False
    True

The first segmented object therefore occupies approximately **30.14% of the complete image**.

---

## Mask Area vs. Bounding-Box Area

A bounding box includes rectangular space surrounding an object.

A segmentation mask represents only the pixels classified as belonging to that object.

The practical compares these two measurements.

Validated results:

| Object | Mask Area | Bounding Box Area | Mask / Box |
|---|---:|---:|---:|
| 0 | 263,695 px | 408,166.97 px | 64.60% |
| 1 | 46,598 px | 98,146.27 px | 47.48% |
| 2 | 20,928 px | 68,870.01 px | 30.39% |
| 3 | 32,967 px | 59,097.94 px | 55.78% |
| 4 | 10,071 px | 20,156.73 px | 49.96% |

These results demonstrate why segmentation provides more precise spatial information than bounding boxes alone.

---

## Object Extraction

The first SAM 3 mask is applied directly to the original image.

Pixels outside the segmentation mask are set to zero.

This produces:

    assets/output/segmented_object.png

The resulting image isolates the segmented object from the surrounding background.

---

## Raw Mask Visualization

The first segmentation mask is also exported as:

    assets/output/raw_mask.png

This provides a direct visualization of the binary pixel-level mask produced by SAM 3.

---

## Mask Serialization

NumPy boolean arrays cannot be stored directly inside JSON.

The practical therefore serializes each segmentation mask using the following process:

1. Flatten the boolean mask.
2. Pack the binary values using `np.packbits`.
3. Convert the packed NumPy array to bytes.
4. Encode the bytes using Base64.
5. Store the encoded string inside JSON.

This provides a compact way to preserve segmentation masks as structured data.

---

## JSON Export

The practical generates:

    assets/output/segmentation_results.json

The JSON output contains:

- Input image filename
- SAM 3 model path
- YOLO bounding boxes
- Detection confidence scores
- Class IDs
- Class names
- Segmentation mask dimensions
- Base64-encoded masks
- Mask areas
- Bounding-box areas
- Mask-to-box percentages

This makes the segmentation results reusable for later analysis.

---

## Mask Decoding Validation

The practical does not only encode the segmentation masks.

It also verifies that the stored representation can be reconstructed correctly.

The first mask is:

1. Decoded from Base64.
2. Converted back into a NumPy byte array.
3. Unpacked using `np.unpackbits`.
4. Reshaped to its original dimensions.
5. Converted back to a boolean mask.
6. Compared with the original SAM 3 mask.

Validated result:

    Decoded mask matches original: True

This confirms that the serialization and reconstruction process preserves the segmentation mask correctly.

---

## Generated Outputs

The validated practical generated:

    raw_mask.png
    segmented_object.png
    segmentation_results.json

These files are stored inside:

    assets/output/

The existing `README.md` in that directory documents the generated results in greater detail.

---

## Practical Structure

    practical/
    ├── README.md
    ├── segmentation_with_sam.py
    └── assets/
        ├── input/
        │   ├── README.md
        │   └── bus.jpg
        │
        └── output/
            ├── README.md
            ├── raw_mask.png
            ├── segmented_object.png
            └── segmentation_results.json

The `input/` directory stores the original media used by the practical.

The `output/` directory stores generated segmentation results.

The Python implementation contains the reproducible version of the workflow demonstrated during the session.

---

## Reproducibility

The practical keeps source media and generated results separate.

This organization prevents generated files from overwriting the original input and makes the experiment easier to inspect, reproduce, and document.

To reproduce the validated workflow:

1. Use a Python environment with the required dependencies.
2. Make the SAM 3 model checkpoint available.
3. Mount Google Drive when using the validated Colab configuration.
4. Run `segmentation_with_sam.py`.
5. Inspect the generated files inside `assets/output/`.
6. Confirm that mask decoding returns `True`.

---

## Final Validation

The complete practical execution produced:

    YOLO detections: 5
    SAM masks generated: 5
    Mask array shape: (5, 1080, 810)
    Decoded mask matches original: True

Final execution status:

    Segmentation practical completed successfully.

---

## Learning Outcome

After completing this practical, I understand how to connect an object detector with a segmentation model and work directly with pixel-level segmentation data.

The key concept is that **YOLO provides the approximate location of an object through a bounding box, while SAM 3 converts that spatial prompt into a much more precise representation of the object's actual shape**.

I also practiced how to:

- Analyze segmentation masks as NumPy arrays
- Measure pixel-level object areas
- Compare masks with bounding boxes
- Extract segmented objects
- Serialize masks for structured storage
- Reconstruct encoded masks
- Validate that stored segmentation data can be recovered correctly

This practical establishes the foundation for more advanced workflows involving segmentation, tracking, video analysis, and structured computer vision pipelines.
