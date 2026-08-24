# 06 — Segmentation with SAM

This session introduces **image segmentation with Segment Anything Model 3 (SAM 3)** and demonstrates how object detection and segmentation can work together in a complete computer vision pipeline.

The main workflow combines:

- **YOLOv8** for object detection
- **SAM 3** for precise pixel-level object segmentation
- **Supervision** for handling detections consistently
- **NumPy** for working directly with segmentation masks
- **OpenCV** for image processing
- **Matplotlib** for mask visualization

The goal is to move beyond rectangular bounding boxes and begin working with **pixel-level object masks**.

The practical implementation for this session was successfully executed and validated in **Google Colab using an NVIDIA T4 GPU**.

---

## Session Objective

The objective of this session is to understand how segmentation differs from object detection and how SAM 3 can generate precise masks using object locations provided by YOLO.

The workflow is:

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
         ↓
    Object Extraction
         ↓
    JSON Serialization
         ↓
    Validation

This allows us to detect an object first and then determine exactly which pixels belong to that object.

---

## What I Learned

During this session, I learned how to:

- Understand the difference between a bounding box and a segmentation mask
- Represent segmentation masks as NumPy boolean arrays
- Detect objects using YOLOv8
- Convert YOLO results into `sv.Detections`
- Use YOLO bounding boxes as prompts for SAM 3
- Generate object masks with SAM 3
- Convert SAM 3 results into `sv.Detections`
- Inspect the shape and content of segmentation masks
- Calculate the number of pixels belonging to an object
- Measure the percentage of an image occupied by an object
- Extract an object from an image using its mask
- Compare mask area with bounding-box area
- Understand how SAM 3 can also use text prompts
- Serialize segmentation masks for storage in JSON
- Encode masks efficiently using NumPy `packbits` and Base64
- Decode serialized masks
- Validate reconstructed masks against the original segmentation data

---

# 1. Bounding Box vs. Segmentation Mask

A **bounding box** represents an object using a rectangle.

Bounding boxes are useful because they are simple and fast to calculate.

However, they also contain background pixels that do not actually belong to the object.

A **segmentation mask** instead identifies the exact pixels belonging to the object.

The mask follows the object's actual shape instead of surrounding it with a rectangle.

Conceptually:

    Bounding Box
         ↓
    Approximate Object Location

while:

    Segmentation Mask
         ↓
    Exact Object Pixels

This additional precision makes segmentation useful for more detailed computer vision analysis.

---

# 2. Segmentation Masks as Data

A segmentation mask can be represented as a **boolean NumPy array**.

Each pixel contains one of two values:

    True  → pixel belongs to the object
    False → pixel belongs to the background

For an image containing three segmented objects, the mask array could have the following shape:

    (3, 480, 640)

This means:

    3   → number of objects
    480 → image height
    640 → image width

Each object therefore has its own two-dimensional mask.

---

## Accessing One Mask

The first object's mask can be accessed with:

    mask = detections.mask[0]

Its shape would be:

    (height, width)

The mask contains one boolean value for every pixel in the original image.

---

# 3. Working with Masks Using NumPy

Because segmentation masks are NumPy arrays, standard NumPy operations can be used directly.

Example:

    mask = detections.mask[0]

    area = mask.sum()

    fraction = area / mask.size

`mask.sum()` counts the number of `True` pixels.

This gives the object's area measured in pixels.

The percentage of the entire image occupied by the object can then be calculated using:

    percentage = fraction * 100

This makes segmentation useful not only for visualization but also for quantitative analysis.

---

## Removing the Background

A segmentation mask can be used as a template to isolate an object.

Example:

    cropped_object = image.copy()

    cropped_object[~mask] = 0

The `~` operator inverts the boolean mask.

Pixels that do **not** belong to the object are changed to black.

The result is the segmented object with its surrounding background removed.

---

# 4. Detecting Objects with YOLO

The first stage of the practical pipeline uses YOLOv8.

Example:

    from ultralytics import YOLO

    yolo_model = YOLO("yolov8n.pt")

    yolo_results = yolo_model(image)[0]

The YOLO result is then converted into a Supervision detections object:

    import supervision as sv

    yolo_detections = sv.Detections.from_ultralytics(
        yolo_results
    )

YOLO provides the bounding boxes that will later be used as prompts for SAM 3.

---

# 5. YOLO Bounding Boxes as SAM Prompts

SAM needs information indicating where it should look for an object.

In this workflow, the bounding boxes generated by YOLO are used as those prompts.

The coordinates can be obtained from:

    yolo_detections.xyxy

SAM expects a standard Python list, so the NumPy array is converted using:

    bboxes = yolo_detections.xyxy.tolist()

The resulting bounding boxes are then passed to SAM 3.

Conceptually:

    YOLO Detection
         ↓
    Bounding Box
         ↓
    SAM 3 Prompt
         ↓
    Segmentation Mask

---

# 6. Loading SAM 3

SAM 3 is loaded using the Ultralytics interface:

    from ultralytics import SAM

    sam_model = SAM(sam_path)

The session uses the SAM 3 model checkpoint:

    sam3.pt

The official SAM 3 model weights are not stored directly inside this GitHub repository because the checkpoint is approximately **3.21 GB**.

For the validated Google Colab practical, the model was stored permanently in Google Drive at:

    /content/drive/MyDrive/SAM3-Models/sam3.pt

The practical verifies that this model exists before attempting to run segmentation.

---

# 7. Generating Segmentation Masks

Once the YOLO bounding boxes are available, they can be used as prompts for SAM 3:

    sam_results = sam_model(
        image,
        bboxes=bboxes
    )[0]

SAM 3 then generates precise segmentation masks for the objects located inside those boxes.

During the validated practical execution:

    YOLO detections: 5
    SAM masks generated: 5

This produced one SAM 3 mask for each YOLO bounding-box prompt.

---

# 8. Converting SAM Results to Supervision

One of the important concepts demonstrated in this session is that Supervision can work with outputs from different computer vision models.

YOLO results can be converted using:

    sv.Detections.from_ultralytics(
        yolo_results
    )

SAM 3 results can be converted using the same method:

    sam_detections = sv.Detections.from_ultralytics(
        sam_results
    )

This demonstrates the framework-agnostic design of Supervision.

After conversion, the masks are available through:

    sam_detections.mask

---

# 9. Inspecting a Segmentation Mask

The first mask can be selected with:

    first_mask = sam_detections.mask[0]

Useful information can then be inspected:

    print(type(first_mask))
    print(first_mask.shape)
    print(np.unique(first_mask))
    print(first_mask.sum())
    print(first_mask.size)

This reveals:

- Data type
- Mask dimensions
- Unique mask values
- Number of object pixels
- Total number of pixels

For a boolean segmentation mask, the unique values represent:

    False
    True

---

# 10. Validated First Mask Results

The practical was executed using:

    Input image: bus.jpg
    Image shape: (1080, 810, 3)

SAM 3 generated a mask array with the shape:

    (5, 1080, 810)

The first mask produced:

    Mask shape: (1080, 810)
    Object pixels: 263695
    Total pixels: 874800
    Image occupied by first segmented object: 30.14%

This means the first segmented object occupies approximately **30.14% of the complete image**.

---

# 11. Raw Mask Visualization

The practical visualizes the first segmentation mask directly using Matplotlib.

Example:

    plt.imshow(
        first_mask,
        cmap="gray"
    )

    plt.title(
        "Raw Segmentation Mask"
    )

    plt.axis(
        "off"
    )

The validated practical saves this visualization as:

    practical/assets/output/raw_mask.png

Displaying the mask directly makes it easier to understand the underlying segmentation data before applying more advanced visual annotations.

---

# 12. Extracting an Object with Pixel Precision

The segmentation mask can be applied directly to the original image.

Example:

    object_crop = image.copy()

    object_crop[~first_mask] = 0

This removes all pixels outside the mask.

The resulting image contains only the segmented object.

The validated practical saves the extracted object as:

    practical/assets/output/segmented_object.png

This demonstrates the main advantage of segmentation:

**Segmentation identifies the object at the individual-pixel level instead of only locating it with a rectangle.**

---

# 13. Mask Area vs. Bounding-Box Area

The practical compares each object's segmentation-mask area with its YOLO bounding-box area.

For each object:

    mask_area = mask.sum()

    box_area = box_width * box_height

    percentage = (
        mask_area / box_area
    ) * 100

This calculates the proportion of the bounding box that is actually occupied by the segmented object.

The validated results were:

| Object | Mask Area | Bounding Box Area | Mask / Box |
|---|---:|---:|---:|
| 0 | 263,695 px | 408,166.97 px | 64.60% |
| 1 | 46,598 px | 98,146.27 px | 47.48% |
| 2 | 20,928 px | 68,870.01 px | 30.39% |
| 3 | 32,967 px | 59,097.94 px | 55.78% |
| 4 | 10,071 px | 20,156.73 px | 49.96% |

A high percentage means that much of the bounding box is occupied by the actual object.

A lower percentage means that the bounding box contains more background.

This demonstrates how segmentation allows more precise geometric analysis than bounding boxes alone.

---

# 14. Text Prompts with SAM 3

SAM 3 can also work with **text prompts**.

Instead of first detecting an object with YOLO and then passing its bounding box to SAM, it is possible to request a semantic concept such as:

    person

Conceptually:

    Image
      ↓
    Text Prompt: "person"
      ↓
    SAM 3
      ↓
    Person Segmentation Masks

This session introduces this capability as a preview.

Later experimentation can explore:

- Multiple concepts
- Confidence thresholds
- Text prompts vs. bounding-box prompts
- Prompt-based segmentation workflows

---

# 15. Saving Segmentation Masks

Segmentation masks are NumPy boolean arrays.

They cannot be written directly into standard JSON without first converting them into a JSON-compatible representation.

The session demonstrates storing masks by:

1. Flattening the mask
2. Compressing the boolean values with `np.packbits`
3. Converting the packed bytes to Base64
4. Saving the Base64 text inside JSON

Example:

    encoded_mask = base64.b64encode(
        np.packbits(
            mask.flatten()
        ).tobytes()
    ).decode("utf-8")

This creates a compact text representation of the segmentation mask.

---

# 16. JSON Detection Structure

The practical stores information including:

- `input_image`
- `sam_model`
- `xyxy`
- `confidence`
- `class_id`
- `class_names`
- `mask_shape`
- `masks_b64`
- `area_comparison`

Conceptual structure:

    {
        "input_image": "bus.jpg",
        "sam_model": "...",
        "xyxy": [],
        "confidence": [],
        "class_id": [],
        "class_names": [],
        "mask_shape": [1080, 810],
        "masks_b64": [],
        "area_comparison": []
    }

The validated results are stored in:

    practical/assets/output/segmentation_results.json

This allows detection information, segmentation masks, and geometric measurements to be stored together.

---

# 17. Recovering a Stored Mask

The Base64 representation can later be decoded.

The process demonstrated in the practical is:

    raw = np.frombuffer(
        base64.b64decode(
            encoded_mask
        ),
        dtype=np.uint8
    )

    mask = np.unpackbits(
        raw
    )[:H * W]

    mask = mask.reshape(
        H,
        W
    ).astype(bool)

This reconstructs the original boolean segmentation mask.

---

# 18. Mask Decoding Validation

The practical verifies that serialization does not destroy the segmentation data.

The reconstructed mask is compared directly with the original mask using:

    np.array_equal(
        first_mask,
        decoded_mask
    )

The validated result was:

    Decoded mask matches original: True

This confirms that the Base64 serialization and reconstruction process successfully preserves the original segmentation mask.

---

# 19. Why Segmentation Matters

Object detection answers:

**Where is the object?**

Segmentation answers:

**Which exact pixels belong to the object?**

That additional precision makes segmentation useful for tasks such as:

- Object extraction
- Background removal
- Pixel-level measurements
- Shape analysis
- Dataset annotation
- Instance segmentation
- Precise visual analytics
- Object-area calculations
- Mask-based computer vision pipelines

---

# 20. Practical Implementation

The reproducible implementation for this session is located in:

    practical/segmentation_with_sam.py

The practical uses:

    practical/assets/input/bus.jpg

and generates:

    practical/assets/output/raw_mask.png
    practical/assets/output/segmented_object.png
    practical/assets/output/segmentation_results.json

The practical was successfully executed using:

- Google Colab
- Python
- NVIDIA T4 GPU
- Ultralytics
- YOLOv8
- SAM 3
- Supervision
- OpenCV
- NumPy
- Matplotlib

---

# 21. Validated Practical Results

YOLOv8 detected:

    5 objects

consisting of:

    4 persons
    1 bus

SAM 3 generated:

    5 segmentation masks

Mask array shape:

    (5, 1080, 810)

First-mask analysis:

    Object pixels: 263695
    Total pixels: 874800
    Image coverage: 30.14%

Mask serialization validation:

    Decoded mask matches original: True

Final execution status:

    Segmentation practical completed successfully.

---

# Key Concepts

| Concept | Description |
|---|---|
| Bounding Box | Rectangle surrounding an object |
| Segmentation Mask | Pixel-level representation of an object |
| Boolean Mask | NumPy array containing `True` and `False` values |
| YOLOv8 | Detects objects and produces bounding boxes |
| SAM 3 | Generates precise object segmentation masks |
| Bounding Box Prompt | Bounding-box coordinates supplied to SAM 3 |
| Text Prompt | Semantic text supplied to SAM 3 |
| `sv.Detections` | Unified Supervision representation for detections |
| Mask Area | Number of pixels belonging to an object |
| Image Coverage | Percentage of the image occupied by a mask |
| `np.packbits` | Packs boolean values into a compact binary representation |
| Base64 | Text encoding used to store packed mask data in JSON |
| Mask Validation | Comparison between reconstructed and original masks |

---

# Main Pipeline

    Image
      ↓
    YOLOv8
      ↓
    Bounding Boxes / Classes / Confidence Scores
      ↓
    Supervision sv.Detections
      ↓
    Bounding Box Prompts
      ↓
    SAM 3
      ↓
    Segmentation Masks
      ↓
    Supervision sv.Detections
      ↓
    Mask Analysis
      ↓
    Object Extraction
      ↓
    Area Calculation
      ↓
    Base64 Serialization
      ↓
    JSON Export
      ↓
    Mask Reconstruction
      ↓
    Validation

---

# Session Files

    06-Segmentation-with-SAM/
    ├── README.md
    ├── CLASS-RECORDING.md
    └── practical/
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

---

# Session Takeaway

The main lesson from this session is that **detection and segmentation solve different but complementary problems**.

YOLOv8 provides efficient object localization through bounding boxes, while SAM 3 transforms those approximate locations into precise pixel-level masks.

Supervision provides a common representation for both models, making it possible to build a pipeline where outputs from different computer vision frameworks can be processed using the same tools.

The completed practical also demonstrates that segmentation data can be analyzed quantitatively, serialized into structured storage, reconstructed, and validated.

This session establishes the foundation for more advanced SAM workflows involving prompts, segmentation visualization, object extraction, tracking, video analysis, and mask-based computer vision pipelines.
