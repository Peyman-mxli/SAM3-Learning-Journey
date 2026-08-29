# Supervision — Professional Computer Vision Utilities

Supervision is an open-source Python library maintained by Roboflow for building model-agnostic computer vision pipelines. It does not replace an inference model such as YOLO or SAM. Instead, it provides a consistent layer for representing detections, filtering results, annotating images and video, tracking objects, defining zones, counting events, evaluating predictions, and managing visual outputs.

## Resource Summary

| Item | Details |
|---|---|
| Category | Computer vision utility library |
| Maintainer | Roboflow |
| Language | Python |
| License | MIT |
| Stable documentation | <https://supervision.roboflow.com/> |
| Annotator catalog | <https://supervision.roboflow.com/annotators/> |
| Source repository | <https://github.com/roboflow/supervision> |
| Package | <https://pypi.org/project/supervision/> |
| Course use | Detection conversion, filtering, annotation, tracking, zones, counting, segmentation visualization, and video processing |

## What Problem Does Supervision Solve?

Computer vision models return predictions in different formats. One model may return PyTorch tensors, another NumPy arrays, and another framework-specific objects. Building the same visualization, filtering, tracking, or counting logic separately for every model creates unnecessary complexity.

Supervision provides a common representation called `sv.Detections`. Once predictions have been converted to this format, the rest of the application can use the same operations regardless of which supported model produced them.

```text
Image or video frame
        ↓
Inference model (YOLO, SAM, Transformers, etc.)
        ↓
sv.Detections
        ↓
Filter · Track · Count · Analyze
        ↓
Annotate and export results
```

Supervision is therefore best understood as the application layer around a model:

- The model performs inference.
- Supervision organizes and processes the predictions.
- OpenCV or another media library reads and writes images or video.
- Your application defines the final rules and analytics.

## Core Concepts

### `sv.Detections`

`sv.Detections` is the central data structure. Depending on the task, it can contain:

| Attribute | Meaning |
|---|---|
| `xyxy` | Bounding boxes in `(x1, y1, x2, y2)` format |
| `confidence` | Confidence score for each prediction |
| `class_id` | Numeric class identifier |
| `mask` | Optional instance-segmentation masks |
| `tracker_id` | Optional persistent identifier assigned by a tracker |
| `data` | Additional fields such as class names |

Because these arrays describe the same detections, filtering must preserve their alignment. Supervision handles that relationship when Boolean or index-based filtering is applied to a `Detections` object.

### Annotators

Annotators draw structured results on an image. Common choices include:

- `sv.BoxAnnotator` for bounding boxes.
- `sv.LabelAnnotator` for class names, confidence values, or tracker IDs.
- `sv.MaskAnnotator` for segmentation masks.
- `sv.TraceAnnotator` for movement paths.
- `sv.PolygonZoneAnnotator` and `sv.LineZoneAnnotator` for analytical regions.

Annotators receive an image and a `Detections` object. Multiple annotators can be applied sequentially to the same scene.

### Tracking

Detection identifies objects independently in each frame. Tracking attempts to preserve an identity across frames. In this course, ByteTrack is used so detections can receive persistent `tracker_id` values. Those IDs enable trajectories, crossing counts, dwell-time analysis, and per-object statistics.

### Zones and Counting

Supervision can define spatial rules such as:

- Is an object inside a polygon?
- Did an object cross a line?
- In which direction did it cross?
- How many tracked objects occupy a region?

These operations transform raw detections into measurements that are useful in real applications.

## Requirements

The current stable documentation requires Python 3.9 or newer. A virtual environment is recommended to prevent dependency conflicts.

Check Python:

```bash
python --version
```

## Installation

### Standard environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install the packages used by the example:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Google Colab

```python
!pip install -q supervision ultralytics opencv-python-headless
```

After installation, restart the runtime only if Colab reports dependency changes that require it.

### Verify the installation

```bash
python -c "import supervision as sv; print(sv.__version__)"
```

## Complete Example — YOLO Detection and Annotation

The included script downloads the official Ultralytics sample image, runs YOLO, converts its predictions to `sv.Detections`, filters low-confidence results, creates labels, draws boxes and labels, and saves the finished image.

Run it from this directory:

```bash
python example_detect_and_annotate.py
```

Expected output:

```text
assets/input/bus.jpg
assets/output/bus_annotated.jpg
```

The essential integration is:

```python
result = model(image)[0]
detections = sv.Detections.from_ultralytics(result)
detections = detections[detections.confidence >= 0.50]

annotated = box_annotator.annotate(
    scene=image.copy(),
    detections=detections,
)
annotated = label_annotator.annotate(
    scene=annotated,
    detections=detections,
    labels=labels,
)
```

### What happens step by step?

1. OpenCV loads the image as a NumPy array.
2. YOLO performs inference and returns an Ultralytics result.
3. `sv.Detections.from_ultralytics()` converts the result into the common Supervision format.
4. A confidence condition filters uncertain detections.
5. A label is generated for every remaining detection.
6. `BoxAnnotator` draws the bounding boxes.
7. `LabelAnnotator` adds the class names and confidence scores.
8. OpenCV saves the final visualization.

## Using Supervision Without YOLO

Supervision can also receive detections created manually:

```python
import numpy as np
import supervision as sv

detections = sv.Detections(
    xyxy=np.array([
        [50, 40, 240, 300],
        [280, 80, 500, 360],
    ], dtype=np.float32),
    confidence=np.array([0.94, 0.87], dtype=np.float32),
    class_id=np.array([0, 1]),
)
```

This is useful when integrating a custom detector or loading predictions from a database or annotation file.

## Filtering Examples

Filter by confidence:

```python
detections = detections[detections.confidence >= 0.50]
```

Keep only the COCO `person` class (`class_id == 0`):

```python
detections = detections[detections.class_id == 0]
```

Filter by bounding-box area:

```python
widths = detections.xyxy[:, 2] - detections.xyxy[:, 0]
heights = detections.xyxy[:, 3] - detections.xyxy[:, 1]
areas = widths * heights
detections = detections[areas >= 5_000]
```

Combine conditions:

```python
keep = (detections.confidence >= 0.50) & (detections.class_id == 0)
detections = detections[keep]
```

## Segmentation Masks

When the source model returns instance masks, the `mask` attribute can be rendered with `MaskAnnotator`:

```python
mask_annotator = sv.MaskAnnotator(opacity=0.55)
annotated = mask_annotator.annotate(
    scene=image.copy(),
    detections=detections,
)
```

This is especially relevant to SAM and SAM 3 workflows because masks can be combined with bounding boxes, labels, tracker IDs, and temporal analytics.

## Typical Course Pipeline

```python
import supervision as sv

tracker = sv.ByteTrack()
detections = sv.Detections.from_ultralytics(result)
detections = tracker.update_with_detections(detections)
```

After tracking, each matching detection can carry a persistent tracker ID. The IDs can then be displayed or used for counting and historical analysis.

## Good Practices

- Pin package versions for reproducible assignments.
- Confirm the installed version before copying examples from a different documentation version.
- Keep the original image unchanged and annotate a copy.
- Apply filtering before expensive annotation or segmentation stages.
- Validate that label count equals detection count.
- Check that an image loaded successfully before inference.
- Create output directories programmatically.
- Save machine-readable results in addition to visualization images.
- Release video readers and writers safely when processing video.

## Common Problems

| Problem | Likely cause | Recommended action |
|---|---|---|
| `ModuleNotFoundError: supervision` | Package is not installed in the active environment | Activate the correct environment and run `python -m pip install supervision` |
| `AttributeError` for an annotator | Code and installed API versions differ | Check `sv.__version__` and use the documentation for that version |
| Image is `None` | Incorrect file path or failed download | Verify the path and check `cv2.imread()` before inference |
| Labels do not match boxes | Labels were created before filtering | Generate labels after applying all detection filters |
| Video IDs constantly change | Tracking state is recreated every frame | Instantiate the tracker once, outside the frame loop |
| Colab display fails | `cv2.imshow()` is unavailable in Colab | Use `cv2_imshow`, Matplotlib, or save the output file |

## Relationship to SAM 3

SAM 3 produces segmentation information; Supervision helps turn that information into a complete application. In this learning journey it is used to:

- Represent bounding boxes and masks consistently.
- Transfer detection metadata to segmentation results.
- Draw masks, boxes, labels, and traces.
- Maintain tracker IDs across video frames.
- Filter detections using confidence, class, area, or position.
- Count objects in regions and across lines.
- Generate understandable output media and analytics.

## Course Mapping

Supervision appears throughout the course rather than in only one isolated lesson. Its introductory documentation is associated with Session 02 (`01_a_introduccion_supervision`), while later sessions extend it through annotators, filtering, tracking, zones, counting, and segmentation.

## Official References

- Documentation: <https://supervision.roboflow.com/>
- Annotator catalog: <https://supervision.roboflow.com/annotators/>
- Detect and annotate guide: <https://supervision.roboflow.com/how_to/detect_and_annotate/>
- GitHub repository: <https://github.com/roboflow/supervision>
- PyPI package: <https://pypi.org/project/supervision/>

## Next Step

After the basic example works, experiment with one controlled change at a time: increase the confidence threshold, keep only people, switch from boxes to masks, or process a short video with ByteTrack.
