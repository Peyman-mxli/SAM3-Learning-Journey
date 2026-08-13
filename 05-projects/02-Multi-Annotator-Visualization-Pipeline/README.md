# Multi-Annotator Visualization Pipeline

## Overview

This project demonstrates how to build a reusable computer vision visualization pipeline using:

- YOLOv8
- Ultralytics
- Supervision
- OpenCV

The application detects objects in an input image and applies multiple Supervision Annotators as visual layers.

The final annotated result is saved automatically inside the `output/` directory.

---

## Project Goals

This project was created to practice the main concepts from the **Annotation and Visualization** lesson.

It demonstrates how to:

- Load an image with OpenCV
- Detect objects with YOLOv8
- Convert YOLO results into `sv.Detections`
- Use a confidence threshold
- Create labels containing class names and confidence scores
- Apply multiple Annotators in sequence
- Use annotation layers
- Save the final visualization

---

## Project Structure

```text
02-Multi-Annotator-Visualization-Pipeline/
│
├── input/
│   └── image.jpg
│
├── output/
│   └── annotated_image.jpg
│
├── multi_annotator_pipeline.py
├── requirements.txt
└── README.md
```

---

## Visualization Pipeline

The application follows this workflow:

```text
input/image.jpg
      ↓
OpenCV
      ↓
YOLOv8
      ↓
Detection Results
      ↓
sv.Detections
      ↓
BoxAnnotator
      ↓
EllipseAnnotator
      ↓
DotAnnotator
      ↓
LabelAnnotator
      ↓
output/annotated_image.jpg
```

Each Annotator receives the result of the previous layer.

This creates a multi-layer visualization pipeline.

---

## Main Python File

The main application is:

```text
multi_annotator_pipeline.py
```

The script contains the complete detection and annotation workflow.

---

## Configuration

The main configuration values are:

```python
MODEL_NAME = "yolov8n.pt"

INPUT_IMAGE = "input/image.jpg"
OUTPUT_IMAGE = "output/annotated_image.jpg"

CONFIDENCE_THRESHOLD = 0.50
```

### Model

```python
MODEL_NAME = "yolov8n.pt"
```

This project uses the YOLOv8 Nano model.

### Confidence Threshold

```python
CONFIDENCE_THRESHOLD = 0.50
```

Only detections with sufficient confidence are included in the final visualization.

---

## Automatic Directory Creation

The script creates the required folders automatically:

```python
Path("input").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
```

This ensures that the expected directory structure exists before the pipeline runs.

---

## Loading the Image

The image is loaded using OpenCV:

```python
image = cv2.imread(INPUT_IMAGE)
```

The script also verifies that the image was loaded correctly:

```python
if image is None:
    raise FileNotFoundError(
        f"Could not load image: {INPUT_IMAGE}"
    )
```

This prevents the rest of the pipeline from running with an invalid input.

---

## Loading YOLO

The model is initialized using:

```python
model = YOLO(MODEL_NAME)
```

YOLO performs the object detection stage of the pipeline.

---

## Running Object Detection

The model is executed using:

```python
results = model(
    image,
    conf=CONFIDENCE_THRESHOLD
)[0]
```

The confidence threshold is applied directly during inference.

---

## Converting YOLO Results to Supervision

YOLO results are converted into a Supervision `Detections` object:

```python
detections = sv.Detections.from_ultralytics(
    results
)
```

This provides a structured representation of the detected objects.

The detections may contain information such as:

- Bounding box coordinates
- Class IDs
- Confidence scores

---

## Creating Detection Labels

Labels are created using the detected class and confidence score:

```python
labels = [
    f"{results.names[class_id]} {confidence:.0%}"
    for class_id, confidence in zip(
        detections.class_id,
        detections.confidence
    )
]
```

Example labels may look like:

```text
person 91%
bus 86%
stop sign 78%
```

---

## Annotators

The project uses four different Supervision Annotators.

### BoxAnnotator

```python
box_annotator = sv.BoxAnnotator(
    color=sv.ColorPalette.DEFAULT,
    thickness=3
)
```

This draws bounding boxes around the detected objects.

`ColorPalette.DEFAULT` allows Supervision to use different colors automatically.

---

### EllipseAnnotator

```python
ellipse_annotator = sv.EllipseAnnotator()
```

This adds an ellipse-based visualization around detections.

---

### DotAnnotator

```python
dot_annotator = sv.DotAnnotator()
```

This adds detection points as another visual layer.

---

### LabelAnnotator

```python
label_annotator = sv.LabelAnnotator(
    text_scale=0.6
)
```

This displays the object class and confidence information.

The labels are deliberately applied last so they remain visually readable.

---

## Annotation Layer Order

The project applies the Annotators in this order:

```text
1. BoxAnnotator
2. EllipseAnnotator
3. DotAnnotator
4. LabelAnnotator
```

Conceptually:

```text
Original Image
      ↓
Bounding Boxes
      ↓
Ellipses
      ↓
Detection Points
      ↓
Labels
      ↓
Final Visualization
```

The order matters because every new Annotator is drawn on top of the previous result.

---

## Saving the Result

The final image is saved using OpenCV:

```python
success = cv2.imwrite(
    OUTPUT_IMAGE,
    annotated_image
)
```

The script verifies that the file was written successfully:

```python
if not success:
    raise RuntimeError(
        f"Could not save image: {OUTPUT_IMAGE}"
    )
```

The final result is stored at:

```text
output/annotated_image.jpg
```

---

## Requirements

Install the project dependencies with:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains:

```text
ultralytics
supervision
opencv-python
```

---

## How to Run the Project

### 1. Add an Input Image

Place an image inside:

```text
input/
```

and name it:

```text
image.jpg
```

The expected path is:

```text
input/image.jpg
```

---

### 2. Install Dependencies

Run:

```bash
pip install -r requirements.txt
```

---

### 3. Run the Application

From the project directory:

```bash
python multi_annotator_pipeline.py
```

---

### 4. Check the Output

After successful execution, the result will be available at:

```text
output/annotated_image.jpg
```

The terminal will also display information such as:

```text
Detected objects: 6
Annotated image saved to: output/annotated_image.jpg
```

The exact number of detected objects depends on the input image.

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| YOLOv8 | Object detection |
| Ultralytics | YOLO model interface |
| Supervision | Detection visualization |
| OpenCV | Image loading and output |
| pathlib | Directory management |

---

## Concepts Practiced

This project reinforces several important computer vision concepts:

- Object detection
- Confidence thresholds
- Detection data structures
- Bounding boxes
- Labels
- Annotation customization
- Color palettes
- Annotation layers
- Annotator composition
- Output generation

---

## Detection vs. Visualization

The project separates two responsibilities.

### YOLO

YOLO determines:

```text
What object was detected?
Where is the object?
How confident is the prediction?
```

### Supervision

Supervision determines:

```text
How should the detection be visualized?
```

The complete relationship is:

```text
YOLO
  ↓
Detection Data
  ↓
Supervision
  ↓
Visual Representation
```

---

## Related Course Material

### Notebook

```text
03-notebooks/01_b_anotacion_visualizacion.ipynb
```

### Course Notes

```text
08-course-notes/02-Annotation-and-Visualization/
```

### Code Examples

```text
04-examples/02-Annotation-and-Visualization/
```

This project combines the lesson concepts into a single reusable Python application.

---

## Key Takeaway

The central idea of this project is that multiple Supervision Annotators can be composed as visual layers.

```text
Detection
    ↓
Layer 1
    ↓
Layer 2
    ↓
Layer 3
    ↓
Layer 4
    ↓
Final Visualization
```

This makes it possible to build custom visualization pipelines without changing the underlying YOLO detections.

---

## Author

**Peyman Miyandashti**

SAM3 Learning Journey  
Computer Vision · Artificial Intelligence · Machine Learning

[LinkedIn](https://www.linkedin.com/in/peyman-mxli/) | [GitHub](https://github.com/Peyman-mxli)
