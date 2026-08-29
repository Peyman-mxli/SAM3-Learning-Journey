# Ultralytics YOLOv8 — Detection, Segmentation, Pose, Classification, and Tracking

YOLOv8 is a computer vision model family released by Ultralytics. In this course, the lightweight `yolov8n.pt` checkpoint supplies fast COCO-pretrained object detections that are converted into Supervision `Detections` and later combined with tracking, zones, counting, and SAM-based segmentation.

## Resource Summary

| Item | Details |
|---|---|
| Type | Real-time computer vision model family |
| Framework | Ultralytics Python package and CLI |
| Course checkpoint | `yolov8n.pt` |
| Pretraining | COCO detection dataset |
| Course association | Session 02 onward |
| Official YOLOv8 guide | <https://docs.ultralytics.com/models/yolov8/> |
| Python usage | <https://docs.ultralytics.com/usage/python/> |
| Prediction mode | <https://docs.ultralytics.com/modes/predict/> |

## What YOLO Does

Object detection answers two questions for every recognized object:

1. What class does the object belong to?
2. Where is it located in the image?

YOLO produces bounding boxes, class IDs, and confidence scores in a single inference pipeline. Different YOLOv8 checkpoint families support different tasks:

| Checkpoint pattern | Task | Output |
|---|---|---|
| `yolov8n.pt` | Detection | Boxes, classes, confidence |
| `yolov8n-seg.pt` | Instance segmentation | Boxes and per-object masks |
| `yolov8n-pose.pt` | Pose | Person boxes and keypoints |
| `yolov8n-cls.pt` | Classification | Image-level class probabilities |
| `yolov8n-obb.pt` | Oriented detection | Rotated bounding boxes |

The suffix `n` means nano. Larger variants (`s`, `m`, `l`, and `x`) generally trade more computation for greater capacity.

## Pipeline

```text
Image, video, directory, stream, or webcam
                    ↓
Preprocessing and tensor conversion
                    ↓
YOLOv8 neural network
                    ↓
Candidate predictions
                    ↓
Confidence filtering and NMS
                    ↓
Ultralytics Results object
```

## Installation

Python 3.9 or newer is recommended for this course environment.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
source .venv/bin/activate
```

Install:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Google Colab:

```python
!pip install -q ultralytics supervision opencv-python-headless
```

Verify:

```bash
yolo checks
python -c "import ultralytics; print(ultralytics.__version__)"
```

## Basic Python Inference

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.predict(
    source="bus.jpg",
    conf=0.50,
    imgsz=640,
)

result = results[0]
print(result.boxes.xyxy)
print(result.boxes.conf)
print(result.boxes.cls)
```

The first use downloads the checkpoint automatically when it is not already cached.

## Results Object

| Attribute | Content |
|---|---|
| `result.boxes.xyxy` | Bounding boxes in pixel coordinates |
| `result.boxes.conf` | Confidence values |
| `result.boxes.cls` | Numeric class IDs |
| `result.names` | Mapping from class ID to readable name |
| `result.masks` | Segmentation masks when using a segmentation checkpoint |
| `result.keypoints` | Pose keypoints when using a pose checkpoint |
| `result.orig_img` | Original image array |

Move tensors to CPU before NumPy conversion:

```python
boxes = result.boxes.xyxy.cpu().numpy()
```

## Complete Example

The included `example_yolov8_inference.py` downloads `bus.jpg`, runs `yolov8n.pt`, exports an annotated image, and writes a JSON file containing box coordinates, class IDs, class names, and confidence values.

```bash
python example_yolov8_inference.py
```

Expected outputs:

```text
assets/output/bus_yolov8.jpg
assets/output/bus_yolov8_detections.json
```

## Verified Runtime Output

The example was executed successfully with `yolov8n.pt` at confidence 0.50. Four detections were retained: one bus and three people.

![Verified YOLOv8 detection output](./assets/output/bus_yolov8.jpg)

- [Structured detection results](./assets/output/bus_yolov8_detections.json)

## Command-Line Usage

```bash
yolo detect predict model=yolov8n.pt source=bus.jpg conf=0.50 save=True
```

Video:

```bash
yolo detect predict model=yolov8n.pt source=vehicles.mp4 conf=0.50 save=True
```

Webcam:

```bash
yolo detect predict model=yolov8n.pt source=0 show=True
```

## Important Prediction Parameters

| Parameter | Meaning |
|---|---|
| `conf` | Minimum confidence threshold |
| `iou` | IoU threshold used by NMS |
| `imgsz` | Inference image size |
| `device` | CPU or GPU device selection |
| `classes` | Class IDs to retain |
| `max_det` | Maximum detections per image |
| `stream=True` | Memory-efficient generator for long inputs |
| `save=True` | Save rendered predictions |

## YOLOv8 and Supervision

```python
import supervision as sv

detections = sv.Detections.from_ultralytics(result)
detections = detections[detections.confidence >= 0.50]
```

After conversion, the course uses Supervision for filtering, annotation, ByteTrack IDs, zones, counting, and video processing.

## Training a Custom Model

Dataset configuration:

```yaml
path: /path/to/dataset
train: images/train
val: images/val
names:
  0: person
  1: helmet
```

Python:

```python
model = YOLO("yolov8n.pt")
model.train(data="dataset.yaml", epochs=50, imgsz=640)
```

Training requires correctly paired images and labels, representative train/validation splits, and evaluation on data that was not used for optimization.

## Evaluation Terms

| Metric | Meaning |
|---|---|
| Precision | Fraction of predictions that are correct |
| Recall | Fraction of ground-truth objects found |
| IoU | Overlap between predicted and reference regions |
| mAP50 | Mean average precision at IoU 0.50 |
| mAP50–95 | Mean AP across stricter IoU thresholds |

Confidence is not accuracy. A confidence value is the model's score for one prediction; model quality requires dataset-level evaluation.

## Common Problems

| Problem | Resolution |
|---|---|
| CUDA unavailable | Run on CPU or install a compatible PyTorch/CUDA environment |
| Model downloads repeatedly | Use a stable writable cache or explicit checkpoint path |
| No detections | Lower `conf`, confirm classes, inspect image scale and domain shift |
| Too many false positives | Raise `conf`, tune NMS IoU, or train on representative data |
| Out-of-memory error | Use a smaller model, smaller `imgsz`, or smaller batch |
| Old code fails | Match examples to the installed Ultralytics version |

## Professional Practices

- Pin versions for reproducible assignments.
- Record checkpoint, threshold, image size, device, and library version.
- Keep raw predictions separate from visualization.
- Use `stream=True` for long videos.
- Evaluate on labeled data instead of judging only rendered images.
- Review the Ultralytics license terms for the intended deployment.

## Official References

- YOLOv8 model guide: <https://docs.ultralytics.com/models/yolov8/>
- Prediction: <https://docs.ultralytics.com/modes/predict/>
- Python API: <https://docs.ultralytics.com/usage/python/>
- CLI: <https://docs.ultralytics.com/usage/cli/>
- Configuration: <https://docs.ultralytics.com/usage/cfg/>
