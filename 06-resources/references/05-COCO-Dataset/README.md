# COCO Dataset — Classes, Annotations, Tasks, and Evaluation

COCO, or Common Objects in Context, is a large-scale dataset designed for object detection, segmentation, keypoint detection, and image captioning. YOLOv8 detection checkpoints used in this course are pretrained on 80 COCO object categories, so COCO class IDs appear throughout the notebooks.

## Resource Summary

| Item | Details |
|---|---|
| Full name | Common Objects in Context |
| Primary purpose | Recognition in complex, natural scenes |
| Scale | Approximately 330,000 images |
| Detection categories | 80 commonly used object categories |
| Other annotations | Instance masks, keypoints, captions, and panoptic labels |
| Course association | Session 02 onward |
| Official website | <https://cocodataset.org/> |

## Why COCO Matters in This Course

When `yolov8n.pt` predicts `class_id == 0`, it means `person` because the checkpoint uses the COCO class order. The model's class mapping—not Supervision—defines that semantic meaning.

```python
class_id = int(result.boxes.cls[0])
class_name = result.names[class_id]
```

Never assume a numeric class ID has the same meaning across custom models.

## Main Tasks

| Task | Annotation |
|---|---|
| Object detection | Bounding boxes and category IDs |
| Instance segmentation | Separate mask for each object instance |
| Keypoint detection | Human joint locations and visibility |
| Panoptic segmentation | Unified “things” and “stuff” scene labels |
| Image captioning | Natural-language scene descriptions |

## COCO JSON Structure

A standard instance-annotation JSON file contains:

```json
{
  "images": [],
  "annotations": [],
  "categories": []
}
```

### Images

```json
{
  "id": 42,
  "file_name": "000000000042.jpg",
  "width": 640,
  "height": 480
}
```

### Categories

```json
{
  "id": 1,
  "name": "person",
  "supercategory": "person"
}
```

### Detection or instance annotation

```json
{
  "id": 1001,
  "image_id": 42,
  "category_id": 1,
  "bbox": [120.0, 80.0, 160.0, 300.0],
  "area": 48000.0,
  "iscrowd": 0,
  "segmentation": []
}
```

Important: COCO JSON bounding boxes use `[x, y, width, height]`, while many inference APIs use `[x1, y1, x2, y2]`.

```python
x1 = x
y1 = y
x2 = x + width
y2 = y + height
```

## Category-ID Warning

Official COCO category IDs are not a continuous zero-based sequence. Many training frameworks remap the 80 categories to contiguous IDs `0–79`. Therefore:

- Read the dataset's category table.
- Read the model's `names` mapping.
- Do not join predictions and annotations by guessed numeric ID.
- Preserve the mapping used during export or training.

## Installation for Inspection

The included inspector uses only Python's standard library:

```bash
python inspect_coco_annotations.py /path/to/instances_val2017.json
```

Optional official-style COCO tooling is commonly installed with:

```bash
python -m pip install pycocotools
```

On platforms where compilation is difficult, use an environment with a compatible prebuilt wheel.

## Included Inspector

`inspect_coco_annotations.py` validates the top-level collections and reports:

- Number of images
- Number of annotations
- Number of categories
- Most frequent annotated categories
- Missing image or category references
- Invalid bounding-box shapes

The script reads metadata only; it does not download the full dataset.

## Evaluation

COCO-style object-detection evaluation reports average precision across IoU thresholds and object sizes.

| Term | Meaning |
|---|---|
| AP | Average precision |
| AP50 | AP at IoU 0.50 |
| AP75 | AP at IoU 0.75 |
| AP50–95 | AP averaged over IoU thresholds 0.50 to 0.95 |
| AR | Average recall |

AP50 is more permissive than AP50–95. Always name the metric precisely.

## Data Integrity Checks

- Every `annotation.image_id` should reference an existing image.
- Every `annotation.category_id` should reference a category.
- Bounding-box width and height should be positive.
- Coordinates should be plausible for the image dimensions.
- Segmentation areas and boxes should describe the same instance.
- Train, validation, and test splits must remain separate.
- Custom exports should preserve licensing and source metadata.

## COCO and SAM

COCO provides labeled masks for conventional supervised evaluation and training. SAM-style models are promptable foundation models, but their outputs can still be compared to COCO-format masks using IoU, Dice, precision, recall, and related metrics.

## Dataset Ethics and Limitations

Large public datasets reflect collection choices, label definitions, geographic distributions, privacy decisions, and annotation uncertainty. A COCO-pretrained model is not automatically reliable for every location, camera, population, or specialized domain. Evaluate it on representative local data.

## Official References

- COCO homepage: <https://cocodataset.org/>
- Dataset paper: <https://arxiv.org/abs/1405.0312>
- Detection evaluation: <https://cocodataset.org/#detection-eval>
- Download page: <https://cocodataset.org/#download>

## Recommended Exercise

Run `yolov8n.pt` on `bus.jpg`, print `result.names`, identify the predicted COCO classes, and compare the model's zero-based mapping with the official category IDs stored in a COCO annotation file.
