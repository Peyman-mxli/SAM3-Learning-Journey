# Original YOLO and Darknet — Historical and Architectural Reference

The original YOLO project introduced a unified approach to real-time object detection: a single neural network predicts bounding boxes and class probabilities directly from an image. Joseph Redmon's Darknet site is preserved here as the historical source cited by the course.

## Resource Summary

| Item | Details |
|---|---|
| YOLO meaning | You Only Look Once |
| Original author | Joseph Redmon and collaborators |
| Framework | Darknet, written in C and CUDA |
| Purpose | Real-time object detection |
| Course association | Session 02 · `01_a_introduccion_supervision` |
| Historical project | <https://pjreddie.com/darknet/yolo/> |
| Original paper | <https://arxiv.org/abs/1506.02640> |

## Core Idea

Earlier detection systems often separated object localization and classification into multiple stages. YOLO reframed detection as a single regression problem evaluated by one neural network.

```text
Input image
    ↓
One convolutional network
    ↓
Bounding boxes + object confidence + class probabilities
    ↓
Final detections
```

This unified design made real-time processing a central part of the YOLO identity.

## Conceptual Output

For each candidate object, a detector must describe:

- Bounding-box position and size
- Probability that an object exists
- Predicted class probabilities
- Final confidence used for filtering

Overlapping candidates are reduced with non-maximum suppression.

## Important Concepts

### Intersection over Union

IoU measures overlap between two regions:

```text
IoU = intersection area / union area
```

It is used for matching predictions to ground truth during evaluation and for suppressing duplicate predictions.

### Non-Maximum Suppression

NMS keeps a high-confidence box and removes sufficiently overlapping alternatives. A very low threshold can remove nearby real objects; a high threshold can preserve duplicates.

### Speed–Accuracy Trade-off

Smaller input sizes and architectures improve speed but may reduce small-object accuracy. Larger models and input sizes consume more memory and computation.

## Darknet

Darknet is the original neural-network framework associated with early YOLO versions. It uses configuration files, binary weights, C code, and optional CUDA acceleration.

Historical Darknet commands commonly followed this form:

```bash
./darknet detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights data/dog.jpg
```

This command is included for historical understanding. The SAM 3 course uses the modern Ultralytics Python API rather than requiring a Darknet build.

## Evolution of the Family

| Generation | Broad contribution |
|---|---|
| YOLOv1 | Unified single-network detection formulation |
| YOLOv2 / YOLO9000 | Improved accuracy, anchors, broader class coverage |
| YOLOv3 | Multi-scale predictions and a widely used Darknet implementation |
| Later YOLO families | Continued development across several independent projects and organizations |

The name “YOLO” now describes a broad model family. Later versions are not all maintained by the original authors, so architectural claims should be associated with the specific version and implementation.

## Original YOLO vs. Course YOLOv8

| Aspect | Original/Darknet reference | Course implementation |
|---|---|---|
| Primary interface | C executable and configuration files | Python and CLI |
| Checkpoint | Darknet `.weights` | PyTorch `.pt` |
| Model used | Early YOLO generations | Ultralytics YOLOv8n |
| Integration | Native Darknet outputs | Ultralytics `Results` → `sv.Detections` |
| Purpose in repository | Historical and academic origin | Practical inference pipeline |

## When This Reference Is Useful

- Understanding why single-stage detection mattered
- Learning the historical origin of modern YOLO APIs
- Interpreting IoU, confidence, NMS, and real-time detection
- Distinguishing the original project from later implementations
- Citing the foundational work correctly

## Limitations of Early YOLO Systems

Early YOLO work discussed localization errors, difficulty with small grouped objects, and constraints introduced by the prediction formulation. Modern systems have changed substantially, but small objects, occlusion, class imbalance, and domain shift remain important detection challenges.

## Citation Practice

When discussing the historical algorithm, cite the paper. When documenting the code actually used in this repository, cite the Ultralytics YOLOv8 documentation and record the package and checkpoint version.

## Official and Primary References

- Darknet YOLO project: <https://pjreddie.com/darknet/yolo/>
- Original YOLO paper: <https://arxiv.org/abs/1506.02640>
- YOLOv2 / YOLO9000 paper: <https://arxiv.org/abs/1612.08242>
- YOLOv3 technical report: <https://arxiv.org/abs/1804.02767>

## Safety Note

Do not download or execute unverified forks simply because they use the Darknet or YOLO name. Prefer primary sources, inspect build instructions, and isolate legacy dependencies from the modern course environment.
