# Resources

Professional references, datasets, assets, downloads, and external materials used throughout the SAM 3 Computer Vision Learning Journey.

Each resource is a self-contained learning guide explaining what the technology or material is, how it works, how to install or obtain it, how it is used in the course, and how to reproduce practical examples safely.

## Reference Guides

| No. | Resource | Purpose | Course use |
|---:|---|---|---|
| 01 | [Supervision](./references/01-Supervision/) | Model-agnostic detections, filtering, tracking, counting, and visualization | Session 02 onward |
| 02 | [Supervision Annotators](./references/02-Supervision-Annotators/) | Boxes, labels, masks, markers, traces, zones, and counters | Session 03 onward |
| 03 | [Ultralytics YOLOv8](./references/03-Ultralytics-YOLOv8/) | Detection, segmentation, pose, classification, training, and inference | Session 02 onward |
| 04 | [Original YOLO and Darknet](./references/04-Original-YOLO-Darknet/) | Historical architecture and foundational academic references | Session 02 |
| 05 | [COCO Dataset](./references/05-COCO-Dataset/) | Categories, annotations, JSON format, tasks, and evaluation | Session 02 onward |
| 06 | [Meta SAM 3 on Hugging Face](./references/06-Meta-SAM3-Hugging-Face/) | Model access, prompts, image segmentation, video segmentation, and tracking | Sessions 07, 12, and 14 |
| 07 | [Hugging Face Access Tokens](./references/07-Hugging-Face-Access-Tokens/) | Secure authentication for gated models and notebooks | Session 14 |
| 08 | [Trackers](./references/08-Trackers/) | Current multi-object tracking package and ByteTrack migration | Session 05 onward |

## Course Assets

[Open the complete asset guide](./assets/)

| Asset | Purpose |
|---|---|
| `bus.jpg` | Multi-object image for detection, filtering, annotation, and segmentation |
| `zidane.jpg` | Second image for reuse and generalization tests |
| `vehicles.mp4` | Video for tracking, zones, counting, and temporal segmentation |

The asset guide includes canonical URLs, manual download commands, validation examples, integrity practices, and an automated downloader.

## Course Downloads

[Open course downloads and official links](./downloads/)

This section documents:

- The Google Drive folder containing sixteen course notebooks
- The ICS calendar containing seventeen course milestones
- The official course glossary
- The syllabus, welcome, calendar, resources, and materials pages
- Safe handling, attribution, and broken-link procedures

## Verified Examples

| Resource | Verified evidence |
|---|---|
| Supervision Annotators | [Four-panel annotator gallery](./references/02-Supervision-Annotators/assets/output/annotator_gallery.jpg) |
| Ultralytics YOLOv8 | [Annotated detection image](./references/03-Ultralytics-YOLOv8/assets/output/bus_yolov8.jpg) and [JSON results](./references/03-Ultralytics-YOLOv8/assets/output/bus_yolov8_detections.json) |

## Resource Standard

Every professional guide includes the applicable parts of this standard:

- Conceptual explanation and architecture
- Installation or access instructions
- Core classes, files, and terminology
- Reproducible practical examples
- Expected inputs and outputs
- Security, licensing, or data-integrity guidance
- Common errors and troubleshooting
- Professional implementation practices
- Course-session mapping
- Authoritative and primary references

## Organization

```text
06-resources/
├── README.md
├── references/
│   ├── 01-Supervision/
│   ├── 02-Supervision-Annotators/
│   ├── 03-Ultralytics-YOLOv8/
│   ├── 04-Original-YOLO-Darknet/
│   ├── 05-COCO-Dataset/
│   ├── 06-Meta-SAM3-Hugging-Face/
│   ├── 07-Hugging-Face-Access-Tokens/
│   └── 08-Trackers/
├── assets/
│   ├── README.md
│   └── download_course_assets.py
└── downloads/
    └── README.md
```

## Maintenance Rule

External APIs, model versions, access requirements, and course links can change. Before updating a guide, verify the current authoritative source and preserve course-era behavior when it is historically relevant.
