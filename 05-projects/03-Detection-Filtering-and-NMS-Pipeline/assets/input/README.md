# Input Images

Place the images used as input for the Detection Filtering and NMS Pipeline in this directory.

The default project configuration expects:

```text
image.jpg
```

The image is loaded by:

```python
INPUT_IMAGE = "assets/input/image.jpg"
```

## Example Structure

```text
assets/
└── input/
    ├── README.md
    └── image.jpg
```

The input image is processed by YOLOv8 before the resulting detections are filtered using Supervision.

## Related Project

[Back to Detection Filtering and NMS Pipeline](../../README.md)
