# Input Images

This directory contains the input images used by the Detection Filtering and NMS Pipeline.

The main test image for this project is:

```text
pedestrian-plaza-detection-test.png
```

The Python pipeline loads the image from:

```python
INPUT_IMAGE = "assets/input/pedestrian-plaza-detection-test.png"
```

## Directory Structure

```text
assets/
└── input/
    ├── README.md
    └── pedestrian-plaza-detection-test.png
```

The image is processed by YOLOv8 and then passed through the filtering pipeline using Supervision.

## Related Project

[Back to Detection Filtering and NMS Pipeline](../../README.md)
