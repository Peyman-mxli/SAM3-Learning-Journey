# Input Assets — Zones and Counting

This directory contains the **source video files** used by the Zones and Counting practical.

The files stored here are used as input for the computer vision pipeline and should remain unchanged during processing.

---

## Directory Structure

```text
input/
├── README.md
└── vehicles.mp4
```

---

## Main Input Video

The practical uses:

```text
vehicles.mp4
```

The complete relative path is:

```text
assets/input/vehicles.mp4
```

The video contains moving vehicles and provides a suitable scene for demonstrating:

- YOLO object detection
- Multi-object tracking
- Persistent tracker IDs
- Polygon-based occupancy
- Line-crossing detection
- Directional counting
- Spatial video analytics

---

## Source

The course notebook uses the public Supervision example video:

```text
vehicles.mp4
```

Original source:

```text
https://media.roboflow.com/supervision/video-examples/vehicles.mp4
```

The notebook downloads it using:

```python
import urllib.request

urllib.request.urlretrieve(
    "https://media.roboflow.com/supervision/video-examples/vehicles.mp4",
    "vehicles.mp4"
)
```

For this repository, the video should instead be stored inside:

```text
practical/assets/input/
```

---

## Expected File

Before running the practical, this directory should contain:

```text
vehicles.mp4
```

The final structure should look like:

```text
practical/
│
├── README.md
├── zones_and_counting_practical.py
│
└── assets/
    ├── README.md
    │
    ├── input/
    │   ├── README.md
    │   └── vehicles.mp4
    │
    └── output/
        └── README.md
```

---

## Input Path in Python

The practical uses:

```python
INPUT_VIDEO = "assets/input/vehicles.mp4"
```

This keeps the input path centralized and easy to modify.

For example:

```python
video_info = sv.VideoInfo.from_video_path(
    INPUT_VIDEO
)
```

The same constant can also be used when processing the video:

```python
sv.process_video(
    source_path=INPUT_VIDEO,
    target_path=OUTPUT_VIDEO,
    callback=callback,
    show_progress=True
)
```

---

## Video Information

Before defining zones, the video dimensions should be inspected.

Example:

```python
video_info = sv.VideoInfo.from_video_path(
    INPUT_VIDEO
)

print(
    f"Resolution: "
    f"{video_info.width} x {video_info.height}"
)
```

The video dimensions are important because polygon and line coordinates are defined in pixel space.

---

## Why Resolution Matters

A polygon created for one video resolution may not appear in the correct location when used with another resolution.

For example:

```text
Video A
1920 × 1080
```

and:

```text
Video B
1280 × 720
```

have different coordinate systems.

Therefore, zone coordinates should be designed according to the actual input video.

---

## Relative Coordinates

The session uses video dimensions to calculate zone coordinates.

For example:

```python
video_info.width // 2
```

represents half of the frame width.

And:

```python
video_info.height // 2
```

represents half of the frame height.

This allows a polygon to be defined relative to the input video.

---

## Polygon Example

The lower-left region can be represented using:

```python
POLYGON_LEFT = np.array([
    [0,                     video_info.height // 2],
    [video_info.width // 2, video_info.height // 2],
    [video_info.width // 2, video_info.height],
    [0,                     video_info.height],
])
```

This depends directly on the input video's width and height.

---

## Line Example

A horizontal line can be positioned at half of the frame height:

```python
line_start = sv.Point(
    x=0,
    y=video_info.height // 2
)

line_end = sv.Point(
    x=video_info.width,
    y=video_info.height // 2
)
```

Again, the coordinates are calculated from the input video dimensions.

---

## Input Workflow

The source video moves through the following pipeline:

```text
vehicles.mp4
     ↓
Read Video Information
     ↓
Read Frame
     ↓
YOLO Detection
     ↓
Supervision Detections
     ↓
ByteTrack
     ↓
Tracked Objects
     ↓
PolygonZone / LineZone
     ↓
Annotation
     ↓
Output Video
```

---

## Do Not Modify the Original Input

The original source video should remain unchanged.

The practical should:

```text
READ
```

from:

```text
assets/input/
```

and:

```text
WRITE
```

to:

```text
assets/output/
```

This prevents generated files from replacing the source material.

---

## Testing with Other Videos

The practical can later be tested with other videos.

Examples:

```text
traffic.mp4
parking_lot.mp4
store_entrance.mp4
pedestrians.mp4
warehouse.mp4
```

If the input video changes, the following may also need adjustment:

- Polygon coordinates
- Counting-line position
- YOLO model or classes
- Confidence threshold
- Tracking behavior
- Annotation sizes
- Output filename

---

## Recommended Naming

Use descriptive filenames for additional experiments.

For example:

```text
vehicles.mp4
parking_lot.mp4
intersection_traffic.mp4
store_entrance.mp4
warehouse_traffic.mp4
```

Avoid ambiguous names such as:

```text
video1.mp4
test.mp4
new.mp4
final.mp4
```

Descriptive names make experiments easier to understand later.

---

## GitHub File Size

Video files can be significantly larger than Markdown and Python files.

Before uploading the input video to GitHub, verify its file size.

If the file is too large, the repository can instead document the original source so the video can be downloaded when running the practical.

The practical should remain reproducible even if the source video is not committed directly.

---

## Reproducibility

A reproducible experiment should clearly identify:

```text
Input
  ↓
Processing
  ↓
Output
```

For this practical:

```text
Input
assets/input/vehicles.mp4

        ↓

Processing
zones_and_counting_practical.py

        ↓

Output
assets/output/
```

This makes it easy for another developer to understand and reproduce the experiment.

---

## Summary

The `input/` directory contains the original media used by the practical.

The main input is:

```text
vehicles.mp4
```

which is processed through:

```text
YOLO
  ↓
ByteTrack
  ↓
PolygonZone
  ↓
LineZone
  ↓
Visualization
```

The original input remains unchanged while all generated results are stored separately in the `output/` directory.

---

## Related Documentation

- [Assets README](../README.md)
- [Practical README](../../README.md)
- [Session README](../../../README.md)

---

## Author

**Peyman Miyandashti**

- [GitHub](https://github.com/Peyman-mxli)
- [LinkedIn](https://www.linkedin.com/in/peyman-mxli/)
