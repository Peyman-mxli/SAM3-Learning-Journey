# Course Assets — Images and Video Used by the Notebooks

This directory documents the external image and video assets used by the course notebooks. The notebooks commonly download these files during their first cells so learners can reproduce the exercises without manually searching for media.

The repository records the canonical source URLs and usage instructions. External media should not be duplicated in Git unless redistribution is clearly permitted and versioning the binary is necessary.

## Asset Catalog

| Asset | Type | Canonical URL | Course use |
|---|---|---|---|
| `bus.jpg` | Image | <https://ultralytics.com/images/bus.jpg> | Eight of the sixteen notebooks |
| `zidane.jpg` | Image | <https://ultralytics.com/images/zidane.jpg> | Six notebooks |
| `vehicles.mp4` | Video | <https://media.roboflow.com/supervision/video-examples/vehicles.mp4> | Sessions 05, 06, and 12 |

## `bus.jpg`

The Ultralytics bus image contains several people and a bus in a street scene. It is useful because one image provides:

- Multiple object instances
- More than one COCO class
- Partial occlusion
- Different object scales
- A clear demonstration of boxes, labels, filtering, and masks

Download:

```bash
curl -L "https://ultralytics.com/images/bus.jpg" -o bus.jpg
```

Python:

```python
from urllib.request import urlretrieve

urlretrieve("https://ultralytics.com/images/bus.jpg", "bus.jpg")
```

Validate with OpenCV:

```python
import cv2

image = cv2.imread("bus.jpg")
if image is None:
    raise RuntimeError("bus.jpg could not be decoded")

print(image.shape)
```

## `zidane.jpg`

The second Ultralytics image checks whether code written for one sample generalizes to another scene. It is appropriate for testing:

- Reusable functions
- Confidence filtering
- Person detection
- Class-specific filtering
- Annotation scaling
- Segmentation on a different composition

Download:

```bash
curl -L "https://ultralytics.com/images/zidane.jpg" -o zidane.jpg
```

## `vehicles.mp4`

The Roboflow traffic video provides temporal continuity for:

- Frame-by-frame inference
- ByteTrack identity assignment
- Trace visualization
- Polygon-zone occupancy
- Line-crossing counts
- SAM 3 temporal segmentation
- Per-object analytics

Download:

```bash
curl -L \
  "https://media.roboflow.com/supervision/video-examples/vehicles.mp4" \
  -o vehicles.mp4
```

Inspect video metadata:

```python
import cv2

capture = cv2.VideoCapture("vehicles.mp4")
if not capture.isOpened():
    raise RuntimeError("vehicles.mp4 could not be opened")

fps = capture.get(cv2.CAP_PROP_FPS)
frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
capture.release()

print({
    "fps": fps,
    "frames": frame_count,
    "width": width,
    "height": height,
})
```

## Automated Downloader

The included script downloads all three files safely into a selected directory and records their size and SHA-256 digest.

```bash
python download_course_assets.py --output-dir ./input
```

Download one asset:

```bash
python download_course_assets.py --asset bus --output-dir ./input
```

Existing files are preserved unless `--overwrite` is provided.

## Integrity and Reproducibility

Remote files can change while retaining the same URL. For a reproducible experiment:

1. Record the download date.
2. Record the source URL.
3. Calculate a SHA-256 digest.
4. Record image or video dimensions.
5. Avoid silently overwriting an existing asset.
6. Preserve the unmodified source separately from generated outputs.

The downloader prints the SHA-256 value after every successful download.

## Input and Output Separation

Recommended project structure:

```text
assets/
├── input/
│   ├── bus.jpg
│   ├── zidane.jpg
│   └── vehicles.mp4
└── output/
    ├── bus_annotated.jpg
    └── vehicles_tracked.mp4
```

Never write an annotated result over the original input.

## Licensing and Attribution

An asset being downloadable does not mean every form of redistribution is permitted. Keep the canonical source URL, review the source site's current terms, provide attribution where required, and avoid republishing external binaries unnecessarily.

## Troubleshooting

| Problem | Resolution |
|---|---|
| Downloaded file is HTML | Redirect, access, or error page was saved; inspect content type and retry canonical URL |
| `cv2.imread()` returns `None` | Verify file path, size, and image decoding |
| Video has zero frames | Confirm download completed and codec support is installed |
| Notebook cannot find asset | Print current working directory and use a resolved path |
| Colab asset disappears | Colab storage is temporary; redownload or mount Drive |

## Canonical Sources

- Ultralytics sample images: <https://ultralytics.com/images/bus.jpg> and <https://ultralytics.com/images/zidane.jpg>
- Roboflow/Supervision sample video: <https://media.roboflow.com/supervision/video-examples/vehicles.mp4>
