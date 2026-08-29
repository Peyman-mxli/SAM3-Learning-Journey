# Supervision Annotators — Professional Visualization Guide

Supervision annotators convert structured computer vision predictions into clear visual evidence. They draw bounding boxes, labels, masks, markers, traces, zones, counters, and other overlays on images or video frames.

Annotators do not detect or segment objects. They visualize information already stored in `sv.Detections`, `sv.KeyPoints`, zones, or tracking results.

## Resource Summary

| Item | Details |
|---|---|
| Category | Computer vision visualization |
| Library | Supervision |
| Input | Image plus structured predictions |
| Output | Annotated NumPy image |
| Course association | Session 03 · `01_b_anotacion_visualizacion` |
| Official catalog | <https://supervision.roboflow.com/annotators/> |
| Practical guide | <https://supervision.roboflow.com/how_to/detect_and_annotate/> |

## Why Annotators Matter

A model output may be technically correct but difficult for a person to interpret. Arrays of coordinates, class IDs, confidence values, masks, and tracker IDs must be converted into a visualization that answers questions such as:

- Where is each detected object?
- What class was assigned?
- How confident is the model?
- Which pixels belong to the segmentation?
- Which identity is being tracked?
- Where has the tracked object moved?
- Is an object inside a region or crossing a line?

Annotators create that communication layer.

```text
Model predictions
       ↓
sv.Detections
       ↓
Visual encoding decisions
       ↓
One or more annotators
       ↓
Human-readable image or video
```

## How Annotation Works

Every standard detection annotator follows the same basic pattern:

```python
annotator = sv.BoxAnnotator()

annotated_image = annotator.annotate(
    scene=image.copy(),
    detections=detections,
)
```

The important inputs are:

| Argument | Purpose |
|---|---|
| `scene` | Image on which the visual overlay will be drawn |
| `detections` | Structured predictions that determine position and metadata |
| `labels` | Optional text list used by label-oriented annotators |

The returned value is another NumPy image. This allows annotators to be composed sequentially.

## Installation

This guide uses Python 3.9 or newer.

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

Install the reproducible example environment:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Google Colab:

```python
!pip install -q supervision ultralytics opencv-python-headless
```

Verify the library:

```bash
python -c "import supervision as sv; print(sv.__version__)"
```

## Main Annotator Families

### Outlines

Outline annotators communicate object extent without covering the underlying image.

| Annotator | Best use |
|---|---|
| `BoxAnnotator` | Standard axis-aligned detections |
| `RoundBoxAnnotator` | Rounded visual style |
| `BoxCornerAnnotator` | Minimal corner-only boxes |
| `EllipseAnnotator` | Person or object emphasis with ellipses |
| `PolygonAnnotator` | Visible instance-mask boundaries |
| `OrientedBoxAnnotator` | Rotated-object detections |

### Shading and Segmentation

| Annotator | Best use |
|---|---|
| `MaskAnnotator` | Semi-transparent instance masks |
| `ColorAnnotator` | Color-filled detection regions |
| `HaloAnnotator` | Soft emphasis around segmentation masks |

Mask-based annotators require `detections.mask`. A box-only detector does not provide segmentation masks.

### Markers

Markers highlight an anchor rather than the full object region.

| Annotator | Visual form |
|---|---|
| `DotAnnotator` | Dot at a selected position |
| `CircleAnnotator` | Circle around detections |
| `TriangleAnnotator` | Directional or compact marker |

Markers are useful for sports analytics, minimaps, crowd visualization, and ground-contact points.

### Labels

`LabelAnnotator` renders text associated with each detection. Typical label fields include:

- Class name
- Confidence score
- Tracker ID
- Custom business or analytical information

Example:

```python
labels = [
    f"#{tracker_id} {class_name} {confidence:.2f}"
    for tracker_id, class_name, confidence in zip(
        detections.tracker_id,
        detections.data["class_name"],
        detections.confidence,
    )
]
```

Always create labels after filtering. Otherwise, label order or count may no longer match the remaining detections.

### Tracking and Movement

`TraceAnnotator` draws historical paths. It requires stable `tracker_id` values, normally produced by a tracker such as ByteTrack.

```python
tracker = sv.ByteTrack()
trace_annotator = sv.TraceAnnotator(trace_length=30)

detections = tracker.update_with_detections(detections)
annotated = trace_annotator.annotate(annotated, detections)
```

The tracker and trace annotator must be created once outside the video-frame loop. Recreating either one for every frame destroys temporal history.

### Zones and Counters

Zone annotators display analytical objects rather than individual predictions:

- `PolygonZoneAnnotator` displays a polygonal region and its occupancy.
- `LineZoneAnnotator` displays a virtual line and crossing totals.

The zone performs the calculation; the corresponding annotator displays the result.

## Color Lookup Strategies

Supervision supports three principal color-mapping strategies:

| Strategy | Color determined by | Best use |
|---|---|---|
| `ColorLookup.CLASS` | Object class | Keep all people one color and all vehicles another |
| `ColorLookup.TRACK` | Persistent tracker ID | Maintain a unique color for each tracked object |
| `ColorLookup.INDEX` | Detection position in the current collection | Visually distinguish detections in one frame |

Example:

```python
box_annotator = sv.BoxAnnotator(
    color_lookup=sv.ColorLookup.CLASS,
)
```

Use class colors for semantic meaning and tracker colors for identity continuity.

## Composition and Drawing Order

Annotators modify the image in sequence. Drawing order therefore affects readability.

Recommended order for segmentation:

```text
1. Masks or shading
2. Traces
3. Boxes or outlines
4. Markers
5. Labels
6. Zones and counters
```

Masks normally go first because they cover larger pixel regions. Labels normally go last so they remain readable.

```python
annotated = image.copy()
annotated = mask_annotator.annotate(annotated, detections)
annotated = box_annotator.annotate(annotated, detections)
annotated = label_annotator.annotate(annotated, detections, labels)
```

## Complete Runnable Example

The included `example_annotator_gallery.py`:

1. Downloads the official `bus.jpg` sample.
2. Runs YOLOv8 detection.
3. Converts the result to `sv.Detections`.
4. Applies a confidence filter.
5. Creates four independently annotated panels.
6. Combines the panels into a comparison gallery.
7. Saves the final image.

Run:

```bash
python example_annotator_gallery.py
```

Expected files:

```text
assets/input/bus.jpg
assets/output/annotator_gallery.jpg
```

The gallery compares:

- Boxes and labels
- Box corners and labels
- Ellipses and labels
- Bottom-center dots and labels

## Verified Output

The example was executed with Supervision 0.30.1 and retained four detections at a confidence threshold of 0.50.

![Verified Supervision annotator gallery](./assets/output/annotator_gallery.jpg)

## Choosing the Right Annotator

| Requirement | Recommended visualization |
|---|---|
| General detection preview | Box + label |
| Dense scene | Corners or dots to reduce visual clutter |
| Instance segmentation | Mask + outline + label |
| Persistent video identity | Tracker-colored box + tracker ID + trace |
| Football or movement analytics | Bottom-center dot + minimap |
| Privacy protection | Blur or pixelation annotator |
| Region occupancy | Polygon zone + zone label |
| Directional counting | Line zone + crossing totals |

## Performance Considerations

Annotation consumes CPU time and memory bandwidth. For long or high-resolution videos:

- Filter detections before drawing.
- Avoid unnecessary annotators.
- Resize only when the output requirements permit it.
- Reuse annotator instances across frames.
- Use masks only when segmentation information is required.
- Write frames directly instead of retaining the full video in memory.
- Separate inference timing from annotation timing during benchmarking.

## Common Problems

| Problem | Cause | Solution |
|---|---|---|
| Labels appear on the wrong boxes | Labels were built before filtering or reordering | Generate labels from the final `Detections` object |
| Mask annotator shows nothing | `detections.mask` is absent | Use a segmentation model or attach valid masks |
| Trace annotator fails or shows no history | `tracker_id` is missing or state is recreated | Run a tracker and preserve state across frames |
| Colors change between video frames | Color lookup uses index rather than tracker ID | Use `ColorLookup.TRACK` |
| Original image is unexpectedly modified | The scene was passed without copying | Begin with `image.copy()` |
| Text is too small | Resolution and default text scale differ | Adjust `text_scale`, `text_thickness`, and padding |
| Old example raises `AttributeError` | Annotator names changed between versions | Check `sv.__version__` and matching versioned documentation |

## Professional Visualization Checklist

- Use a consistent color meaning.
- Include confidence only when it helps interpretation.
- Include tracker IDs only for temporal workflows.
- Avoid overlapping labels when the scene is dense.
- Keep masks transparent enough to preserve scene context.
- Use sufficient line and text thickness for the output resolution.
- Preserve an unmodified source image.
- Save the model settings with the visual result.
- Never treat a visualization alone as a complete evaluation.

## Relationship to This Course

The annotator catalog is introduced in Session 03, but annotators remain important throughout the journey:

- Detection boxes and labels
- Filtering comparisons
- ByteTrack IDs and traces
- Polygon and line zones
- SAM and SAM 3 masks
- Confidence-dependent opacity
- Football-player anchors and minimaps

## Official References

- Annotator catalog: <https://supervision.roboflow.com/annotators/>
- Detect and annotate: <https://supervision.roboflow.com/how_to/detect_and_annotate/>
- Supervision documentation: <https://supervision.roboflow.com/>
- Source repository: <https://github.com/roboflow/supervision>

## Next Exercise

Run the gallery, then modify only one variable at a time: change the confidence threshold, select a different color lookup, increase label size, or keep only the `person` class. Compare how each decision changes clarity.
