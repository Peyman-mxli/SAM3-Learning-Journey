# 11 — SAM 3 Video Segmentation

This session extends the SAM 3 learning journey from static-image prompting and segmentation into **video segmentation, tracking, temporal analysis, and semantic video prompts**.

The lesson explores two different pipelines for processing video:

- **Option A — YOLO + ByteTrack + SAM 3:** YOLO detects objects, ByteTrack assigns persistent IDs, and SAM 3 generates precise masks.
- **Option B — SAM 3 with text prompts:** `SAM3VideoSemanticPredictor` detects, segments, and tracks semantic concepts directly from text.

The original course notebook and class recording are preserved in this folder.

---

## Session Status

```text
Session:         11 — SAM 3 Video Segmentation
Course Notebook: ✅ Preserved
Class Recording: ✅ Added
Documentation:   ✅ Completed
```

---

# Session Objective

The objective of this lesson is to understand how SAM 3 can be applied to video while maintaining object identity, pixel-level masks, and temporal information across frames.

The session combines concepts from earlier lessons:

```text
Object Detection
      +
Object Tracking
      +
Segmentation Masks
      +
Video Processing
      +
Semantic Text Prompts
```

The main learning goals are:

- Process video frame by frame
- Detect objects with YOLO
- Assign persistent object IDs with ByteTrack
- Use bounding boxes as SAM 3 prompts
- Transfer tracking and class attributes from YOLO detections to SAM detections
- Visualize masks, labels, and movement traces
- Filter objects before segmentation
- Analyze mask area over time
- Use `SAM3VideoSemanticPredictor` with text prompts
- Compare detector-guided segmentation with direct semantic video segmentation

---

# Topics Covered

- Video processing with Supervision
- `sv.VideoInfo`
- `sv.process_video`
- YOLOv8 object detection
- ByteTrack object tracking
- Persistent tracker IDs
- SAM 3 bounding-box prompts
- Frame-by-frame segmentation
- `sv.MaskAnnotator`
- `sv.LabelAnnotator`
- `sv.TraceAnnotator`
- Attribute transfer between detection and segmentation results
- Polygon-zone filtering
- Dynamic mask opacity
- Mask-area analysis through time
- Semantic video prompts
- `SAM3VideoSemanticPredictor`
- Streaming video inference
- OpenCV video writing
- GPU and CPU performance considerations

---

# Session Structure

```text
11-SAM3-Video-Segmentation/
│
├── README.md
├── CLASS-RECORDING.md
├── 05_b_segmentacion_sam_video.ipynb
│
└── practical/
    └── README.md
```

---

# Class Recording

The class recording for this session is available on YouTube:

[SAM 3 en Video — Segmentación y Tracking](https://youtu.be/_EuNGCYS35k)

Repository documentation:

[CLASS-RECORDING.md](./CLASS-RECORDING.md)

---

# Original Class Notebook

The original course notebook is preserved as:

[05_b_segmentacion_sam_video.ipynb](./05_b_segmentacion_sam_video.ipynb)

The notebook contains the complete lesson workflow, including both video pipelines, three interactive experiments, and an extension challenge.

---

# From Static Images to Video

Earlier SAM lessons process independent images:

```text
Image
  ↓
Prompt
  ↓
SAM 3
  ↓
Segmentation Mask
```

Video introduces a sequence of related frames:

```text
Frame 1 → Frame 2 → Frame 3 → ... → Frame N
```

The processing system must now preserve useful information through time:

```text
Object Detection
       ↓
Persistent Identity
       ↓
Precise Mask
       ↓
Temporal Observation
```

This lesson connects static segmentation with video tracking and semantic video analysis.

---

# Two Video-Segmentation Pipelines

## Option A — YOLO + ByteTrack + SAM 3

```text
Video Frame
     ↓
YOLO
     ↓
Object Detections
     ↓
ByteTrack
     ↓
Persistent Tracker IDs
     ↓
Bounding-Box Prompts
     ↓
SAM 3
     ↓
Pixel-Level Masks
     ↓
Mask + Label + Trace Annotation
```

This option provides greater control over detection classes, confidence thresholds, tracker IDs, zones, and filtering rules.

---

## Option B — SAM 3 Directly with Text

```text
Video
  ↓
Text Prompts
["car", "bus", "truck"]
  ↓
SAM3VideoSemanticPredictor
  ↓
Detection + Segmentation + Tracking
  ↓
Annotated Video
```

This option removes the need for a separate YOLO detector and ByteTrack stage.

It is useful for rapid semantic exploration when the desired concepts can be described with natural-language prompts.

---

# Option A — Model Initialization

The notebook installs the main dependencies:

```python
%pip install supervision ultralytics trackers
%pip install -q rfdetr "trackers==2.4.0"
```

The models and tracker are initialized with:

```python
yolo_model = YOLO("yolov8n.pt")
sam_model = SAM(sam_path)
tracker = ByteTrackTracker()
```

The sample video is downloaded from the Supervision video examples:

```text
assets/vehicles.mp4
```

Video metadata is inspected using:

```python
video_info = sv.VideoInfo.from_video_path(
    "assets/vehicles.mp4"
)
```

---

# Complete YOLO + ByteTrack + SAM 3 Pipeline

For every frame, the notebook performs:

```text
1. Detect objects with YOLO
2. Convert results to sv.Detections
3. Assign persistent IDs with ByteTrack
4. Extract bounding boxes
5. Send bounding boxes to SAM 3
6. Convert SAM results to sv.Detections
7. Transfer tracking and class attributes
8. Draw masks, labels, and traces
9. Write the annotated frame to the output video
```

The primary output path defined by the notebook is:

```text
assets/vehicles_sam.mp4
```

---

# Attribute Transfer

SAM 3 generates precise masks, but the segmentation result does not automatically contain all the tracking and class information produced by YOLO and ByteTrack.

Before transfer:

```text
SAM detections
├── masks
├── bounding boxes
├── tracker_id: missing
└── YOLO class data: missing
```

The lesson transfers attributes when the number and order of detections match:

```python
if len(sam_det) == len(yolo_det):
    sam_det.tracker_id = yolo_det.tracker_id
    sam_det.class_id = yolo_det.class_id
    sam_det.confidence = yolo_det.confidence
```

After transfer:

```text
SAM detections
├── precise masks
├── persistent tracker IDs
├── YOLO class IDs
└── YOLO confidence values
```

This makes it possible to combine segmentation, tracking, labels, confidence, and trajectory visualization.

---

# Video Annotation

The pipeline uses three Supervision annotators:

```python
mask_annotator = sv.MaskAnnotator(opacity=0.6)
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()
```

Their roles are:

| Annotator | Purpose |
|---|---|
| `MaskAnnotator` | Draws pixel-level segmentation masks |
| `LabelAnnotator` | Displays tracker IDs and class names |
| `TraceAnnotator` | Visualizes object movement through time |

---

# Experiment 1 — Segment Only Objects Inside a Zone

The first experiment combines `PolygonZone` with SAM 3.

```text
Frame
  ↓
YOLO + ByteTrack
  ↓
Polygon-Zone Test
  ↓
Keep Objects Inside Zone
  ↓
SAM 3
  ↓
Segment Only Relevant Objects
```

The filter is applied before SAM inference:

```python
en_zona = zone.trigger(detections=yolo_det)
yolo_det = yolo_det[en_zona]
```

This avoids spending segmentation time on objects outside the region of interest.

The notebook output path is:

```text
assets/vehicles_sam_zona.mp4
```

---

# Experiment 2 — Dynamic Opacity by Confidence

The second experiment uses each YOLO confidence value as the opacity of its SAM mask.

```text
Higher Detection Confidence
            ↓
More Opaque Segmentation Mask

Lower Detection Confidence
            ↓
More Transparent Segmentation Mask
```

Each object is annotated separately:

```python
opacidad = float(det_i.confidence[0])
annotated = sv.MaskAnnotator(
    opacity=opacidad
).annotate(
    scene=annotated,
    detections=det_i
)
```

The notebook output path is:

```text
assets/vehicles_sam_opacidad.mp4
```

---

# Experiment 3 — Mask Area Through Time

The third experiment measures the mask area for every tracked object:

```python
area = int(sam_det.mask[i].sum())
```

The values are stored by tracker ID:

```text
tracker_id
    ↓
Mask Area per Frame
    ↓
Temporal Area Series
```

This can help interpret physical changes:

- A sustained area increase may indicate that an object is approaching the camera.
- A sustained decrease may indicate that an object is moving away.
- Sudden changes may indicate occlusion, re-detection, or segmentation instability.

The notebook produces an area chart for the three objects with the most recorded frames.

The annotated-video path is:

```text
assets/vehicles_sam_areas.mp4
```

---

# Option B — Semantic Video Segmentation

The second pipeline uses:

```python
from ultralytics.models.sam import (
    SAM3VideoSemanticPredictor
)
```

The predictor is configured for segmentation:

```python
overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model="sam3.pt"
)
```

When CUDA is available, half precision is enabled:

```python
if torch.cuda.is_available():
    overrides["half"] = True
```

The semantic prompts are:

```python
text=["car", "bus", "truck"]
```

Streaming inference processes results frame by frame without loading the complete video into memory:

```python
resultados_video = video_predictor(
    source="assets/vehicles.mp4",
    text=["car", "bus", "truck"],
    stream=True
)
```

The output path is:

```text
assets/vehicles_texto.mp4
```

---

# Option A vs. Option B

| Feature | Option A: YOLO + SAM 3 | Option B: SAM 3 with Text |
|---|---|---|
| Previous detector | YOLO required | Not required |
| Prompt type | Bounding boxes | Natural-language text |
| Tracking | ByteTrack | Internal SAM 3 behavior |
| Model stages | Two models plus tracker | One semantic predictor |
| Control | High | Prompt-dependent |
| Class filtering | YOLO class IDs | Text concepts |
| Best use | Controlled pipelines and known detector classes | Rapid semantic exploration |

---

# Performance Considerations

SAM 3 is computationally heavier than standard YOLO inference.

```text
GPU
 ↓
Suitable for practical video processing
and potential real-time workflows

CPU
 ↓
Better suited to offline processing
```

Filtering detections before SAM inference can reduce unnecessary segmentation work.

---

# NumPy 2.0 Compatibility Patch

The notebook includes a compatibility patch for two-dimensional cross products used by Supervision geometry operations.

The patch intercepts 2D inputs and calculates the scalar cross product directly while preserving the original NumPy behavior for other inputs.

This is included to keep polygon-zone geometry compatible with the notebook environment.

---

# Extension Challenge

The lesson challenge is to segment only vehicles whose `tracker_id` is less than or equal to 5.

```python
primeros = yolo_det[
    yolo_det.tracker_id <= 5
]
```

The challenge encourages analysis of what happens after the first five tracked objects leave the frame.

Key questions:

- Are later tracker IDs ignored?
- Does the output continue without masks?
- Should filtering use permanent IDs or active-object order?
- How should the system behave after early objects disappear?

---

# Expected Lesson Outputs

The notebook defines the following outputs:

```text
assets/
│
├── vehicles.mp4
├── vehicles_sam.mp4
├── vehicles_sam_zona.mp4
├── vehicles_sam_opacidad.mp4
├── vehicles_sam_areas.mp4
└── vehicles_texto.mp4
```

These are expected notebook outputs. Execution results will be documented only after Colab validation.

---

# Technologies Used

- Python
- Google Colab
- SAM 3
- Ultralytics
- YOLOv8
- `SAM3VideoSemanticPredictor`
- Supervision
- ByteTrack
- OpenCV
- NumPy
- Matplotlib
- PyTorch

---

# Learning Outcomes

After completing this session, I understand:

- How video segmentation differs from independent image segmentation
- How YOLO, ByteTrack, and SAM 3 can be connected in one pipeline
- How bounding boxes guide SAM 3 segmentation
- Why tracker IDs and class attributes must be transferred to SAM detections
- How to visualize masks, labels, and movement traces together
- Why filtering before SAM can improve efficiency
- How polygon zones can control which objects are segmented
- How confidence can control mask visualization
- How mask area can be analyzed over time
- How semantic text prompts can drive video segmentation
- How streaming inference reduces memory requirements
- The trade-offs between detector-guided and direct semantic segmentation

---

# Session Resources

## Class Recording

[SAM 3 en Video — Segmentación y Tracking](https://youtu.be/_EuNGCYS35k)

## Recording Documentation

[CLASS-RECORDING.md](./CLASS-RECORDING.md)

## Original Notebook

[05_b_segmentacion_sam_video.ipynb](./05_b_segmentacion_sam_video.ipynb)

## Practical

[practical/README.md](./practical/README.md)

## Reusable Python Implementations

Video segmentation:

[practical/sam3_video_segmentation.py](./practical/sam3_video_segmentation.py)

Four-point football-pitch homography:

[practical/football_pitch_homography.py](./practical/football_pitch_homography.py)

Visual homography example:

[practical/assets/output/football_pitch_four_points.svg](./practical/assets/output/football_pitch_four_points.svg)

## Practical Dependencies

[practical/requirements.txt](./practical/requirements.txt)

## Practical Assets

[practical/assets/](./practical/assets/)

# Session Progression

```text
07 — Advanced MaskAnnotator and SAM2
        ↓
08 — SAM 3 Text Prompts
        ↓
09 — SAM Encoder-Decoder
        ↓
10 — SAM 3 Point Prompts
        ↓
11 — SAM 3 Video Segmentation
```

This session combines the concepts of detection, tracking, segmentation, filtering, visualization, and semantic prompting inside complete video-processing workflows.

---

# Current Status

```text
Session 11 — SAM 3 Video Segmentation

Documentation:   COMPLETED   ✅
Class Notebook:  PRESERVED   ✅
Class Recording: ADDED       ✅
```

**Session 11 documentation has been created successfully.**
