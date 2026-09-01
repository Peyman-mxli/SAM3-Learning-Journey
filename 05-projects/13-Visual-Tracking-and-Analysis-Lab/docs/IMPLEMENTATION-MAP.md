# Laboratory Requirement → Implementation Map

This document translates the original laboratory proposal into concrete engineering components.

| Laboratory requirement | Project 13 implementation |
|---|---|
| Load image/video | OpenCV input layer |
| Preserve date/source | Session metadata in SQLite |
| Detect entities | Ultralytics YOLO |
| Tracking | Supervision ByteTrack |
| Temporal identity | Persistent tracker IDs |
| Segment entities | SAM 3 adapter boundary |
| Store observations | SQLite observations table |
| Save confidence | Observation confidence field |
| Save notes | Notes field |
| Historical query | Session and observation queries |
| Compare sessions | Aggregated session metrics |
| Dashboard | Streamlit |
| Export evidence | CSV/JSON-ready DataFrame layer |
| Precision/Recall | Evaluation module target |
| IoU/Dice | Segmentation evaluation target |
| Failure documentation | Results + limitations workflow |

## Data model

### sessions

- session_id
- source_path
- source_type
- created_at
- notes

### observations

- observation_id
- session_id
- frame_index
- timestamp_seconds
- tracker_id
- class_id
- class_name
- confidence
- x1
- y1
- x2
- y2
- center_x
- center_y
- mask_area
- notes

## Processing contract

Every retained observation should be traceable to:

```text
session
  → frame
    → detection
      → tracker identity
        → optional segmentation
          → structured record
```

## Evaluation contract

When ground truth is available, evaluate:

### Detection

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
```

### Segmentation

```text
IoU  = intersection / union
Dice = 2 × intersection / (prediction + ground truth)
```

### Tracking

Document:

- track duration
- ID switches
- fragmentation
- missed tracks
- confidence stability
- movement in image space

## Failure taxonomy

Every validation run should check:

- poor illumination
- glare
- motion blur
- small objects
- large scale changes
- partial occlusion
- full occlusion
- crowded scenes
- camera movement
- unusual viewpoints
- out-of-distribution objects
- segmentation leakage
- tracker identity switches

## Portfolio completion checklist

- [x] Dedicated project structure
- [x] Laboratory proposal preserved
- [x] Engineering implementation map
- [x] SQLite persistence layer
- [x] Executable YOLO + ByteTrack processing skeleton
- [x] Streamlit dashboard entry point
- [ ] Add 2–5 project-specific video test runs
- [ ] Preserve Project 13 generated outputs
- [ ] Add ground-truth evaluation dataset
- [ ] Generate Project 13 Precision / Recall report
- [ ] Generate Project 13 IoU / Dice report
- [ ] Add final screenshots and demo evidence

The unchecked items require fresh Project 13 execution evidence and should not be marked complete until those runs are performed.
