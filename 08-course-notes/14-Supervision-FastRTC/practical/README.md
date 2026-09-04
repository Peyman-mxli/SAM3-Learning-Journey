# Practical — Local Live Detection with FastRTC

This practical is the execution phase of Class 14.

## Goal

Run a live webcam through:

```text
FastRTC → YOLOv8 → Supervision → Browser
```

and verify that detections are rendered in real time on the local machine.

## Success Criteria

The practical is considered validated when all of the following are confirmed:

- the local FastRTC/Gradio interface opens
- the browser receives webcam video
- YOLOv8 processes incoming frames
- `sv.Detections` is created successfully
- bounding boxes are visible
- labels and confidence scores are visible
- latency remains usable with `skip_frames=True`
- CPU or GPU device selection is recorded

## Parameters from the Notebook

```python
MODEL_NAME = "yolov8n.pt"
CONFIDENCE = 0.30
IMAGE_SIZE = 640
```

## Evidence to Save After the Real Run

Recommended evidence:

```text
practical/
├── README.md
└── outputs/
    ├── live_detection_screenshot.jpg
    └── execution_summary.json
```

Do not add fabricated results. The outputs should be saved only after the local webcam application has actually been executed.

## Execution Summary Template

The JSON result can record:

```json
{
  "model": "yolov8n.pt",
  "confidence": 0.30,
  "image_size": 640,
  "python": "<real version>",
  "fastrtc": "<real version>",
  "gradio": "<real version>",
  "ultralytics": "<real version>",
  "supervision": "<real version>",
  "aiortc": "<real version>",
  "cuda_available": false,
  "device": "cpu or real GPU name",
  "webcam_stream_verified": true
}
```

## Current Status

**Notebook workflow:** Ready  
**Repository documentation:** Ready  
**Live webcam run:** Pending
