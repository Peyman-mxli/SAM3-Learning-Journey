# Practical — Local Live Detection with FastRTC

This practical is the execution phase of Class 14.

## Goal

Run a live webcam through:

```text
FastRTC → YOLOv8 → Supervision → Browser
```

and verify that detections are rendered in real time on the local machine.

## Validation Result

The practical was executed successfully on the local Windows machine using Google Antigravity IDE.

The verified pipeline was:

```text
Webcam
   ↓
FastRTC / WebRTC
   ↓
YOLOv8n
   ↓
sv.Detections
   ↓
Supervision BoxAnnotator + LabelAnnotator
   ↓
Annotated live video in the browser
```

## Real Environment

```text
Python: 3.11.9
FastRTC: 0.0.34
Gradio: 5.50.0
Ultralytics: 8.4.138
Supervision: 0.30.1
aiortc: 1.15.0
CUDA available: False
Device: CPU
Local URL: http://127.0.0.1:7860
```

## Model Parameters

```python
MODEL_NAME = "yolov8n.pt"
CONFIDENCE = 0.30
IMAGE_SIZE = 640
DEVICE = "cpu"
```

## Verified Live Detections

The saved evidence frame contains two visible real-time detections:

```text
person      0.86
cell phone  0.74
```

The bounding boxes, class labels, and confidence scores were rendered correctly on the live webcam stream.

## Important FastRTC Interaction

After browser camera permission is granted, the live processing session begins when the **Record** control in the FastRTC/Gradio interface is activated. Once active, incoming webcam frames are sent through the YOLOv8 + Supervision processing function and returned to the browser with annotations.

## Evidence

```text
practical/
├── README.md
├── run_live_detection.py
├── requirements.txt
└── outputs/
    ├── README.md
    ├── live_detection_screenshot.jpg
    └── execution_summary.json
```

The screenshot is real evidence from the successful local execution and is not a simulated result.

## Success Criteria

- Local FastRTC/Gradio interface opened: **Verified**
- Webcam permission and video stream: **Verified**
- YOLOv8 frame processing: **Verified**
- `sv.Detections` conversion: **Verified**
- Bounding boxes rendered: **Verified**
- Labels rendered: **Verified**
- Confidence scores rendered: **Verified**
- CPU execution recorded: **Verified**
- Local WebRTC workflow completed: **Verified**

## Current Status

**Notebook workflow:** Completed  
**Repository documentation:** Completed  
**Live webcam run:** Completed and verified  
**Class 14 practical:** Completed
