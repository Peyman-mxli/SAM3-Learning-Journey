# Outputs — Class 14 Supervision + FastRTC

This folder contains **real evidence from the verified local Class 14 webcam run**.

## Verified Evidence

```text
outputs/
├── README.md
├── live_detection_screenshot.jpg
└── execution_summary.json
```

## Real Execution Result

The Class 14 application was executed locally on Windows through Google Antigravity IDE and FastRTC/Gradio at:

```text
http://127.0.0.1:7860
```

The live webcam stream was successfully processed through YOLOv8 and Supervision.

Verified visible detections in the saved evidence frame:

```text
person      0.86
cell phone  0.74
```

The screenshot demonstrates:

- live webcam frames received through WebRTC
- YOLOv8 inference on incoming frames
- conversion to `sv.Detections`
- Supervision bounding boxes
- class labels
- confidence scores

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
```

## Model Configuration

```text
Model: yolov8n.pt
Confidence: 0.30
Image size: 640
skip_frames: True
Host: 127.0.0.1
Public sharing: False
```

## Status

**Live execution:** Verified  
**Detection annotations:** Verified  
**Evidence captured:** Verified  
**Class 14 practical:** Completed
