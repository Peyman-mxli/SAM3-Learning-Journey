# Outputs — Class 14 Supervision + FastRTC

This folder is reserved for **real evidence from the local Class 14 webcam run**.

Do not add fabricated screenshots or manually invented execution results.

## Expected Evidence

```text
outputs/
├── README.md
├── live_detection_screenshot.jpg
└── execution_summary.json
```

## Validation Requirements

A Class 14 run is considered verified when the local application actually demonstrates:

- FastRTC/Gradio opening on a local `127.0.0.1` URL
- browser webcam permission granted
- live camera frames received through WebRTC
- YOLOv8 inference on the incoming frames
- conversion to `sv.Detections`
- Supervision bounding boxes rendered
- labels and confidence scores rendered
- the real Python/package versions and CPU/GPU device recorded

## Source Notebook Parameters

```text
Model: yolov8n.pt
Confidence: 0.30
Image size: 640
skip_frames: True
Host: 127.0.0.1
Public sharing: False
```

The uploaded course notebook explicitly uses a local Windows/Antigravity workflow and states that local execution does not require Colab, TURN, Twilio, a Hugging Face token, or a public link.
