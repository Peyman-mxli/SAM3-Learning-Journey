# Notebook Guide — Class 14 Supervision + FastRTC

This folder contains the original Class 14 notebook used to build a **local live webcam detection pipeline**.

## Notebook

[`class_14_supervision_fastrtc_antigravity.ipynb`](./class_14_supervision_fastrtc_antigravity.ipynb)

## Execution Target

The notebook is designed for:

```text
Windows
Google Antigravity IDE
Python 3.11 / 3.12
Local webcam
127.0.0.1
```

## Recommended Execution Order

```text
1. Create `.venv`
2. Activate the environment
3. Install Jupyter kernel support
4. Select `Python (FastRTC Local)`
5. Install FastRTC + Ultralytics + Supervision
6. Verify package versions and CUDA status
7. Load yolov8n.pt
8. Define process_frame()
9. Start FastRTC Stream
10. Allow camera access in the browser
11. Verify live detections
```

## Core Settings

```python
MODEL_NAME = "yolov8n.pt"
CONFIDENCE = 0.30
IMAGE_SIZE = 640
DEVICE = 0 if torch.cuda.is_available() else "cpu"
```

## Real-Time Processing

The notebook receives webcam frames as NumPy arrays, passes them through YOLOv8, converts results into `sv.Detections`, and draws boxes and labels before sending the annotated frame back through FastRTC.

## Important Latency Setting

```python
VideoStreamHandler(process_frame, skip_frames=True)
```

This prevents old frames from accumulating when inference is slower than the webcam frame rate.

## Local Launch

```python
stream.ui.launch(
    server_name="127.0.0.1",
    share=False,
    inbrowser=True,
    debug=True,
)
```

The application is intentionally local and does not create a public share URL.

## Validation Status

The notebook source is preserved and documented. Live webcam execution evidence is still pending and should be added only after an actual local run.
