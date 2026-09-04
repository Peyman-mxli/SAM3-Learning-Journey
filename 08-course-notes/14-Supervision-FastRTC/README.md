# 14 — Supervision + FastRTC

This session documents a **local real-time computer vision workflow** built with **FastRTC, YOLOv8, and Supervision**.

The supplied notebook is designed to run locally from **Google Antigravity IDE on Windows**, using the computer's webcam through a local WebRTC connection.

The core pipeline is:

```text
Webcam
   ↓
Local WebRTC / FastRTC
   ↓
YOLOv8
   ↓
sv.Detections
   ↓
Supervision annotations
   ↓
Processed video in the browser
```

Unlike the remote/cloud workflows used in other sessions, this setup runs on `127.0.0.1` and does not require Colab, a public share link, TURN/Twilio, or a Hugging Face token.

---

## What I Learn

- Build a local webcam computer vision pipeline
- Use FastRTC for real-time WebRTC video streaming
- Run YOLOv8 inference frame by frame
- Convert Ultralytics results to `sv.Detections`
- Annotate live detections with Supervision
- Display class names and confidence scores
- Select CPU or CUDA automatically
- Use `skip_frames=True` to reduce real-time latency
- Launch a local browser interface through Gradio/FastRTC
- Troubleshoot camera, CUDA, port, and latency issues

---

## Why It Matters

Earlier course work focused heavily on images, prerecorded video, segmentation, and cloud notebooks. Class 14 moves the workflow into **live local inference**.

The important transition is:

```text
Static image / saved video
          ↓
Live webcam frames
          ↓
Continuous inference
          ↓
Real-time annotations
          ↓
Interactive local application
```

This is useful for understanding how a computer vision model can become part of an actual live application rather than only an offline notebook experiment.

---

## Class Recording

**YouTube:** https://youtu.be/H_DDf3NCV5M

See [`CLASS-RECORDING.md`](./CLASS-RECORDING.md) for the recording reference.

---

## Environment

The notebook is written for:

```text
Operating system: Windows
IDE: Google Antigravity
Recommended Python: 3.11 or 3.12
Execution: Local machine
Web endpoint: 127.0.0.1
```

The notebook creates a dedicated Python virtual environment and Jupyter kernel:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip ipykernel
python -m ipykernel install --user --name fastrtc-local --display-name "Python (FastRTC Local)"
```

---

## Dependencies

The practical notebook installs:

```text
fastrtc==0.0.34
ultralytics
supervision
```

FastRTC also brings the WebRTC/UI stack needed by the notebook, including Gradio and aiortc.

The notebook verifies versions for:

```text
fastrtc
gradio
ultralytics
supervision
aiortc
```

and checks whether CUDA is available.

---

## Model Configuration

The notebook intentionally uses:

```python
MODEL_NAME = "yolov8n.pt"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
```

`yolov8n.pt` is the Nano version of YOLOv8, which makes it suitable for webcam inference and more practical on CPU-constrained systems.

The first execution downloads the model weights automatically if they are not already present.

---

## Supervision Annotators

Two annotators are created:

```python
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(
    text_scale=0.5,
    text_thickness=1
)
```

The live visualization therefore includes:

- bounding boxes
- detected class names
- model confidence values

---

## Frame Processing

The main function is:

```python
process_frame(frame: np.ndarray) -> np.ndarray
```

For every incoming webcam frame, the notebook performs:

```text
NumPy frame
    ↓
YOLOv8 predict()
    ↓
Ultralytics Results
    ↓
sv.Detections.from_ultralytics()
    ↓
BoxAnnotator
    ↓
LabelAnnotator
    ↓
Annotated NumPy frame
```

The configured inference parameters are:

```python
CONFIDENCE = 0.30
IMAGE_SIZE = 640
```

---

## Detection Labels

When class IDs and confidence values are available, labels are created using:

```python
f"{model.names[int(class_id)]} {confidence:.2f}"
```

A rendered detection can therefore look conceptually like:

```text
person 0.91
car 0.84
bottle 0.73
```

---

## FastRTC Stream

The notebook creates the local video stream with:

```python
stream = Stream(
    handler=VideoStreamHandler(
        process_frame,
        skip_frames=True
    ),
    modality="video",
    mode="send-receive",
    rtc_configuration=None,
    concurrency_limit=1,
    time_limit=300,
)
```

### `skip_frames=True`

This is especially important in real-time computer vision.

Without frame skipping, a slow model can build up a queue of old frames and create increasing latency.

Conceptually:

```text
Camera produces frames faster than inference
                  ↓
          Unprocessed queue grows
                  ↓
              Video delay
```

With frame skipping:

```text
Keep processing recent frames
          ↓
Drop obsolete frames when necessary
          ↓
Lower accumulated latency
```

---

## Local Browser Launch

The interface is launched using:

```python
stream.ui.launch(
    server_name="127.0.0.1",
    share=False,
    inbrowser=True,
    debug=True,
)
```

Important properties:

```text
server_name = 127.0.0.1
share       = False
inbrowser   = True
```

This means the application stays local to the machine rather than exposing a public URL.

---

## Local vs. Remote Architecture

### Local Class 14 workflow

```text
Webcam
  ↓
Browser
  ↓
127.0.0.1
  ↓
FastRTC
  ↓
YOLOv8 + Supervision
  ↓
Browser
```

### Remote workflow

A remote/public deployment generally introduces additional concerns such as:

```text
HTTPS
TURN / STUN
Remote networking
Public hosting
Authentication / access control
```

The supplied notebook deliberately avoids those complications by remaining local.

---

## Troubleshooting from the Notebook

### Local page does not open

Use the URL printed by Gradio. If port `7860` is already occupied, another available port may be selected.

### Camera does not appear

Allow camera access for `127.0.0.1` in the browser and close other applications that may already be using the webcam.

### CUDA is `False`

The notebook can run on CPU. NVIDIA GPU execution requires a compatible PyTorch/CUDA installation for the local driver.

### Video is delayed

Keep:

```python
skip_frames=True
```

and, if necessary, reduce:

```python
IMAGE_SIZE = 480
```

or use a GPU.

### Access from another computer

`localhost` / `127.0.0.1` only serves the local machine. Remote access would require a different network/deployment configuration.

---

## Notebook

The supplied Class 14 notebook is preserved in:

[`notebook/class_14_supervision_fastrtc_antigravity.ipynb`](./notebook/class_14_supervision_fastrtc_antigravity.ipynb)

Notebook-specific documentation is available in:

[`notebook/README.md`](./notebook/README.md)

---

## Practical Validation

The source notebook and workflow are documented in the repository. A live webcam execution result has **not yet been claimed as verified** in this repository.

The next practical step is to run the notebook locally in Antigravity, verify the webcam stream, record the environment versions, and save screenshots or other evidence of the working live detection interface.

See:

[`practical/README.md`](./practical/README.md)

---

## Repository Structure

```text
14-Supervision-FastRTC/
├── README.md
├── CLASS-RECORDING.md
├── notebook/
│   ├── README.md
│   └── class_14_supervision_fastrtc_antigravity.ipynb
├── practical/
│   └── README.md
└── references/
    └── README.md
```

---

## Key Takeaways

- FastRTC can connect a local webcam to a Python computer vision processing function through WebRTC.
- YOLOv8 performs the object detection stage.
- `sv.Detections` keeps the Supervision workflow consistent with previous course sessions.
- `BoxAnnotator` and `LabelAnnotator` provide real-time visual feedback.
- Local execution on `127.0.0.1` avoids the additional infrastructure required by public WebRTC deployments.
- `skip_frames=True` is an important real-time strategy when inference cannot match camera FPS.
- GPU acceleration is optional because the notebook can fall back to CPU.

---

## Status

**Documentation:** Completed  
**Class recording:** Added  
**Notebook:** Added  
**Live local execution:** Pending verification

---

## Author

**Peyman Miyandashti**

- [GitHub — Peyman-mxli](https://github.com/Peyman-mxli)
- [LinkedIn — peyman-mxli](https://www.linkedin.com/in/peyman-mxli/)
