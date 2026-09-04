"""Class 14 — Local live detection with FastRTC + YOLOv8 + Supervision.

Run this script locally on Windows after activating the Class 14 virtual
environment. It starts a local WebRTC/Gradio application on 127.0.0.1.

Source workflow: class_14_supervision_fastrtc_antigravity.ipynb
"""

from importlib.metadata import version
import sys

import numpy as np
import supervision as sv
import torch
from fastrtc import Stream, VideoStreamHandler
from ultralytics import YOLO

MODEL_NAME = "yolov8n.pt"
CONFIDENCE = 0.30
IMAGE_SIZE = 640
DEVICE = 0 if torch.cuda.is_available() else "cpu"


def print_environment() -> None:
    print("Python executable:", sys.executable)
    print("Python version:", sys.version.split()[0])
    for package in ("fastrtc", "gradio", "ultralytics", "supervision", "aiortc"):
        print(f"{package}: {version(package)}")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print(f"Model: {MODEL_NAME} | device: {DEVICE}")


print_environment()

model = YOLO(MODEL_NAME)
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)


def process_frame(frame: np.ndarray) -> np.ndarray:
    if frame is None:
        return frame

    result = model.predict(
        source=frame,
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        verbose=False,
    )[0]

    detections = sv.Detections.from_ultralytics(result)

    annotated = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections,
    )

    if detections.class_id is not None and detections.confidence is not None:
        labels = [
            f"{model.names[int(class_id)]} {confidence:.2f}"
            for class_id, confidence in zip(
                detections.class_id,
                detections.confidence,
            )
        ]

        annotated = label_annotator.annotate(
            scene=annotated,
            detections=detections,
            labels=labels,
        )

    return annotated


stream = Stream(
    handler=VideoStreamHandler(process_frame, skip_frames=True),
    modality="video",
    mode="send-receive",
    rtc_configuration=None,
    concurrency_limit=1,
    time_limit=300,
    ui_args={
        "hide_title": True,
        "full_screen": False,
    },
)

stream.ui.launch(
    server_name="127.0.0.1",
    share=False,
    inbrowser=True,
    debug=True,
)
