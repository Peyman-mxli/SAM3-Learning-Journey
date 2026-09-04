# References — Class 14 Supervision + FastRTC

## Class Recording

https://youtu.be/H_DDf3NCV5M

## Technologies Used in the Supplied Notebook

- FastRTC
- Gradio
- aiortc
- Ultralytics YOLOv8
- Supervision
- PyTorch
- NumPy

## Notebook Model

```text
yolov8n.pt
```

## Local Architecture

```text
Webcam
   ↓
WebRTC
   ↓
FastRTC
   ↓
YOLOv8
   ↓
Supervision
   ↓
127.0.0.1 browser interface
```

## Note

This reference file documents only technologies and settings present in the supplied Class 14 notebook. Real execution versions and hardware results should be added after the notebook is run locally.
