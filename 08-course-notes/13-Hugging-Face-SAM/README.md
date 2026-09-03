# 13 — Hugging Face + SAM 3

This session documents the optional advanced course notebook **“SAM 3 nativo desde HuggingFace”**, where I work with SAM 3 directly through the Hugging Face `transformers` API instead of relying only on the Ultralytics wrapper.

The main lesson is that **Supervision remains the common representation layer**: native SAM 3 outputs are converted into `sv.Detections`, after which the same annotators and visualization workflow used throughout the course can be reused.

---

## What I Learn

- Load `facebook/sam3` with `Sam3Processor` and `Sam3Model`
- Authenticate securely with Hugging Face
- Run native SAM 3 inference with text prompts
- Run native SAM 3 inference with bounding-box prompts
- Post-process native model outputs
- Convert `masks`, `boxes`, and `scores` into `sv.Detections`
- Visualize masks with `sv.MaskAnnotator`
- Compare text prompting with YOLO bounding-box prompting
- Explore multiple semantic concepts in one request
- Study the effect of confidence thresholds

---

## Why It Matters

Earlier sessions used higher-level wrappers because they make experimentation fast and convenient. This class shows what happens underneath that abstraction.

The native Hugging Face workflow is more explicit:

```text
Image + Prompt
      ↓
Sam3Processor
      ↓
Model Inputs
      ↓
Sam3Model
      ↓
Raw Model Output
      ↓
post_process_instance_segmentation()
      ↓
masks + boxes + scores
      ↓
Manual Conversion
      ↓
sv.Detections
      ↓
Supervision Visualization
```

This gives finer control over the model while preserving compatibility with the rest of my Computer Vision workflow.

---

## Native API vs. Ultralytics Wrapper

The course notebook contrasts the two approaches:

```text
Ultralytics
SAM("sam3.pt")(image, bboxes=...)
        ↓
Simple, compact wrapper

Hugging Face
processor(images, prompt)
        ↓
model()
        ↓
post_process_instance_segmentation()
        ↓
More explicit and flexible
```

The notebook specifically presents the native API as useful for broader prompt support, threshold control, batch inference, and direct model weights.

---

## Environment and Dependencies

The notebook installs and uses:

```text
transformers
torch
supervision
ultralytics
opencv-python
numpy
matplotlib
Pillow
huggingface_hub
```

It downloads the standard course test images:

```text
assets/bus.jpg
assets/zidane.jpg
```

The runtime device is selected automatically:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

---

## Hugging Face Authentication

The course requires:

- A Hugging Face account
- Approved access to `facebook/sam3`
- A Hugging Face access token

For repository safety, the committed notebook does **not** contain a hard-coded token.

Use:

```python
from huggingface_hub import login
login()
```

or:

```bash
huggingface-cli login
```

Never commit a real token to GitHub.

---

## Loading Native SAM 3

The processor prepares the inputs expected by the model, while the model generates segmentation predictions.

```python
processor = Sam3Processor.from_pretrained("facebook/sam3")
model = Sam3Model.from_pretrained("facebook/sam3").to(device)
model.eval()
```

---

## Text-Prompt Inference

The first practical example uses:

```text
Prompt: "person"
Image:  bus.jpg
```

The native inference flow is:

```python
inputs = processor(
    images=image_pil,
    text="person",
    return_tensors="pt"
).to(device)

with torch.no_grad():
    outputs = model(**inputs)
```

Then the model result is post-processed:

```python
results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=[image_pil.size[::-1]]
)[0]
```

---

## Native Output Structure

After post-processing, the notebook works with:

```text
masks
boxes
scores
```

These are converted from PyTorch tensors into NumPy arrays before being wrapped by Supervision.

```python
def sam3_a_detections(results: dict) -> sv.Detections:
    masks = results["masks"].cpu().numpy().astype(bool)
    xyxy = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()

    return sv.Detections(
        xyxy=xyxy,
        mask=masks,
        confidence=scores
    )
```

Conceptually:

```text
GPU Tensor
    ↓
CPU Tensor
    ↓
NumPy
    ↓
sv.Detections
```

---

## Supervision Visualization

Once the result becomes `sv.Detections`, Supervision can visualize it exactly like detections from other APIs.

The notebook uses:

```text
sv.MaskAnnotator
sv.LabelAnnotator
sv.ColorLookup.INDEX
```

The important architectural idea is:

```text
Different Model APIs
        ↓
sv.Detections
        ↓
Same Visualization Layer
```

---

## Experiment 1 — Multiple Concepts

The notebook tests multiple semantic concepts in one call:

```python
text=[["person", "vehicle"]]
```

with:

```text
threshold = 0.3
mask_threshold = 0.3
```

The resulting detections are converted to `sv.Detections` and visualized with the same mask pipeline.

---

## Experiment 2 — Text vs. Bounding Boxes

The lesson compares two prompting methods.

### Text Prompt

```text
"person"
```

### Bounding-Box Prompt

YOLOv8 first detects objects:

```python
yolo_model = YOLO("yolov8n.pt")
yolo_r = yolo_model(image_bgr)[0]
yolo_det = sv.Detections.from_ultralytics(yolo_r)
```

The YOLO boxes are then passed to native SAM 3:

```python
inputs_bbox = processor(
    images=image_pil,
    input_boxes=[boxes_list],
    input_boxes_labels=[[1] * len(boxes_list)],
    return_tensors="pt"
).to(device)
```

This experiment is designed to visually compare masks generated from semantic text prompting against masks generated from detector-provided boxes.

---

## Experiment 3 — Confidence Threshold

The notebook compares:

```text
threshold = 0.2
threshold = 0.5
threshold = 0.8
```

The lesson explains the intended trade-off:

```text
Lower threshold
      ↓
More detections
      ↓
Potentially more false positives

Higher threshold
      ↓
Fewer detections
      ↓
Only more confident results
```

---

## Repository Structure

```text
13-Hugging-Face-SAM/
│
├── README.md
├── CLASS-RECORDING.md
├── notebook/
│   └── class_13_huggingface_sam.ipynb
├── practical/
│   └── README.md
└── references/
    └── README.md
```

---

## Class Recording

[Watch Class 13 — Hugging Face + SAM](https://youtu.be/5gBU0k45R84)

The recording is documented separately in [CLASS-RECORDING.md](./CLASS-RECORDING.md).

---

## Original Course Notebook

The sanitized course notebook is preserved here:

[Open `class_13_huggingface_sam.ipynb`](./notebook/class_13_huggingface_sam.ipynb)

The notebook content is based on the supplied Class 13 course file. Only the exposed Hugging Face token was removed before publication.

---

## Learning Outcome

After studying this session, I understand how to move from a high-level SAM wrapper to the native Hugging Face API, inspect the native segmentation result, transform that result into the common `sv.Detections` representation, and continue using the same Supervision visualization workflow.

I also understand how text prompts, detector-generated bounding boxes, and confidence thresholds can change the segmentation workflow.

---

## Author

**Peyman Miyandashti**

- [GitHub — Peyman-mxli](https://github.com/Peyman-mxli)
- [LinkedIn — peyman-mxli](https://www.linkedin.com/in/peyman-mxli/)
