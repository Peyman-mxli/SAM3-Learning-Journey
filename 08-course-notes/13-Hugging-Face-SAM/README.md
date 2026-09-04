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

## Quick Start

Recommended execution flow:

```text
1. Open the notebook in Google Colab
2. Enable a GPU runtime
3. Install compatible dependencies
4. Authenticate with Hugging Face
5. Load facebook/sam3
6. Download the test images
7. Run text-prompt inference
8. Post-process the native outputs
9. Convert the results to sv.Detections
10. Visualize the masks and labels
11. Run the YOLOv8 + SAM 3 bounding-box experiment
12. Compare prompt types and confidence thresholds
```

The notebook-specific guide is available in [`notebook/README.md`](./notebook/README.md).

---

## Verified Execution

This class has now been successfully executed in Google Colab with the gated native Hugging Face SAM 3 model.

```text
Model:       facebook/sam3
GPU:         Tesla T4
CUDA:        12.8
Torch:       2.11.0+cu128
TorchVision: 0.26.0+cu128
TorchAudio:  2.11.0+cu128
```

The practical run completed text prompting, multi-concept prompting, YOLOv8 bounding-box prompting, Supervision visualization, and confidence-threshold comparison.

### Verified Results

| Experiment | Result |
|---|---:|
| Text prompt `person` | 4 objects |
| Multi-concept `person + vehicle` | 5 objects |
| YOLOv8 detection | 6 objects |
| SAM 3 with YOLO boxes | 6 masks |
| Threshold `0.2` | 5 objects |
| Threshold `0.5` | 4 objects |
| Threshold `0.8` | 4 objects |

YOLOv8 detected one bus, four persons, and one stop sign.

The verified machine-readable summary is stored in [`outputs/execution_summary.json`](./outputs/execution_summary.json).

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

The native API provides broader prompt support, threshold control, batch inference, and direct access to the model weights.

---

## Environment and Dependencies

The notebook uses:

```text
transformers
torch
torchvision
torchaudio
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

The real run returned **4 objects**, all with masks.

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

The notebook uses:

```text
sv.MaskAnnotator
sv.LabelAnnotator
sv.ColorLookup.INDEX
```

The architectural idea remains:

```text
Different Model APIs
        ↓
sv.Detections
        ↓
Same Visualization Layer
```

---

## Experiment 1 — Multiple Concepts

The notebook tests:

```python
text=[["person", "vehicle"]]
```

with:

```text
threshold = 0.3
mask_threshold = 0.3
```

The verified run returned **5 objects**.

---

## Experiment 2 — Text vs. Bounding Boxes

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

YOLOv8 detected **6 objects**, and SAM 3 produced **6 masks** from those bounding-box prompts.

---

## Experiment 3 — Confidence Threshold

The real comparison produced:

```text
threshold = 0.2 → 5 objects
threshold = 0.5 → 4 objects
threshold = 0.8 → 4 objects
```

This demonstrates the expected trade-off: a lower threshold allows more predictions to survive post-processing.

---

## Troubleshooting

### PyTorch / TorchAudio CUDA mismatch

During the real Colab run, SAM 3 initially failed to import because PyTorch was built for CUDA 13.0 while TorchAudio was built for CUDA 12.8.

The working solution was to install matching CUDA 12.8 builds:

```text
Torch:       2.11.0+cu128
TorchVision: 0.26.0+cu128
TorchAudio:  2.11.0+cu128
```

### Gated model or access denied

Make sure the Hugging Face account used for authentication has permission to access `facebook/sam3`.

### Invalid or missing Hugging Face token

Run interactive authentication again and avoid storing the token directly in notebook source code.

### CUDA is unavailable

Check the Colab runtime configuration and confirm that a GPU hardware accelerator is enabled.

### GPU memory error

Restart the runtime, clear unused variables when needed, and avoid loading unnecessary models at the same time.

### Empty detections

A high confidence threshold can remove valid predictions. Test lower thresholds and compare the masks.

### Tensor conversion problems

```python
tensor.cpu().numpy()
```

### Mask or box shape mismatch

Verify that post-processing uses the original image size and that masks, boxes, and scores correspond to the same prediction batch.

---

## Real Output Artifacts

The completed Colab execution generated:

```text
outputs/
├── README.md
├── person_text_prompt.jpg
├── multi_concept_prompt.jpg
├── yolo_bbox_prompt.jpg
├── text_vs_bbox_comparison.jpg
├── threshold_comparison.jpg
└── execution_summary.json
```

The output guide and verified measurements are documented in [`outputs/README.md`](./outputs/README.md).

---

## Key Takeaways

- Hugging Face exposes SAM 3 at a lower and more explicit abstraction level than a high-level wrapper.
- `Sam3Processor` prepares image and prompt inputs for the model.
- Native outputs require post-processing before they become useful segmentation results.
- `sv.Detections` provides a common representation for different model APIs.
- Text prompts and detector-generated bounding boxes offer different ways to guide segmentation.
- Lower confidence thresholds can preserve additional predictions.
- Matching PyTorch ecosystem CUDA builds is important for a stable runtime.
- Authentication secrets must stay outside committed source code.

---

## Repository Structure

```text
13-Hugging-Face-SAM/
│
├── README.md
├── CLASS-RECORDING.md
├── notebook/
│   ├── README.md
│   └── class_13_huggingface_sam.ipynb
├── practical/
│   ├── README.md
│   └── run_class13.py
├── outputs/
│   ├── README.md
│   └── execution_summary.json
└── references/
    └── README.md
```

---

## Class Recording

[Watch Class 13 — Hugging Face + SAM](https://youtu.be/w9DE806d4oU)

The recording is documented separately in [CLASS-RECORDING.md](./CLASS-RECORDING.md).

---

## Original Course Notebook

The sanitized course notebook is preserved here:

[Open `class_13_huggingface_sam.ipynb`](./notebook/class_13_huggingface_sam.ipynb)

The notebook content is based on the supplied Class 13 course file. The exposed Hugging Face token was removed before publication.

---

## Learning Outcome

After completing this session, I can load native SAM 3 through Hugging Face, authenticate securely, run semantic text prompting, convert native results into `sv.Detections`, use YOLOv8 bounding boxes as SAM 3 prompts, visualize segmentation masks with Supervision, and analyze the effect of confidence thresholds using verified practical results.

---

## Author

**Peyman Miyandashti**

- [GitHub — Peyman-mxli](https://github.com/Peyman-mxli)
- [LinkedIn — peyman-mxli](https://www.linkedin.com/in/peyman-mxli/)
