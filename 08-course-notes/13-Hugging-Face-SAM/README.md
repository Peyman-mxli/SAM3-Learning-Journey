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

## Verified Execution Status

**Completed successfully in Google Colab with a Tesla T4 GPU.**

Verified environment:

```text
Torch:       2.11.0+cu128
TorchVision: 0.26.0+cu128
TorchAudio:  2.11.0+cu128
CUDA:        12.8
GPU:         Tesla T4
Model:       facebook/sam3
```

The native SAM 3 model loaded successfully after Hugging Face authentication.

### Verified Results

| Experiment | Result |
| --- | ---: |
| Text prompt `person` | 4 objects |
| Multi-concept prompt `person`, `vehicle` | 5 objects |
| YOLOv8 detections | 6 objects |
| SAM 3 using YOLO bounding boxes | 6 masks |
| Threshold `0.2` | 5 objects |
| Threshold `0.5` | 4 objects |
| Threshold `0.8` | 4 objects |

YOLOv8 detected **1 bus, 4 persons, and 1 stop sign**. SAM 3 generated segmentation masks from all 6 detector-provided bounding boxes.

The real visual artifacts and machine-readable summary are stored in [`outputs/`](./outputs/).

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
13. Save the real artifacts to outputs/
```

The notebook-specific guide is available in [`notebook/README.md`](./notebook/README.md).

---

## Native API vs. Ultralytics Wrapper

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

The native API provides broader prompt support, threshold control, batch inference, and direct access to model weights.

---

## Hugging Face Authentication

The course requires:

- A Hugging Face account
- Approved access to `facebook/sam3`
- A Hugging Face access token

Use interactive authentication:

```python
from huggingface_hub import login
login()
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

The first verified practical example uses:

```text
Prompt: "person"
Image:  bus.jpg
Result: 4 objects
```

---

## Native Output Structure

After post-processing, the notebook works with:

```text
masks
boxes
scores
```

These are converted into `sv.Detections` for the Supervision workflow.

---

## Experiment 1 — Multiple Concepts

```python
text=[["person", "vehicle"]]
```

Verified result: **5 objects**.

---

## Experiment 2 — Text vs. Bounding Boxes

YOLOv8 first detected **6 objects**. Those boxes were then passed to native SAM 3, which produced **6 masks**.

---

## Experiment 3 — Confidence Threshold

Verified counts:

```text
threshold = 0.2 → 5 objects
threshold = 0.5 → 4 objects
threshold = 0.8 → 4 objects
```

---

## Runtime Issue Resolved

An initial import error occurred because PyTorch used CUDA 13.0 while TorchAudio used CUDA 12.8.

The runtime was fixed by installing a matching CUDA 12.8 PyTorch stack. After restarting Colab, `Sam3Processor` and `Sam3Model` imported and loaded successfully.

---

## Real Outputs

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

These files represent real execution evidence from the Class 13 workflow.

---

## Repository Structure

```text
13-Hugging-Face-SAM/
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
│   ├── person_text_prompt.jpg
│   ├── multi_concept_prompt.jpg
│   ├── yolo_bbox_prompt.jpg
│   ├── text_vs_bbox_comparison.jpg
│   ├── threshold_comparison.jpg
│   └── execution_summary.json
└── references/
    └── README.md
```

---

## Class Recording

[Watch Class 13 — Hugging Face + SAM](https://youtu.be/w9DE806d4oU)

---

## Learning Outcome

After completing this session, I understand how to use native SAM 3 through Hugging Face, convert its outputs into `sv.Detections`, compare semantic text prompting with detector-provided bounding boxes, and evaluate how confidence thresholds change the number of retained detections.

I also documented a real CUDA dependency mismatch and its resolution, making this session reproducible and technically verified.

---

## Author

**Peyman Miyandashti**

- [GitHub — Peyman-mxli](https://github.com/Peyman-mxli)
- [LinkedIn — peyman-mxli](https://www.linkedin.com/in/peyman-mxli/)
