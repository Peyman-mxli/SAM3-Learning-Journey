# Practical — Native SAM 3 with Hugging Face

This folder contains the hands-on execution workflow for Class 13.

## Practical Flow

```text
Hugging Face Authentication
        ↓
Load Sam3Processor + Sam3Model
        ↓
Load bus.jpg
        ↓
Text Prompt: "person"
        ↓
Native SAM 3 Inference
        ↓
Post-Process Instance Segmentation
        ↓
masks + boxes + scores
        ↓
Convert to sv.Detections
        ↓
Supervision Annotation
        ↓
Save Real Output Artifacts
```

## Execution-Ready Runner

The reproducible runner is:

```text
practical/run_class13.py
```

Run it from the Class 13 folder after installing the dependencies and authenticating with Hugging Face:

```bash
python practical/run_class13.py
```

The script does **not** contain or save a Hugging Face token.

## Experiments Completed

1. Native text-prompt segmentation with `"person"`
2. Multiple concepts using `[["person", "vehicle"]]`
3. YOLOv8 detection followed by SAM 3 bounding-box prompting
4. Side-by-side comparison of text prompts and box prompts
5. Confidence-threshold comparison at `0.2`, `0.5`, and `0.8`

## Verified Practical Results

| Experiment | Result |
| --- | --- |
| Text prompt `person` | 4 objects |
| Multi-concept `person`, `vehicle` | 5 objects |
| YOLOv8 detection | 6 objects |
| SAM 3 bbox prompts | 6 masks |
| Threshold `0.2` | 5 objects |
| Threshold `0.5` | 4 objects |
| Threshold `0.8` | 4 objects |

YOLOv8 detected 1 bus, 4 persons, and 1 stop sign. SAM 3 then generated masks for all 6 YOLO-provided bounding boxes.

## Real Outputs

The authenticated Colab execution produced:

```text
outputs/
├── person_text_prompt.jpg
├── multi_concept_prompt.jpg
├── yolo_bbox_prompt.jpg
├── text_vs_bbox_comparison.jpg
├── threshold_comparison.jpg
└── execution_summary.json
```

These are real execution artifacts from the Class 13 workflow.

## Validation Status

**Completed and verified.**

`facebook/sam3` was loaded successfully on a Tesla T4 GPU using an approved Hugging Face account. Native text prompting, multi-concept prompting, YOLO bounding-box prompting, and threshold experiments were all executed successfully.

## Environment Fix Recorded During Execution

A CUDA mismatch initially occurred because PyTorch used CUDA 13.0 while TorchAudio used CUDA 12.8. The runtime was corrected by installing a matching CUDA 12.8 stack:

```text
Torch:      2.11.0+cu128
TorchVision: 0.26.0+cu128
TorchAudio:  2.11.0+cu128
CUDA:       12.8
GPU:        Tesla T4
```

After the fix, `Sam3Processor` and `Sam3Model` imported and loaded successfully.

## Security

`facebook/sam3` is a gated Hugging Face model. Authentication happens only inside the execution environment.

Never commit:

- Hugging Face access tokens
- Colab secret values
- API keys
- credential files

No authentication secret is stored in this repository.
