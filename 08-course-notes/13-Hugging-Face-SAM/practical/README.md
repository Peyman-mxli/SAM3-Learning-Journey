# Practical — Native SAM 3 with Hugging Face

This folder contains the hands-on execution workflow for Class 13 and now documents a **successfully completed authenticated run** of native SAM 3 through Hugging Face.

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
YOLOv8 Bounding-Box Prompting
        ↓
Threshold Comparison
        ↓
Save Real Output Artifacts
```

## Execution-Ready Runner

The reproducible runner is:

```text
practical/run_class13.py
```

Run it from the Class 13 folder after installing compatible dependencies and authenticating with Hugging Face:

```bash
python practical/run_class13.py
```

The script does **not** contain or save a Hugging Face token.

## Verified Runtime

The practical was successfully executed in Google Colab with:

```text
GPU:         Tesla T4
CUDA:        12.8
Torch:       2.11.0+cu128
TorchVision: 0.26.0+cu128
TorchAudio:  2.11.0+cu128
Model:       facebook/sam3
```

A previous CUDA mismatch between PyTorch 13.0 and TorchAudio 12.8 caused the SAM 3 import to fail. Installing matching CUDA 12.8 builds resolved the issue.

## Experiments and Real Results

### 1. Native text prompt — `person`

SAM 3 found **4 objects** and produced masks for all four.

Confidence values:

```text
0.9657163
0.9512506
0.97732687
0.9753788
```

### 2. Multiple concepts — `person` + `vehicle`

SAM 3 found **5 objects**.

Confidence values:

```text
0.72931856
0.7100393
0.7358877
0.39503664
0.7390171
```

### 3. YOLOv8 → SAM 3 bounding-box prompting

YOLOv8 found **6 objects**:

```text
bus         0.873
person      0.866
person      0.853
person      0.825
person      0.261
stop sign   0.255
```

The six YOLO boxes were then sent to native SAM 3. SAM 3 returned **6 segmentation masks**.

SAM 3 confidence values:

```text
0.97003
0.97052
0.95566
0.94650
0.97990
0.94823
```

### 4. Text vs. bounding-box comparison

The side-by-side experiment confirms that the same image can be segmented through two different prompting strategies:

```text
Text semantic prompt
        vs.
Detector-provided bounding boxes
```

This demonstrates why `sv.Detections` is useful as a common representation between different model APIs.

### 5. Confidence-threshold comparison

The actual run produced:

| Threshold | Objects |
|---:|---:|
| `0.2` | 5 |
| `0.5` | 4 |
| `0.8` | 4 |

The result shows the expected behavior: lowering the threshold allowed one additional prediction to survive post-processing.

## Real Output Set

The completed Colab run generated:

```text
outputs/
├── person_text_prompt.jpg
├── multi_concept_prompt.jpg
├── yolo_bbox_prompt.jpg
├── text_vs_bbox_comparison.jpg
├── threshold_comparison.jpg
└── execution_summary.json
```

The verified machine-readable results are stored in [`../outputs/execution_summary.json`](../outputs/execution_summary.json).

## Validation Status

**Completed.** The native `facebook/sam3` model was successfully loaded under an approved and authenticated Hugging Face account, inference ran on CUDA, all three segmentation workflows completed, and the threshold experiment produced verified numeric results.

## Security

`facebook/sam3` is a gated Hugging Face model. Authentication must happen in the execution environment using an approved account.

Never commit:

- Hugging Face access tokens
- Colab secret values
- API keys
- credential files

The repository contains no authentication secret from this practical run.
