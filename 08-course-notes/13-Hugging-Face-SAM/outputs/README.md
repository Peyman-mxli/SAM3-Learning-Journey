# Outputs — Class 13 Hugging Face + SAM 3

This folder contains **real artifacts produced from an authenticated Google Colab execution of Class 13** using the gated Hugging Face model `facebook/sam3` on a Tesla T4 GPU.

The results documented here come from the actual practical run and are not estimated or invented.

## Verified Execution

```text
Model:       facebook/sam3
Device:      CUDA
GPU:         Tesla T4
Image:       bus.jpg
Runtime:     Google Colab
```

The working PyTorch stack used during the successful run was:

```text
Torch:       2.11.0+cu128
TorchVision: 0.26.0+cu128
TorchAudio:  2.11.0+cu128
CUDA:        12.8
```

## Verified Results

| Experiment | Result |
|---|---:|
| Text prompt `person` | 4 objects |
| Multi-concept prompt `person + vehicle` | 5 objects |
| YOLOv8 detections | 6 objects |
| SAM 3 from YOLO bounding boxes | 6 masks |
| Threshold `0.2` | 5 objects |
| Threshold `0.5` | 4 objects |
| Threshold `0.8` | 4 objects |

YOLOv8 detected:

```text
1 bus
4 persons
1 stop sign
```

## Output Files

The complete Colab run produced:

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

`execution_summary.json` is committed in this folder and contains the verified numeric results from the run.

## Artifact Meaning

- `person_text_prompt.jpg` — native SAM 3 segmentation using the text prompt `person`.
- `multi_concept_prompt.jpg` — SAM 3 segmentation using `person` and `vehicle` in the same request.
- `yolo_bbox_prompt.jpg` — SAM 3 masks generated from YOLOv8 bounding-box prompts.
- `text_vs_bbox_comparison.jpg` — side-by-side comparison of native text prompting and YOLO-provided bounding boxes.
- `threshold_comparison.jpg` — visual comparison at thresholds `0.2`, `0.5`, and `0.8`.
- `execution_summary.json` — machine-readable verified counts and confidence values.

## Reproducibility

The practical workflow can be repeated with:

```bash
python practical/run_class13.py
```

The environment must be authenticated with Hugging Face and the account must have approved access to `facebook/sam3`.

## Important Compatibility Note

During the Colab setup, a CUDA mismatch between PyTorch and TorchAudio initially prevented `Sam3Processor` from importing. The working stack was restored by installing matching CUDA 12.8 builds of `torch`, `torchvision`, and `torchaudio`.

## Security

No Hugging Face token, Colab secret value, API key, or credential file is stored in this repository.
