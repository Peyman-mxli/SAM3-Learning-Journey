# Outputs — Class 13 Hugging Face + SAM 3

This folder contains the **real artifacts produced by executing the Class 13 practical workflow** with native SAM 3 through Hugging Face.

The experiment was run successfully in Google Colab with a Tesla T4 GPU after installing a CUDA-compatible PyTorch stack and authenticating with an approved Hugging Face account.

## Verified Generated Files

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

## Verified Results

### Text prompt — `person`

- Objects found: **4**
- Masks generated: **Yes**
- Boxes shape: `(4, 4)`
- Mask shape: `(4, 1080, 810)`

### Multi-concept prompt — `person`, `vehicle`

- Objects found: **5**
- Masks generated: **Yes**
- Boxes shape: `(5, 4)`
- Mask shape: `(5, 1080, 810)`

### YOLOv8 detections

YOLOv8 detected **6 objects**:

- 1 bus
- 4 persons
- 1 stop sign

### SAM 3 with YOLO bounding-box prompts

- Objects segmented: **6**
- Masks generated: **Yes**
- Boxes shape: `(6, 4)`
- Mask shape: `(6, 1080, 810)`

### Confidence-threshold comparison

| Threshold | Objects found |
| ---: | ---: |
| 0.2 | 5 |
| 0.5 | 4 |
| 0.8 | 4 |

## Artifact Meaning

- `person_text_prompt.jpg` — native SAM 3 segmentation using the text prompt `person`.
- `multi_concept_prompt.jpg` — segmentation using the concepts `person` and `vehicle`.
- `yolo_bbox_prompt.jpg` — SAM 3 segmentation driven by bounding boxes detected with YOLOv8.
- `text_vs_bbox_comparison.jpg` — side-by-side visual comparison between text prompting and YOLO bounding-box prompting.
- `threshold_comparison.jpg` — visual comparison of confidence thresholds `0.2`, `0.5`, and `0.8`.
- `execution_summary.json` — machine-readable record of the verified execution results.

## Reproducibility

Use:

```bash
python practical/run_class13.py
```

The environment must already be authenticated with Hugging Face and have approved access to `facebook/sam3`.

## Security

No Hugging Face access token, Colab secret, API key, or credential file is committed to this repository.
