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

## Experiments

1. Native text-prompt segmentation with `"person"`
2. Multiple concepts using `[["person", "vehicle"]]`
3. YOLOv8 detection followed by SAM 3 bounding-box prompting
4. Side-by-side comparison of text prompts and box prompts
5. Confidence-threshold comparison at `0.2`, `0.5`, and `0.8`

## Real Outputs

After a successful authenticated execution, the runner saves:

```text
outputs/
├── person_text_prompt.jpg
├── multi_concept_prompt.jpg
├── yolo_bbox_prompt.jpg
├── text_vs_bbox_comparison.jpg
├── threshold_comparison.jpg
└── execution_summary.json
```

The JSON summary records the real detection counts and execution environment rather than relying on manually entered values.

## Current Validation Status

The repository now contains an execution-ready workflow, but the actual `facebook/sam3` inference cannot be claimed as completed until the model is loaded under an approved, authenticated Hugging Face account and the generated artifacts are produced.

This distinction is intentional: the repository documents real execution evidence only and does not invent numeric results.

## Security

`facebook/sam3` is a gated Hugging Face model. Authentication must happen in the execution environment using your own approved account.

Never commit:

- Hugging Face access tokens
- Colab secret values
- API keys
- credential files

The original supplied notebook contained a hard-coded Hugging Face token. The repository version removes it and uses secure authentication instead.
