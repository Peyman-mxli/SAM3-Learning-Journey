# Practical — Native SAM 3 with Hugging Face

This folder documents the hands-on experiments included in Class 13.

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
MaskAnnotator + LabelAnnotator
```

## Experiments

1. Native text-prompt segmentation with `"person"`
2. Multiple concepts using `[["person", "vehicle"]]`
3. YOLOv8 detection followed by SAM 3 bounding-box prompting
4. Side-by-side comparison of text prompts and box prompts
5. Confidence-threshold comparison at `0.2`, `0.5`, and `0.8`

## Validation Note

The supplied notebook contains the code for these experiments, but the uploaded copy does not include trusted execution outputs that prove every experiment was successfully run in the current repository environment. For that reason, this documentation records the course workflow without inventing numeric results.

## Security

The original supplied notebook contained a hard-coded Hugging Face token. The repository version removes it and uses interactive authentication instead.
