# Notebook — Class 13 Hugging Face + SAM 3

This folder contains the sanitized Class 13 notebook used to study **native SAM 3 inference through Hugging Face Transformers**.

## Notebook

- `class_13_huggingface_sam.ipynb`

## Recommended Runtime

Use **Google Colab with a GPU runtime** whenever available.

```text
Runtime
└── Change runtime type
    └── Hardware accelerator: GPU
```

The notebook automatically selects CUDA when available:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

## Before Running

Make sure you have:

- A Hugging Face account
- Access to `facebook/sam3`
- A valid Hugging Face access token
- Internet access for downloading model weights and test images
- Enough GPU memory for model inference

## Authentication

Do not hard-code a real Hugging Face token inside the notebook.

Use interactive authentication instead:

```python
from huggingface_hub import login
login()
```

or:

```bash
huggingface-cli login
```

## Dependencies

The notebook uses the following main packages:

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

## Execution Order

```text
1. Open the notebook in Google Colab
2. Enable a GPU runtime
3. Install/import dependencies
4. Authenticate with Hugging Face
5. Load Sam3Processor and Sam3Model
6. Download/load the course test images
7. Run text-prompt segmentation
8. Post-process native SAM 3 outputs
9. Convert masks, boxes, and scores to sv.Detections
10. Visualize with Supervision
11. Run multiple-concept prompting
12. Run YOLOv8 detection + SAM 3 box prompting
13. Compare text prompts vs. bounding-box prompts
14. Compare confidence thresholds
```

## Test Images

The class notebook uses standard Ultralytics sample images:

```text
assets/bus.jpg
assets/zidane.jpg
```

## Main Output Flow

```text
Image + Prompt
      ↓
Sam3Processor
      ↓
Sam3Model
      ↓
Raw Outputs
      ↓
post_process_instance_segmentation()
      ↓
masks + boxes + scores
      ↓
NumPy Conversion
      ↓
sv.Detections
      ↓
Supervision Visualization
```

## Security Note

The repository notebook is sanitized for publication. A real access token must never be committed to GitHub, printed in outputs, or stored directly in source code.

## Validation Note

The notebook documents the Class 13 workflow, but repository documentation does not claim numeric results unless they are supported by trusted execution outputs.

## Related Documentation

- [Main Class 13 README](../README.md)
- [Practical Notes](../practical/README.md)
- [References](../references/README.md)
- [Class Recording](../CLASS-RECORDING.md)
