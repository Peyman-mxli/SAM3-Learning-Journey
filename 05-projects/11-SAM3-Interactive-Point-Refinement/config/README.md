# Configuration

`pipeline.json` controls the maximum number of YOLO-discovered objects, negative-point offset, mask opacity, and YOLO checkpoint.

The SAM checkpoint is intentionally not stored in configuration. Set it through `SAM3_MODEL_PATH`; otherwise the pipeline uses `/content/drive/MyDrive/SAM3-Models/sam3.pt`.
