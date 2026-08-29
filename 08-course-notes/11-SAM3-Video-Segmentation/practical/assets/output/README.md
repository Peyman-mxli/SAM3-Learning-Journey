# Generated Outputs

The practical script writes generated artifacts to this directory.

| File | Description |
|---|---|
| `vehicles_sam.mp4` | Full YOLO + ByteTrack + SAM 3 visualization |
| `vehicles_sam_zone.mp4` | Segmentation restricted to the polygon zone |
| `vehicles_sam_opacity.mp4` | Confidence-controlled mask opacity |
| `vehicles_sam_areas.mp4` | Mask visualization used during area analysis |
| `vehicles_text_prompts.mp4` | Direct semantic video segmentation |
| `mask_area_chart.png` | Temporal mask-area chart |
| `mask_areas.json` | Frame-level mask-area observations by tracker ID |

Generated files should be documented only after their execution and visual verification.
