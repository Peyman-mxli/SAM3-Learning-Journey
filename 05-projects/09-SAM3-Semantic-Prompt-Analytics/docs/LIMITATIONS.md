# Project 09 Limitations

- The preserved measurements come from one image (`bus.jpg`).
- Detailed confidence and mask-area values were recorded only for `person`.
- Semantic prompts can return partial or ambiguous matches.
- Lower thresholds can increase false positives; higher thresholds can omit valid objects.
- Very small masks may be unreliable even when the concept is correct.
- Prompt wording can change the number and quality of returned masks.
- The SAM 3 checkpoint is external and must be available in Google Drive.
- GPU inference is recommended.
- The standalone project pipeline has passed syntax validation but has not yet been rerun end to end as a separate script.

These limitations are documented explicitly so notebook validation is not confused with standalone pipeline validation.
