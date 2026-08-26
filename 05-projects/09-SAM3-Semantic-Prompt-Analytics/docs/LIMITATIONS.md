# Project 09 Limitations

- The preserved measurements come from one image (`bus.jpg`).
- Detailed confidence and mask-area values were recorded only for `person`.
- Semantic prompts can return partial or ambiguous matches.
- Lower thresholds can increase false positives; higher thresholds can omit valid objects.
- Very small masks may be unreliable even when the concept is correct.
- Prompt wording can change the number and quality of returned masks.
- The SAM 3 checkpoint is external and must be available in Google Drive.
- GPU inference is recommended.
- Ultralytics currently warns that the `half` option is deprecated and recommends its newer quantization configuration.

The standalone pipeline has now been validated end to end. These remaining limitations describe scope and future compatibility rather than incomplete execution.
