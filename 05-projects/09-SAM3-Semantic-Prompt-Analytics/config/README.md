# Configuration

`prompts.json` controls semantic concepts and reliability thresholds without modifying source code.

- `prompts` — natural-language concepts processed for each image
- `model_confidence` — SAM3 inference threshold
- `confidence_min` — minimum confidence for a reliable object
- `area_min` — minimum mask area in pixels
- `filtered_prompt` — prompt used for the labeled filtered visualization
