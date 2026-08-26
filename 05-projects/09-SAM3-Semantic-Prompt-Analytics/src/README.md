# Source Code

`semantic_prompt_analytics.py` implements the complete Project 09 pipeline.

Responsibilities:

- Load configuration
- Discover input images
- Initialize `SAM3SemanticPredictor`
- Reuse an image across multiple prompts
- Calculate confidence and mask area
- Classify reliable detections
- Generate filtered and comparison visualizations
- Export JSON and CSV analytics

The models are loaded once and reused across all configured inputs.
