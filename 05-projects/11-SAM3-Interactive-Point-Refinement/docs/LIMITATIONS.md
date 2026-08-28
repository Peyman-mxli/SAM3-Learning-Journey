# Limitations

- YOLO box centers are convenient prompts but are not guaranteed to fall on the intended semantic region.
- A fixed negative-point offset may land on another foreground object in crowded scenes.
- Mask confidence is model-specific and is not calibrated probability.
- Results depend on the YOLO and SAM checkpoints, input resolution, and hardware.
- The current pipeline analyzes one image and up to three discovered objects.
- Point prompts require spatial intent; they do not replace semantic text discovery in every workflow.

Validate prompts visually before using results in consequential applications.
