from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class SegmentationResult:
    prompt: str
    masks: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray


class SAM3TextSegmenter:
    """
    Thin adapter around the SAM 3 image processor used in the course environment.

    The SAM 3 repository/checkpoint must already be available locally.
    This module deliberately fails clearly when those dependencies are absent.
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        try:
            import torch
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as exc:
            raise RuntimeError(
                "SAM 3 is not installed. Use the course/Colab SAM 3 environment "
                "and provide a local checkpoint before running segmentation."
            ) from exc

        self.torch = torch
        self.device = device
        self.model = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            device=device,
            load_from_HF=False,
            enable_segmentation=True,
        )
        self.processor = Sam3Processor(self.model)

    def segment_text(self, image_rgb: np.ndarray, prompt: str) -> SegmentationResult:
        image = Image.fromarray(image_rgb)

        autocast_enabled = self.device.startswith("cuda")
        if autocast_enabled:
            context = self.torch.autocast(
                device_type="cuda",
                dtype=self.torch.bfloat16,
            )
        else:
            from contextlib import nullcontext
            context = nullcontext()

        with context:
            state = self.processor.set_image(image)
            output = self.processor.set_text_prompt(
                state=state,
                prompt=prompt,
            )

        masks = np.asarray(output.get("masks", []))
        boxes = np.asarray(output.get("boxes", []))
        scores = np.asarray(output.get("scores", []))

        return SegmentationResult(
            prompt=prompt,
            masks=masks,
            boxes=boxes,
            scores=scores,
        )
