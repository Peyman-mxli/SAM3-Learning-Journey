"""
segmenter.py

SAM 3 segmentation module for the Visual Tracking and Analysis System.

This module uses Meta SAM 3 for prompt-based image segmentation.
It supports text-prompt segmentation and uses CUDA bfloat16 autocast
to avoid dtype mismatch issues observed during Colab inference.
"""

import torch

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class ObjectSegmenter:
    """
    SAM 3 based image segmenter.

    Parameters
    ----------
    checkpoint_path : str
        Local path to the SAM 3 checkpoint.

    device : str
        Torch device used for inference.
    """

    def __init__(
        self,
        checkpoint_path,
        device="cuda"
    ):
        self.device = device

        self.model = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            device=self.device,
            load_from_HF=False,
            enable_segmentation=True
        )

        self.processor = Sam3Processor(
            self.model
        )

    def segment_with_text_prompt(
        self,
        image,
        prompt
    ):
        """
        Segment objects using a text prompt.

        Parameters
        ----------
        image : PIL.Image.Image
            RGB image used for SAM 3 inference.

        prompt : str
            Text description of the target object.

        Returns
        -------
        dict
            SAM 3 output containing masks, boxes, and scores.
        """

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        ):
            inference_state = self.processor.set_image(
                image
            )

            output = self.processor.set_text_prompt(
                state=inference_state,
                prompt=prompt
            )

        return output
