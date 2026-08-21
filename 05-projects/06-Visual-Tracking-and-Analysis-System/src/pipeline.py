"""
pipeline.py

Integrated computer vision pipeline for the
Visual Tracking and Analysis System.

This module combines:

- YOLO object detection
- ByteTrack object tracking
- SAM 3 text-prompt segmentation
- Supervision visualization

The pipeline is designed first for image-based
integration testing before extending to full
recorded-video processing.
"""

from PIL import Image

from src.detector import ObjectDetector
from src.tracker import ObjectTracker
from src.segmenter import ObjectSegmenter
from src.visualization import TrackingVisualizer


class VisualAnalysisPipeline:
    """
    Integrated computer vision pipeline.

    Parameters
    ----------
    sam3_checkpoint_path : str
        Local path to the SAM 3 checkpoint.

    model_name : str
        Ultralytics YOLO model name.

    confidence_threshold : float
        Minimum YOLO confidence threshold.

    device : str
        Device used by SAM 3.
    """

    def __init__(
        self,
        sam3_checkpoint_path,
        model_name="yolov8n.pt",
        confidence_threshold=0.50,
        device="cuda"
    ):
        self.detector = ObjectDetector(
            model_name=model_name,
            confidence_threshold=confidence_threshold
        )

        self.tracker = ObjectTracker()

        self.segmenter = ObjectSegmenter(
            checkpoint_path=sam3_checkpoint_path,
            device=device
        )

        self.visualizer = TrackingVisualizer()

    def process_image(
        self,
        image_bgr,
        segmentation_prompt=None
    ):
        """
        Process one image through the integrated pipeline.

        Parameters
        ----------
        image_bgr
            OpenCV image in BGR format.

        segmentation_prompt : str or None
            Optional SAM 3 text prompt.

            Example:

            "person"

        Returns
        -------
        dict
            Dictionary containing detections,
            tracked detections, segmentation
            output, and annotated image.
        """

        # ------------------------------------------
        # Object detection
        # ------------------------------------------

        detections = self.detector.detect(
            image_bgr
        )

        # ------------------------------------------
        # Object tracking
        # ------------------------------------------

        tracked_detections = self.tracker.update(
            detections
        )

        # ------------------------------------------
        # Visualization
        # ------------------------------------------

        class_names = self.detector.get_class_names()

        annotated_image = self.visualizer.annotate(
            image=image_bgr,
            detections=tracked_detections,
            class_names=class_names
        )

        # ------------------------------------------
        # SAM 3 segmentation
        # ------------------------------------------

        segmentation_output = None

        if segmentation_prompt:
            image_rgb = image_bgr[:, :, ::-1]

            pil_image = Image.fromarray(
                image_rgb
            )

            segmentation_output = (
                self.segmenter.segment_with_text_prompt(
                    image=pil_image,
                    prompt=segmentation_prompt
                )
            )

        # ------------------------------------------
        # Final structured result
        # ------------------------------------------

        return {
            "detections": detections,
            "tracked_detections": tracked_detections,
            "segmentation": segmentation_output,
            "annotated_image": annotated_image
        }
