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
        """
        Initialize all components required by the pipeline.
        """

        # ------------------------------------------
        # Object detector
        # ------------------------------------------

        self.detector = ObjectDetector(
            model_name=model_name,
            confidence_threshold=confidence_threshold
        )

        # ------------------------------------------
        # Object tracker
        # ------------------------------------------

        self.tracker = ObjectTracker()

        # ------------------------------------------
        # SAM 3 segmenter
        # ------------------------------------------

        self.segmenter = ObjectSegmenter(
            checkpoint_path=sam3_checkpoint_path,
            device=device
        )

        # ------------------------------------------
        # Visualization
        # ------------------------------------------

        self.visualizer = TrackingVisualizer()

    def process_image(
        self,
        image_bgr,
        segmentation_prompt=None
    ):
        """
        Process one image through the complete pipeline.

        The processing order is:

        1. YOLO object detection
        2. ByteTrack object tracking
        3. SAM 3 segmentation
        4. Combined visualization

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
            Dictionary containing:

            - detections
            - tracked detections
            - SAM 3 segmentation output
            - final annotated image
        """

        # ------------------------------------------
        # Validate input
        # ------------------------------------------

        if image_bgr is None:
            raise ValueError(
                "image_bgr cannot be None."
            )

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
        # SAM 3 segmentation
        # ------------------------------------------

        segmentation_output = None

        if segmentation_prompt:
            # OpenCV uses BGR.
            # PIL / SAM 3 expects RGB.

            image_rgb = image_bgr[
                :,
                :,
                ::-1
            ]

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
        # Combined visualization
        # ------------------------------------------

        class_names = (
            self.detector.get_class_names()
        )

        annotated_image = (
            self.visualizer.annotate(
                image=image_bgr,
                detections=tracked_detections,
                class_names=class_names,
                segmentation_output=segmentation_output
            )
        )

        # ------------------------------------------
        # Final structured result
        # ------------------------------------------

        result = {
            "detections": detections,
            "tracked_detections": tracked_detections,
            "segmentation": segmentation_output,
            "annotated_image": annotated_image
        }

        return result

    def reset_tracker(self):
        """
        Reset ByteTrack before processing a new
        independent image or video sequence.
        """

        self.tracker.reset()
