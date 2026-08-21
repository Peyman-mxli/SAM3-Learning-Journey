"""
detector.py

Object detection module for the Visual Tracking and Analysis System.

This module uses Ultralytics YOLO for object detection and converts
the model predictions into Supervision Detections objects so they can
be used by the rest of the processing pipeline.
"""

from ultralytics import YOLO
import supervision as sv


class ObjectDetector:
    """
    YOLO-based object detector.

    Parameters
    ----------
    model_name : str
        Ultralytics YOLO model to load.

    confidence_threshold : float
        Minimum confidence required to keep a detection.
    """

    def __init__(
        self,
        model_name="yolov8n.pt",
        confidence_threshold=0.50
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold

        self.model = YOLO(self.model_name)

    def detect(self, image):
        """
        Run object detection on an image.

        Parameters
        ----------
        image
            Image represented as a NumPy/OpenCV array.

        Returns
        -------
        supervision.Detections
            Filtered detections.
        """

        result = self.model(
            image,
            conf=self.confidence_threshold,
            verbose=False
        )[0]

        detections = sv.Detections.from_ultralytics(result)

        return detections

    def get_class_names(self):
        """
        Return the class-name mapping used by the YOLO model.
        """

        return self.model.names
