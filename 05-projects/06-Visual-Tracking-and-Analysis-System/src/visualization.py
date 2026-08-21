"""
visualization.py

Visualization module for the Visual Tracking and Analysis System.

This module uses Supervision annotators to display bounding boxes,
labels, tracker IDs, and object trajectories on processed frames.
"""

import supervision as sv


class TrackingVisualizer:
    """
    Visualization helper for detection and tracking results.
    """

    def __init__(self):
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.trace_annotator = sv.TraceAnnotator()

    def annotate(
        self,
        image,
        detections,
        class_names
    ):
        """
        Annotate an image with detection and tracking information.

        Parameters
        ----------
        image
            Image represented as a NumPy/OpenCV array.

        detections : supervision.Detections
            Detection or tracking results.

        class_names : dict
            Mapping between class IDs and class names.

        Returns
        -------
        image
            Annotated image.
        """

        labels = []

        for index in range(len(detections)):
            class_id = int(
                detections.class_id[index]
            )

            confidence = float(
                detections.confidence[index]
            )

            class_name = class_names.get(
                class_id,
                str(class_id)
            )

            tracker_id = None

            if detections.tracker_id is not None:
                tracker_id = int(
                    detections.tracker_id[index]
                )

            if tracker_id is not None:
                label = (
                    f"#{tracker_id} "
                    f"{class_name} "
                    f"{confidence:.2f}"
                )
            else:
                label = (
                    f"{class_name} "
                    f"{confidence:.2f}"
                )

            labels.append(label)

        annotated_image = image.copy()

        annotated_image = self.box_annotator.annotate(
            scene=annotated_image,
            detections=detections
        )

        annotated_image = self.label_annotator.annotate(
            scene=annotated_image,
            detections=detections,
            labels=labels
        )

        if detections.tracker_id is not None:
            annotated_image = self.trace_annotator.annotate(
                scene=annotated_image,
                detections=detections
            )

        return annotated_image
