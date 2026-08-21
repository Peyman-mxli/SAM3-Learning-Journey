"""
visualization.py

Visualization module for the Visual Tracking and Analysis System.

This module uses Supervision annotators to display bounding boxes,
labels, tracker IDs, object trajectories, and SAM 3 segmentation
masks on processed images and video frames.
"""

import cv2
import numpy as np
import supervision as sv


class TrackingVisualizer:
    """
    Visualization helper for detection, tracking,
    and segmentation results.
    """

    def __init__(self):
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.trace_annotator = sv.TraceAnnotator()

    def apply_segmentation_masks(
        self,
        image,
        segmentation_output,
        alpha=0.40
    ):
        """
        Overlay SAM 3 segmentation masks on an image.

        Parameters
        ----------
        image
            Image represented as a NumPy/OpenCV BGR array.

        segmentation_output : dict or None
            SAM 3 output containing segmentation masks.

        alpha : float
            Transparency of the segmentation overlay.

        Returns
        -------
        image
            Image with segmentation masks applied.
        """

        if segmentation_output is None:
            return image.copy()

        masks = segmentation_output.get(
            "masks"
        )

        if masks is None:
            return image.copy()

        annotated_image = image.copy()

        mask_colors = [
            (255, 0, 255),
            (0, 255, 255),
            (255, 255, 0),
            (0, 165, 255),
            (255, 0, 0),
            (0, 255, 0),
        ]

        for index, mask in enumerate(masks):
            mask_np = (
                mask
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            mask_np = np.squeeze(
                mask_np
            )

            binary_mask = (
                mask_np > 0.5
            )

            if not np.any(binary_mask):
                continue

            color = mask_colors[
                index % len(mask_colors)
            ]

            overlay = np.zeros_like(
                annotated_image,
                dtype=np.uint8
            )

            overlay[
                binary_mask
            ] = color

            blended = cv2.addWeighted(
                annotated_image,
                1.0,
                overlay,
                alpha,
                0
            )

            annotated_image[
                binary_mask
            ] = blended[
                binary_mask
            ]

        return annotated_image

    def annotate(
        self,
        image,
        detections,
        class_names,
        segmentation_output=None
    ):
        """
        Annotate an image with segmentation, detection,
        and tracking information.

        Parameters
        ----------
        image
            Image represented as a NumPy/OpenCV BGR array.

        detections : supervision.Detections
            Detection or tracking results.

        class_names : dict
            Mapping between class IDs and class names.

        segmentation_output : dict or None
            Optional SAM 3 segmentation output.

        Returns
        -------
        image
            Fully annotated image.
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

        # ------------------------------------------
        # Base image
        # ------------------------------------------

        annotated_image = image.copy()

        # ------------------------------------------
        # SAM 3 segmentation masks
        # ------------------------------------------

        annotated_image = (
            self.apply_segmentation_masks(
                image=annotated_image,
                segmentation_output=segmentation_output
            )
        )

        # ------------------------------------------
        # Bounding boxes
        # ------------------------------------------

        annotated_image = self.box_annotator.annotate(
            scene=annotated_image,
            detections=detections
        )

        # ------------------------------------------
        # Labels and tracker IDs
        # ------------------------------------------

        annotated_image = self.label_annotator.annotate(
            scene=annotated_image,
            detections=detections,
            labels=labels
        )

        # ------------------------------------------
        # Object trajectories
        # ------------------------------------------

        if detections.tracker_id is not None:
            annotated_image = self.trace_annotator.annotate(
                scene=annotated_image,
                detections=detections
            )

        return annotated_image
