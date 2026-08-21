"""
tracker.py

Object tracking module for the Visual Tracking and Analysis System.

This module uses Supervision ByteTrack to assign persistent tracker
IDs to detections across consecutive video frames.
"""

import supervision as sv


class ObjectTracker:
    """
    Multi-object tracker based on Supervision ByteTrack.
    """

    def __init__(self):
        self.tracker = sv.ByteTrack()

    def update(self, detections):
        """
        Update the tracker with detections from the current frame.

        Parameters
        ----------
        detections : supervision.Detections
            Detections generated for the current video frame.

        Returns
        -------
        supervision.Detections
            Detections containing persistent tracker IDs.
        """

        tracked_detections = self.tracker.update_with_detections(
            detections
        )

        return tracked_detections

    def reset(self):
        """
        Reset the tracker for a new processing session.
        """

        self.tracker.reset()
