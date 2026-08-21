"""
analytics.py

Video Analytics Module
Visual Tracking and Analysis System

This module converts frame-by-frame object tracking results into
structured observations that can later be stored in SQLite and
used for statistical analysis.
"""

from collections import defaultdict
from typing import Dict, List, Optional

import supervision as sv


class VideoAnalytics:
    """
    Collects and summarizes temporal object-tracking observations.
    """

    def __init__(self, fps: float):
        if fps <= 0:
            raise ValueError("FPS must be greater than zero.")

        self.fps = float(fps)

        self.observations: List[Dict] = []

        self.tracker_frames = defaultdict(list)

        self.class_counts = defaultdict(int)

    # --------------------------------------------------
    # Add frame observations
    # --------------------------------------------------

    def add_frame(
        self,
        frame_number: int,
        detections: sv.Detections,
        class_names: Dict[int, str],
    ) -> None:
        """
        Record tracked detections from one video frame.
        """

        if detections is None:
            return

        if len(detections) == 0:
            return

        if detections.tracker_id is None:
            return

        timestamp_seconds = (
            frame_number / self.fps
        )

        for index in range(len(detections)):

            tracker_id = int(
                detections.tracker_id[index]
            )

            class_id: Optional[int] = None
            confidence: Optional[float] = None

            if detections.class_id is not None:
                class_id = int(
                    detections.class_id[index]
                )

            if detections.confidence is not None:
                confidence = float(
                    detections.confidence[index]
                )

            if class_id is not None:
                class_name = class_names.get(
                    class_id,
                    str(class_id),
                )
            else:
                class_name = "unknown"

            x1, y1, x2, y2 = (
                detections.xyxy[index]
            )

            observation = {
                "frame_number": int(
                    frame_number
                ),
                "timestamp_seconds": float(
                    timestamp_seconds
                ),
                "tracker_id": tracker_id,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
            }

            self.observations.append(
                observation
            )

            self.tracker_frames[
                tracker_id
            ].append(
                frame_number
            )

            self.class_counts[
                class_name
            ] += 1

    # --------------------------------------------------
    # Observation count
    # --------------------------------------------------

    def total_observations(self) -> int:
        """
        Return the number of stored object observations.
        """

        return len(
            self.observations
        )

    # --------------------------------------------------
    # Unique tracker IDs
    # --------------------------------------------------

    def unique_tracker_ids(self) -> List[int]:
        """
        Return all observed tracker IDs.
        """

        return sorted(
            self.tracker_frames.keys()
        )

    # --------------------------------------------------
    # Unique object count
    # --------------------------------------------------

    def unique_object_count(self) -> int:
        """
        Return the number of unique tracker identities.
        """

        return len(
            self.tracker_frames
        )

    # --------------------------------------------------
    # Tracker duration
    # --------------------------------------------------

    def tracker_duration(
        self,
        tracker_id: int,
    ) -> float:
        """
        Estimate how long a tracker ID was observed.

        Duration is calculated from the number of frames
        in which the tracker ID was present.
        """

        frames = self.tracker_frames.get(
            tracker_id,
            [],
        )

        return len(frames) / self.fps

    # --------------------------------------------------
    # Tracker summary
    # --------------------------------------------------

    def tracker_summary(self) -> List[Dict]:
        """
        Generate one summary record for each tracker ID.
        """

        summary = []

        for tracker_id in self.unique_tracker_ids():

            frames = self.tracker_frames[
                tracker_id
            ]

            first_frame = min(frames)
            last_frame = max(frames)

            summary.append(
                {
                    "tracker_id": tracker_id,
                    "first_frame": first_frame,
                    "last_frame": last_frame,
                    "frames_observed": len(
                        frames
                    ),
                    "duration_seconds": (
                        len(frames) / self.fps
                    ),
                }
            )

        return summary

    # --------------------------------------------------
    # Class summary
    # --------------------------------------------------

    def class_summary(self) -> Dict[str, int]:
        """
        Return the number of observations for each class.
        """

        return dict(
            self.class_counts
        )

    # --------------------------------------------------
    # Full summary
    # --------------------------------------------------

    def summary(self) -> Dict:
        """
        Return a high-level analytics summary.
        """

        return {
            "fps": self.fps,
            "total_observations":
                self.total_observations(),
            "unique_tracker_ids":
                self.unique_tracker_ids(),
            "unique_object_count":
                self.unique_object_count(),
            "class_observations":
                self.class_summary(),
            "trackers":
                self.tracker_summary(),
        }

    # --------------------------------------------------
    # Reset
    # --------------------------------------------------

    def reset(self) -> None:
        """
        Clear all collected analytics.
        """

        self.observations.clear()
        self.tracker_frames.clear()
        self.class_counts.clear()
