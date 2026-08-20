```python
"""
09-complete-tracking-pipeline.py

Complete Object Tracking Pipeline

This example combines the main concepts from the
Object Tracking lesson into one complete workflow.

The pipeline includes:

1. Load a YOLO model
2. Load a sample video
3. Read video information
4. Detect objects
5. Convert results to sv.Detections
6. Filter detections before tracking
7. Track objects with ByteTrack
8. Assign persistent tracker IDs
9. Count frames visible per tracker ID
10. Estimate visible time
11. Count unique tracker IDs
12. Display class + tracker ID
13. Draw bounding boxes
14. Draw object trajectories
15. Save the final annotated video

This example follows the concepts introduced in the
Object Tracking course notebook.
"""

from pathlib import Path
import urllib.request

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "yolov8n.pt"

VIDEO_URL = (
    "https://media.roboflow.com/"
    "supervision/video-examples/vehicles.mp4"
)

SOURCE_VIDEO = "assets/vehicles.mp4"

OUTPUT_VIDEO = "assets/vehicles_complete_tracking.mp4"

# COCO class ID:
# 2 = car
TARGET_CLASS = 2


# --------------------------------------------------
# Create assets directory
# --------------------------------------------------

Path("assets").mkdir(exist_ok=True)


# --------------------------------------------------
# Download sample video if needed
# --------------------------------------------------

if not Path(SOURCE_VIDEO).exists():

    print("Downloading sample video...")

    urllib.request.urlretrieve(
        VIDEO_URL,
        SOURCE_VIDEO
    )

    print(
        f"Video saved to: {SOURCE_VIDEO}"
    )


# --------------------------------------------------
# Inspect video information
# --------------------------------------------------

video_info = sv.VideoInfo.from_video_path(
    SOURCE_VIDEO
)

VIDEO_FPS = video_info.fps


print("\nVideo information:")

print(
    f"Resolution: "
    f"{video_info.width} x {video_info.height}"
)

print(
    f"FPS: {VIDEO_FPS}"
)

print(
    f"Total frames: "
    f"{video_info.total_frames}"
)

print(
    f"Duration: "
    f"{video_info.total_frames / VIDEO_FPS:.2f} seconds"
)


# --------------------------------------------------
# Load YOLO model
# --------------------------------------------------

model = YOLO(MODEL_NAME)


# --------------------------------------------------
# Create ByteTrack tracker
# --------------------------------------------------

tracker = sv.ByteTrack()


# --------------------------------------------------
# Reset tracker
# --------------------------------------------------

tracker.reset()


# --------------------------------------------------
# Create annotators
# --------------------------------------------------

box_annotator = sv.BoxAnnotator()

label_annotator = sv.LabelAnnotator()

trace_annotator = sv.TraceAnnotator()


# --------------------------------------------------
# Tracking analytics
#
# frame_count:
# tracker_id -> number of frames visible
#
# unique_tracker_ids:
# stores every tracker ID observed
# --------------------------------------------------

frame_count = {}

unique_tracker_ids = set()


# --------------------------------------------------
# Frame processing callback
# --------------------------------------------------

def process_frame(
    frame: np.ndarray,
    frame_idx: int
) -> np.ndarray:

    # ----------------------------------------------
    # Run YOLO detection
    # ----------------------------------------------

    results = model(
        frame,
        verbose=False
    )[0]


    # ----------------------------------------------
    # Convert YOLO results to Supervision
    # ----------------------------------------------

    detections = sv.Detections.from_ultralytics(
        results
    )


    # ----------------------------------------------
    # Filter BEFORE tracking
    #
    # Track only cars.
    # ----------------------------------------------

    detections = detections[
        detections.class_id == TARGET_CLASS
    ]


    # ----------------------------------------------
    # Apply ByteTrack
    # ----------------------------------------------

    detections = tracker.update_with_detections(
        detections
    )


    # ----------------------------------------------
    # Update tracking analytics
    # ----------------------------------------------

    if detections.tracker_id is not None:

        for tracker_id in detections.tracker_id:

            tracker_id = int(
                tracker_id
            )

            # Count frames visible
            frame_count[tracker_id] = (
                frame_count.get(
                    tracker_id,
                    0
                ) + 1
            )

            # Store unique ID
            unique_tracker_ids.add(
                tracker_id
            )


    # ----------------------------------------------
    # Create tracking labels
    #
    # Example:
    #
    # car #5 | 42f | 1.40s
    # ----------------------------------------------

    labels = []

    if detections.tracker_id is not None:

        for class_id, tracker_id in zip(
            detections.class_id,
            detections.tracker_id
        ):

            class_id = int(
                class_id
            )

            tracker_id = int(
                tracker_id
            )

            class_name = results.names[
                class_id
            ]

            frames_visible = frame_count[
                tracker_id
            ]

            visible_seconds = (
                frames_visible /
                VIDEO_FPS
            )

            label = (
                f"{class_name} "
                f"#{tracker_id} | "
                f"{frames_visible}f | "
                f"{visible_seconds:.2f}s"
            )

            labels.append(
                label
            )


    # ----------------------------------------------
    # Draw bounding boxes
    # ----------------------------------------------

    annotated_frame = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )


    # ----------------------------------------------
    # Draw labels
    # ----------------------------------------------

    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )


    # ----------------------------------------------
    # Draw object trajectories
    # ----------------------------------------------

    annotated_frame = trace_annotator.annotate(
        scene=annotated_frame,
        detections=detections
    )


    # ----------------------------------------------
    # Display current unique tracker count
    # ----------------------------------------------

    unique_count = len(
        unique_tracker_ids
    )

    cv2.putText(
        annotated_frame,
        f"Unique tracked cars: {unique_count}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    # ----------------------------------------------
    # Display current frame
    # ----------------------------------------------

    cv2.putText(
        annotated_frame,
        f"Frame: {frame_idx}",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    # ----------------------------------------------
    # Return processed frame
    # ----------------------------------------------

    return annotated_frame


# --------------------------------------------------
# Process complete video
# --------------------------------------------------

print(
    "\nStarting complete tracking pipeline..."
)

sv.process_video(
    source_path=SOURCE_VIDEO,
    target_path=OUTPUT_VIDEO,
    callback=process_frame,
    show_progress=True
)


# --------------------------------------------------
# Final analytics
# --------------------------------------------------

print(
    "\nTracking analytics"
)

print(
    "-" * 60
)


for tracker_id, frames_visible in sorted(
    frame_count.items()
):

    visible_seconds = (
        frames_visible /
        VIDEO_FPS
    )

    print(
        f"Tracker ID {tracker_id}: "
        f"{frames_visible} frames | "
        f"{visible_seconds:.2f} seconds"
    )


# --------------------------------------------------
# Unique tracker IDs
# --------------------------------------------------

print(
    "\nUnique tracker IDs:"
)

print(
    sorted(unique_tracker_ids)
)


print(
    "\nTotal unique tracked cars:"
)

print(
    len(unique_tracker_ids)
)


# --------------------------------------------------
# Final pipeline summary
# --------------------------------------------------

print(
    "\nComplete pipeline:"
)

print(
    "Video"
)

print(
    "  -> YOLO"
)

print(
    "  -> sv.Detections"
)

print(
    "  -> Class Filter"
)

print(
    "  -> ByteTrack"
)

print(
    "  -> tracker_id"
)

print(
    "  -> Frame Count"
)

print(
    "  -> Visible Time"
)

print(
    "  -> Unique ID Count"
)

print(
    "  -> BoxAnnotator"
)

print(
    "  -> LabelAnnotator"
)

print(
    "  -> TraceAnnotator"
)

print(
    "  -> Output Video"
)


# --------------------------------------------------
# Output
# --------------------------------------------------

print(
    f"\nFinal video saved to: "
    f"{OUTPUT_VIDEO}"
)


# --------------------------------------------------
# Important tracking note
# --------------------------------------------------

print(
    "\nImportant:"
)

print(
    "Tracker IDs represent identities inside "
    "the current tracking sequence."
)

print(
    "If ByteTrack loses an object and later "
    "assigns it another ID, the same physical "
    "object may appear as multiple tracker IDs."
)


# --------------------------------------------------
# Completion
# --------------------------------------------------

print(
    "\nComplete object tracking pipeline "
    "finished successfully."
)
```
