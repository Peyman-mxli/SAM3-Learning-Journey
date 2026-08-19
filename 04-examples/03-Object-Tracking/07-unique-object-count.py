```python
"""
07-unique-object-count.py

Count Unique Tracked Objects

This example demonstrates how tracker IDs can be used
to count unique objects across an entire video.

Instead of counting every detection in every frame,
we store each tracker_id inside a Python set.

Because a set stores unique values, the final size
of the set represents the number of unique tracker
IDs observed during the tracking sequence.

The example:

1. Detects objects with YOLO
2. Converts results to sv.Detections
3. Tracks objects with ByteTrack
4. Stores tracker IDs in a set
5. Displays class and tracker ID labels
6. Displays the current unique object count
7. Saves the annotated output video
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

OUTPUT_VIDEO = "assets/vehicles_unique_count.mp4"


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
# Store unique tracker IDs
#
# A Python set automatically keeps only unique values.
#
# Example:
#
# {1, 2, 3, 4}
#
# If ID 2 appears again, it is not added twice.
# --------------------------------------------------

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
    # Apply ByteTrack
    # ----------------------------------------------

    detections = tracker.update_with_detections(
        detections
    )


    # ----------------------------------------------
    # Add tracker IDs to the unique ID set
    # ----------------------------------------------

    if detections.tracker_id is not None:

        for tracker_id in detections.tracker_id:

            unique_tracker_ids.add(
                int(tracker_id)
            )


    # ----------------------------------------------
    # Create labels
    #
    # Example:
    #
    # car #1
    # truck #2
    # bus #3
    # ----------------------------------------------

    labels = []

    if detections.tracker_id is not None:

        for class_id, tracker_id in zip(
            detections.class_id,
            detections.tracker_id
        ):

            class_id = int(class_id)

            tracker_id = int(tracker_id)

            class_name = results.names[
                class_id
            ]

            labels.append(
                f"{class_name} "
                f"#{tracker_id}"
            )


    # ----------------------------------------------
    # Draw bounding boxes
    # ----------------------------------------------

    annotated_frame = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )


    # ----------------------------------------------
    # Draw object labels
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
    # Current unique object count
    # ----------------------------------------------

    unique_count = len(
        unique_tracker_ids
    )


    # ----------------------------------------------
    # Display unique count on frame
    # ----------------------------------------------

    cv2.putText(
        annotated_frame,
        f"Unique tracked objects: {unique_count}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
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
    "\nUnique tracker IDs:"
)

print(
    sorted(unique_tracker_ids)
)


print(
    "\nTotal unique tracker IDs:"
)

print(
    len(unique_tracker_ids)
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print(
    f"\nOutput video saved to: "
    f"{OUTPUT_VIDEO}"
)

print(
    "The unique object count was calculated "
    "using tracker IDs generated by ByteTrack."
)

print(
    "\nImportant:"
)

print(
    "A tracker ID is a tracking identity, "
    "not a permanent real-world identity."
)

print(
    "If tracking is lost and the same physical "
    "object receives a new tracker ID, it may be "
    "counted as another unique tracked object."
)

print(
    "\nUnique object counting completed successfully."
)
```
