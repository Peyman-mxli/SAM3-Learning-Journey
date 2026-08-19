"""
object_tracking_practical.py

Object Tracking Practical Exercise

This practical demonstrates the basic object-tracking workflow using:

- OpenCV
- Supervision
- ByteTrack
- Synthetic detections
- Persistent tracker IDs
- Tracking annotations
- Object trajectories

The input video is synthetic, so detections are generated manually instead
of relying on YOLO classification.

Input:
    assets/input/tracking_demo.mp4

Output:
    assets/output/tracked_demo.mp4
"""

from pathlib import Path

import cv2
import numpy as np
import supervision as sv


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_VIDEO = "assets/input/tracking_demo.mp4"
OUTPUT_VIDEO = "assets/output/tracked_demo.mp4"

FRAME_WIDTH = 960
FRAME_HEIGHT = 540

FPS = 30


# --------------------------------------------------
# Create required directories
# --------------------------------------------------

Path("assets/output").mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Initialize ByteTrack
# --------------------------------------------------

tracker = sv.ByteTrack(
    frame_rate=FPS
)


# --------------------------------------------------
# Create annotators
# --------------------------------------------------

box_annotator = sv.BoxAnnotator()

label_annotator = sv.LabelAnnotator()

trace_annotator = sv.TraceAnnotator(
    trace_length=30
)


# --------------------------------------------------
# Open input video
# --------------------------------------------------

cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    raise FileNotFoundError(
        f"Could not open video: {INPUT_VIDEO}"
    )


# --------------------------------------------------
# Read video properties
# --------------------------------------------------

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)


print("Video Information")
print("-----------------")
print(f"Width: {width}")
print(f"Height: {height}")
print(f"FPS: {fps}")
print(f"Frames: {total_frames}")


# --------------------------------------------------
# Create output video writer
# --------------------------------------------------

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)


# --------------------------------------------------
# Synthetic detection generator
# --------------------------------------------------

def create_synthetic_detections(frame_index, total_frames):
    """
    Generate bounding boxes that follow the same
    approximate motion paths as the objects inside
    tracking_demo.mp4.
    """

    progress = frame_index / max(
        total_frames - 1,
        1
    )

    boxes = []

    class_ids = []

    confidences = []


    # --------------------------------------------------
    # Object 1
    # Person-like object
    # --------------------------------------------------

    x = int(
        80 + 760 * progress
    )

    y = int(
        245
        + 20
        * np.sin(
            2
            * np.pi
            * progress
        )
    )

    boxes.append(
        [
            x - 35,
            y - 70,
            x + 35,
            y + 80
        ]
    )

    class_ids.append(0)

    confidences.append(0.95)


    # --------------------------------------------------
    # Object 2
    # Car-like object
    # --------------------------------------------------

    x = int(
        850 - 720 * progress
    )

    y = 385

    boxes.append(
        [
            x - 60,
            y - 55,
            x + 60,
            y + 40
        ]
    )

    class_ids.append(1)

    confidences.append(0.96)


    # --------------------------------------------------
    # Object 3
    # Appears later in the video
    # --------------------------------------------------

    if frame_index > 70:

        local_progress = (
            frame_index - 70
        ) / max(
            total_frames - 70,
            1
        )

        x = int(
            120
            + 650
            * local_progress
        )

        y = int(
            110
            + 45
            * np.sin(
                4
                * np.pi
                * local_progress
            )
        )

        boxes.append(
            [
                x - 28,
                y - 28,
                x + 28,
                y + 28
            ]
        )

        class_ids.append(2)

        confidences.append(0.93)


    detections = sv.Detections(
        xyxy=np.array(
            boxes,
            dtype=np.float32
        ),
        confidence=np.array(
            confidences,
            dtype=np.float32
        ),
        class_id=np.array(
            class_ids,
            dtype=int
        )
    )

    return detections


# --------------------------------------------------
# Class names
# --------------------------------------------------

CLASS_NAMES = {
    0: "object_a",
    1: "object_b",
    2: "object_c"
}


# --------------------------------------------------
# Process video
# --------------------------------------------------

frame_index = 0

while True:

    success, frame = cap.read()

    if not success:
        break


    # --------------------------------------------------
    # Create detections
    # --------------------------------------------------

    detections = create_synthetic_detections(
        frame_index,
        total_frames
    )


    # --------------------------------------------------
    # Update ByteTrack
    # --------------------------------------------------

    tracked_detections = tracker.update_with_detections(
        detections
    )


    # --------------------------------------------------
    # Create tracking labels
    # --------------------------------------------------

    labels = []

    for class_id, tracker_id in zip(
        tracked_detections.class_id,
        tracked_detections.tracker_id
    ):

        class_name = CLASS_NAMES.get(
            int(class_id),
            "object"
        )

        labels.append(
            f"{class_name} #{tracker_id}"
        )


    # --------------------------------------------------
    # Annotate frame
    # --------------------------------------------------

    annotated_frame = frame.copy()

    annotated_frame = trace_annotator.annotate(
        scene=annotated_frame,
        detections=tracked_detections
    )

    annotated_frame = box_annotator.annotate(
        scene=annotated_frame,
        detections=tracked_detections
    )

    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=tracked_detections,
        labels=labels
    )


    # --------------------------------------------------
    # Add frame information
    # --------------------------------------------------

    cv2.putText(
        annotated_frame,
        f"Frame: {frame_index + 1}/{total_frames}",
        (25, 500),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (50, 50, 50),
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        annotated_frame,
        f"Tracked Objects: {len(tracked_detections)}",
        (25, 525),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (50, 50, 50),
        2,
        cv2.LINE_AA
    )


    # --------------------------------------------------
    # Write frame
    # --------------------------------------------------

    writer.write(
        annotated_frame
    )


    frame_index += 1


# --------------------------------------------------
# Release resources
# --------------------------------------------------

cap.release()

writer.release()


# --------------------------------------------------
# Final message
# --------------------------------------------------

print()
print("Tracking completed successfully.")
print(
    f"Output saved to: {OUTPUT_VIDEO}"
)
