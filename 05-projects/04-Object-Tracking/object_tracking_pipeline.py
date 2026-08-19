```python
"""
object_tracking_pipeline.py

YOLO + Supervision + ByteTrack Object Tracking Pipeline

This project combines the main concepts from the Object
Tracking lesson into one reusable computer vision application.

Pipeline:

    Input Video
        ↓
    YOLO Detection
        ↓
    sv.Detections
        ↓
    Class Filtering
        ↓
    ByteTrack
        ↓
    tracker_id
        ↓
    Tracking Analytics
        ↓
    Boxes + Labels + Traces
        ↓
    Output Video

The project tracks cars, counts how many frames each tracked
object remains visible, estimates visible time, and records
the number of unique tracker IDs.
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

INPUT_VIDEO = "assets/input/vehicles.mp4"

OUTPUT_VIDEO = "assets/output/tracked_vehicles.mp4"

# COCO class ID:
# 2 = car
TARGET_CLASS_ID = 2


# --------------------------------------------------
# Create required directories
# --------------------------------------------------

Path("assets/input").mkdir(
    parents=True,
    exist_ok=True
)

Path("assets/output").mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Download input video if needed
# --------------------------------------------------

if not Path(INPUT_VIDEO).exists():

    print(
        "Input video not found."
    )

    print(
        "Downloading sample vehicle video..."
    )

    urllib.request.urlretrieve(
        VIDEO_URL,
        INPUT_VIDEO
    )

    print(
        f"Video saved to: {INPUT_VIDEO}"
    )


# --------------------------------------------------
# Verify input video
# --------------------------------------------------

if not Path(INPUT_VIDEO).exists():

    raise FileNotFoundError(
        f"Could not find input video: "
        f"{INPUT_VIDEO}"
    )


# --------------------------------------------------
# Read video information
# --------------------------------------------------

video_info = sv.VideoInfo.from_video_path(
    INPUT_VIDEO
)

VIDEO_FPS = video_info.fps


if VIDEO_FPS <= 0:

    raise ValueError(
        "Invalid video FPS."
    )


video_duration = (
    video_info.total_frames /
    VIDEO_FPS
)


print(
    "\nVideo information"
)

print(
    "-" * 50
)

print(
    f"Input: {INPUT_VIDEO}"
)

print(
    f"Resolution: "
    f"{video_info.width} x "
    f"{video_info.height}"
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
    f"{video_duration:.2f} seconds"
)


# --------------------------------------------------
# Load YOLO model
# --------------------------------------------------

print(
    "\nLoading YOLO model..."
)

model = YOLO(
    MODEL_NAME
)

print(
    f"Model loaded: {MODEL_NAME}"
)


# --------------------------------------------------
# Create ByteTrack tracker
# --------------------------------------------------

tracker = sv.ByteTrack()


# --------------------------------------------------
# Reset tracker before processing
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
#
#     tracker_id -> number of frames visible
#
# Example:
#
#     {
#         1: 25,
#         2: 41,
#         3: 87
#     }
#
# unique_tracker_ids:
#
#     stores every unique tracker ID observed
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

    """
    Process one video frame.

    Steps:

    1. Detect objects with YOLO
    2. Convert detections to sv.Detections
    3. Filter detections by class
    4. Track filtered detections with ByteTrack
    5. Update tracking analytics
    6. Create labels
    7. Draw boxes, labels, and traces
    8. Display current tracking statistics

    Returns:
        np.ndarray:
            Annotated video frame.
    """

    # ----------------------------------------------
    # Run YOLO object detection
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
    # Filter detections BEFORE tracking
    #
    # Keep only the target class.
    #
    # TARGET_CLASS_ID = 2
    # COCO class 2 = car
    # ----------------------------------------------

    detections = detections[
        detections.class_id
        == TARGET_CLASS_ID
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

            # --------------------------------------
            # Count visible frames
            # --------------------------------------

            frame_count[tracker_id] = (
                frame_count.get(
                    tracker_id,
                    0
                ) + 1
            )


            # --------------------------------------
            # Store unique tracker ID
            # --------------------------------------

            unique_tracker_ids.add(
                tracker_id
            )


    # ----------------------------------------------
    # Create object labels
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


            # --------------------------------------
            # Get detected class name
            # --------------------------------------

            class_name = results.names[
                class_id
            ]


            # --------------------------------------
            # Get visible frame count
            # --------------------------------------

            frames_visible = frame_count[
                tracker_id
            ]


            # --------------------------------------
            # Convert frames into seconds
            # --------------------------------------

            visible_seconds = (
                frames_visible /
                VIDEO_FPS
            )


            # --------------------------------------
            # Create label
            # --------------------------------------

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
    # Draw tracking trajectories
    # ----------------------------------------------

    annotated_frame = trace_annotator.annotate(
        scene=annotated_frame,
        detections=detections
    )


    # ----------------------------------------------
    # Calculate current unique tracker count
    # ----------------------------------------------

    unique_count = len(
        unique_tracker_ids
    )


    # ----------------------------------------------
    # Display unique object count
    # ----------------------------------------------

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
    # Display current frame index
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
    # Return annotated frame
    # ----------------------------------------------

    return annotated_frame


# --------------------------------------------------
# Process complete video
# --------------------------------------------------

print(
    "\nStarting object tracking..."
)

print(
    f"Tracking COCO class ID: "
    f"{TARGET_CLASS_ID}"
)

print(
    f"Output: {OUTPUT_VIDEO}\n"
)


sv.process_video(
    source_path=INPUT_VIDEO,
    target_path=OUTPUT_VIDEO,
    callback=process_frame,
    show_progress=True
)


# --------------------------------------------------
# Print final tracking analytics
# --------------------------------------------------

print(
    "\nTracking analytics"
)

print(
    "-" * 60
)


if frame_count:

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

else:

    print(
        "No tracked objects were found."
    )


# --------------------------------------------------
# Print unique tracker IDs
# --------------------------------------------------

print(
    "\nUnique tracker IDs:"
)

print(
    sorted(
        unique_tracker_ids
    )
)


# --------------------------------------------------
# Print total unique tracker count
# --------------------------------------------------

print(
    "\nTotal unique tracked cars:"
)

print(
    len(
        unique_tracker_ids
    )
)


# --------------------------------------------------
# Verify output
# --------------------------------------------------

if Path(OUTPUT_VIDEO).exists():

    output_size_mb = (
        Path(OUTPUT_VIDEO).stat().st_size
        /
        (1024 * 1024)
    )

    print(
        "\nOutput video created successfully."
    )

    print(
        f"File: {OUTPUT_VIDEO}"
    )

    print(
        f"Size: "
        f"{output_size_mb:.2f} MB"
    )

else:

    raise RuntimeError(
        f"Output video was not created: "
        f"{OUTPUT_VIDEO}"
    )


# --------------------------------------------------
# Pipeline summary
# --------------------------------------------------

print(
    "\nPipeline completed:"
)

print(
    "YOLO"
)

print(
    "  -> sv.Detections"
)

print(
    "  -> Class Filtering"
)

print(
    "  -> ByteTrack"
)

print(
    "  -> tracker_id"
)

print(
    "  -> Frame Counting"
)

print(
    "  -> Visible Time"
)

print(
    "  -> Unique Tracker IDs"
)

print(
    "  -> Box Annotation"
)

print(
    "  -> Label Annotation"
)

print(
    "  -> Trace Annotation"
)

print(
    "  -> Output Video"
)


# --------------------------------------------------
# Important note
# --------------------------------------------------

print(
    "\nImportant:"
)

print(
    "Tracker IDs represent identities maintained "
    "by ByteTrack during this tracking sequence."
)

print(
    "A tracker ID should not automatically be "
    "treated as a permanent real-world identity."
)


# --------------------------------------------------
# Completion
# --------------------------------------------------

print(
    "\nObject Tracking Project: SUCCESS"
)
```
