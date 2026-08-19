# Object Tracking — Output Assets

This directory contains the generated output files produced by the practical exercises in the **Object Tracking** section of the SAM3 Computer Vision Learning Journey.

The files stored here are generated after processing the input video through the object-tracking pipeline.

---

## Purpose

The `output/` directory keeps generated tracking results separate from the original input assets.

The practical exercise reads the source video from:

```text
assets/input/tracking_demo.mp4
```

and generates the processed tracking result inside this directory.

---

## Final Output

The final browser-compatible output video is:

```text
tracked_demo_h264.mp4
```

The video was converted to **H.264** after the tracking process so it can be played correctly in modern browsers and environments such as Google Colab.

---

## Output Directory Structure

```text
output/
├── README.md
└── tracked_demo_h264.mp4
```

---

## Output Video

The generated `tracked_demo_h264.mp4` demonstrates the final result of the Object Tracking practical exercise.

The video contains:

- Bounding boxes
- Object labels
- Persistent tracking IDs
- `tracker_id` values
- Tracking annotations
- Object trajectories
- Frame information
- Basic tracking information

---

## Tracking Pipeline

The final video was generated through the following workflow:

```text
tracking_demo.mp4
        ↓
Read Video Frames
        ↓
Synthetic Object Detections
        ↓
sv.Detections
        ↓
Supervision
        ↓
ByteTrack
        ↓
tracker_id
        ↓
Tracking Annotations
        ↓
Object Trajectories
        ↓
tracked_demo.mp4
        ↓
H.264 Conversion
        ↓
tracked_demo_h264.mp4
```

---

## Why H.264 Was Used

The initial tracking script generated:

```text
tracked_demo.mp4
```

using OpenCV's MP4 video writer.

Although the tracking process completed successfully, the original MP4 encoding was not reliably playable inside the Google Colab browser video player.

The generated video was therefore converted using FFmpeg to:

```text
tracked_demo_h264.mp4
```

with:

```text
H.264
yuv420p
```

This produced a browser-compatible version of the final tracking result.

---

## Input and Output

The original video remains inside:

```text
assets/input/
```

The final processed tracking result is stored inside:

```text
assets/output/
```

The complete asset structure is:

```text
assets/
├── README.md
│
├── input/
│   ├── README.md
│   └── tracking_demo.mp4
│
└── output/
    ├── README.md
    └── tracked_demo_h264.mp4
```

---

## Practical Implementation

The tracking process is implemented in:

```text
practical/object_tracking_practical.py
```

The script:

1. Opens the input video.
2. Reads the video frame by frame.
3. Generates synthetic object detections.
4. Converts the detections into `sv.Detections`.
5. Passes detections through ByteTrack.
6. Assigns persistent `tracker_id` values.
7. Draws bounding boxes.
8. Adds object labels.
9. Visualizes object trajectories.
10. Writes the processed frames to an output video.

---

## Tracking IDs

ByteTrack assigns persistent identifiers to objects across consecutive frames.

The final video demonstrates three tracked objects:

```text
object_a #1
object_b #2
object_c #3
```

These IDs remain associated with the objects while they move through the scene.

---

## Object Trajectories

Supervision's tracking annotations visualize the recent movement history of each tracked object.

Conceptually:

```text
Previous Positions
        ↓
● → ● → ● → ●
            ↑
     Current Position
```

This makes object movement easier to analyze across consecutive video frames.

---

## Final Result

The practical successfully demonstrates the transition from independent object detections to persistent object tracking.

```text
Detection
    ↓
Object Association
    ↓
ByteTrack
    ↓
Persistent tracker_id
    ↓
Tracking Visualization
    ↓
Tracking Analytics
```

The final output video confirms that objects can maintain their identities while moving through the scene.

---

## Related Input

The source video used by this exercise is:

```text
../input/tracking_demo.mp4
```

---

## Related Course Material

These output assets support the **Object Tracking** practical exercises and complement the concept notes and class recording.

[Watch the Object Tracking class recording on YouTube](https://youtu.be/UXN0l33NqF4)

---

## Repository

[View the SAM3 Learning Journey on GitHub](https://github.com/Peyman-mxli/SAM3-Learning-Journey)

---

## Author

**Peyman Miyandashti**

- [GitHub](https://github.com/Peyman-mxli)
- [LinkedIn](https://www.linkedin.com/in/peyman-mxli/)
