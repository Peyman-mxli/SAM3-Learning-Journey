# Object Tracking

This section of the **SAM3 Computer Vision Learning Journey** explores how computer vision systems move beyond detecting objects in individual images and begin maintaining object identities across consecutive video frames.

Object tracking connects object detection, video processing, detection association, persistent tracking IDs, trajectories, and tracking analytics into a complete video-based computer vision workflow.

The practical implementation in this section was successfully tested in **Google Colab** using Supervision and ByteTrack.

---

## Learning Objectives

By completing this section, we learn how to:

- Understand the difference between object detection and object tracking
- Process video frame by frame
- Represent detections using `sv.Detections`
- Associate detections between consecutive frames
- Use ByteTrack for multi-object tracking
- Understand persistent `tracker_id` values
- Create tracking labels
- Draw bounding boxes around tracked objects
- Visualize object trajectories
- Maintain object identities while objects move
- Export processed tracking videos
- Convert generated videos to browser-compatible H.264
- Prepare tracking pipelines for future YOLO integration

---

## From Object Detection to Object Tracking

Object detection answers:

```text
What objects are visible in this frame?
```

For example:

```text
Frame 1
├── person
├── person
└── car
```

When the next frame arrives, object detection runs again:

```text
Frame 2
├── person
├── person
└── car
```

Detection alone does not necessarily tell us whether a person detected in Frame 2 is the same person detected in Frame 1.

Object tracking adds this relationship.

```text
Frame 1 → person #1
Frame 2 → person #1
Frame 3 → person #1
Frame 4 → person #1
```

The object changes position, but the tracking system attempts to maintain its identity.

---

## Core Tracking Pipeline

The main conceptual workflow is:

```text
Input Video
    ↓
Video Frames
    ↓
Object Detection
    ↓
sv.Detections
    ↓
Detection Filtering
    ↓
ByteTrack
    ↓
tracker_id
    ↓
Tracking Annotations
    ↓
Object Trajectories
    ↓
Tracking Analytics
    ↓
Processed Video
```

---

## ByteTrack

ByteTrack is used to associate detections between consecutive video frames.

Instead of treating each detection as completely independent, the tracker compares detections over time and attempts to determine which detections belong to the same physical object.

Conceptually:

```text
Detection
    ↓
ByteTrack
    ↓
Object Association
    ↓
Persistent tracker_id
```

A tracked object may therefore appear as:

```text
person #1
```

while another object of the same class may appear as:

```text
person #2
```

The class is the same, but the tracking identity is different.

---

## `tracker_id`

After detections are processed by ByteTrack, Supervision can associate a tracking ID with each tracked detection.

Conceptually:

```python
detections.tracker_id
```

Example:

```text
[1, 2, 3]
```

These IDs make it possible to distinguish individual objects across consecutive frames.

---

## Tracking Labels

Tracking labels can combine the detected object class with its tracking ID.

Example:

```text
person #1
person #2
car #3
```

This makes the tracking result much easier to interpret visually.

---

## Object Trajectories

Tracking systems can also store recent positions of tracked objects.

These positions can be visualized as trajectories.

Conceptually:

```text
Previous Positions
        ↓
● → ● → ● → ●
            ↑
     Current Position
```

Trajectories make it possible to understand:

- Movement direction
- Movement history
- Object paths
- Object behavior
- Interaction between tracked objects

---

## Tracking Analytics

Once objects have persistent identities, additional analytics become possible.

Examples include:

- Counting unique objects
- Measuring movement
- Following individual objects
- Detecting line crossings
- Measuring time spent in an area
- Analyzing trajectories
- Monitoring entrances and exits
- Measuring object flow

The transition is:

```text
Detection
    ↓
Tracking
    ↓
Persistent Identity
    ↓
Movement History
    ↓
Tracking Analytics
```

---

# Practical Exercise

The practical exercise for this section is located in:

```text
practical/
```

It demonstrates the core tracking workflow using:

- Python
- OpenCV
- NumPy
- Supervision
- ByteTrack
- Synthetic detections
- Tracking annotations
- Object trajectories
- FFmpeg
- Google Colab

[Open the Object Tracking practical documentation](./practical/README.md)

---

## Practical Input Video

The practical uses a custom 10-second demonstration video:

```text
practical/assets/input/tracking_demo.mp4
```

Video properties:

```text
Duration: 10 seconds
Resolution: 960 × 540
Frame Rate: 30 FPS
Total Frames: 300
Format: MP4
```

The video contains several moving synthetic objects created specifically for learning object-tracking concepts.

[View the input assets documentation](./practical/assets/input/README.md)

[Open the tracking demo video](./practical/assets/input/tracking_demo.mp4)

---

## Why Synthetic Detections Are Used

The demonstration video contains synthetic objects rather than standard real-world YOLO classes.

Because of this, the first practical focuses directly on the tracking stage instead of depending on potentially unreliable YOLO classifications.

The practical pipeline is therefore:

```text
Synthetic Video
      ↓
Known Object Positions
      ↓
Synthetic Detections
      ↓
sv.Detections
      ↓
ByteTrack
      ↓
tracker_id
      ↓
Tracking Visualization
```

This isolates the tracking problem and makes ByteTrack behavior easier to understand.

---

## Practical Implementation

The main Python implementation is:

[`object_tracking_practical.py`](./practical/object_tracking_practical.py)

The script:

1. Opens the input video.
2. Reads the video properties.
3. Initializes ByteTrack.
4. Creates Supervision annotators.
5. Reads the video frame by frame.
6. Generates synthetic detections.
7. Converts them into `sv.Detections`.
8. Sends detections to ByteTrack.
9. Retrieves persistent tracking IDs.
10. Creates tracking labels.
11. Draws bounding boxes.
12. Draws movement traces.
13. Adds frame information.
14. Writes processed frames to an output video.

---

## Practical Tracking Pipeline

The completed practical follows:

```text
tracking_demo.mp4
        ↓
Read Video Frames
        ↓
Synthetic Object Detections
        ↓
sv.Detections
        ↓
ByteTrack
        ↓
tracker_id
        ↓
Bounding Boxes
        ↓
Tracking Labels
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

## Practical Tracking Result

The practical successfully tracks three moving objects.

The final labels include:

```text
object_a #1
object_b #2
object_c #3
```

Each object maintains its own tracking identity while moving through the scene.

---

## Final Output Video

The final browser-compatible tracking result is:

```text
practical/assets/output/tracked_demo_h264.mp4
```

[View the output assets documentation](./practical/assets/output/README.md)

[Open the final tracked video](./practical/assets/output/tracked_demo_h264.mp4)

The final video demonstrates:

- Bounding boxes
- Persistent tracking IDs
- Tracking labels
- Object trajectories
- Frame information
- Multi-object tracking across 300 frames

---

## H.264 Conversion

The initial OpenCV output was generated as:

```text
tracked_demo.mp4
```

The video was successfully processed, but the original `mp4v` encoding was not reliably playable in the Google Colab browser player.

The final video was therefore converted with FFmpeg:

```bash
ffmpeg -y \
-i assets/output/tracked_demo.mp4 \
-c:v libx264 \
-pix_fmt yuv420p \
-movflags +faststart \
assets/output/tracked_demo_h264.mp4
```

This produced the browser-compatible final result:

```text
tracked_demo_h264.mp4
```

---

## Google Colab Validation

The practical was successfully executed and validated in Google Colab.

Environment:

```text
OpenCV: 5.0.0
NumPy: 2.0.2
Supervision: 0.30.0
```

Environment verification:

```text
Environment test: SUCCESS
```

Input verification:

```text
Video exists: True
Video path: assets/input/tracking_demo.mp4
```

Tracking execution:

```text
Video Information
-----------------
Width: 960
Height: 540
FPS: 30.0
Frames: 300

Tracking completed successfully.
Output saved to: assets/output/tracked_demo.mp4
```

Output verification:

```text
Output exists: True
Output path: assets/output/tracked_demo.mp4
```

The final H.264 version was then displayed and visually verified directly inside Google Colab.

---

# Class Recording

The class recording associated with this section is documented in:

[`CLASS-RECORDING.md`](./CLASS-RECORDING.md)

The recording is available on YouTube:

[Watch the Object Tracking class recording on YouTube](https://youtu.be/UXN0l33NqF4)

The recording can be used together with the concept notes and practical implementation to review the complete lesson.

---

# Section Structure

```text
04-Object-Tracking/
├── README.md
├── CLASS-RECORDING.md
│
├── concepts/
│   └── ...
│
└── practical/
    ├── README.md
    ├── object_tracking_practical.py
    │
    └── assets/
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

# What We Completed

This section now includes:

- Object Tracking theory
- Detection-to-tracking workflow
- ByteTrack concepts
- `tracker_id` explanation
- Tracking annotations
- Object trajectories
- Tracking analytics concepts
- Class recording
- Practical documentation
- Custom 10-second input video
- Python tracking implementation
- Google Colab testing
- 300-frame tracking execution
- Final annotated tracking video
- H.264 browser-compatible output

---

# Next Step

The natural next step is to replace synthetic detections with real object detections.

The extended pipeline will become:

```text
Real-World Video
        ↓
YOLO
        ↓
sv.Detections
        ↓
Detection Filtering
        ↓
ByteTrack
        ↓
tracker_id
        ↓
Tracking Annotations
        ↓
Object Trajectories
        ↓
Tracking Analytics
        ↓
Processed Video
```

This will combine the object-detection techniques from previous sections with the tracking techniques developed in this section.

---

## Repository

[View the SAM3 Learning Journey on GitHub](https://github.com/Peyman-mxli/SAM3-Learning-Journey)

---

## Author

**Peyman Miyandashti**

- [GitHub](https://github.com/Peyman-mxli)
- [LinkedIn](https://www.linkedin.com/in/peyman-mxli/)
