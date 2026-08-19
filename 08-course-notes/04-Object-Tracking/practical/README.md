# Object Tracking — Practical Exercises

This directory contains the practical work associated with the **Object Tracking** section of the SAM3 Computer Vision Learning Journey.

The practical extends object detection concepts into video-based tracking workflows, where detections are associated across consecutive frames and assigned persistent tracking identities.

The exercise was implemented and tested successfully in **Google Colab**.

---

## Practical Focus

The practical focuses on:

- OpenCV video processing
- Supervision
- ByteTrack
- `sv.Detections`
- Synthetic detections
- Persistent tracking IDs
- `tracker_id`
- Bounding-box annotations
- Tracking labels
- Object trajectories
- Frame-by-frame processing
- Video export
- H.264 video conversion

---

## Learning Objectives

Through this practical exercise, we learned how to:

- Load an MP4 video with OpenCV
- Read a video frame by frame
- Generate detections for known moving objects
- Represent detections using `sv.Detections`
- Pass detections to ByteTrack
- Associate objects between consecutive frames
- Retrieve persistent `tracker_id` values
- Create labels containing object names and tracking IDs
- Draw bounding boxes
- Visualize object trajectories
- Write processed frames to a new video
- Verify the generated output
- Convert an OpenCV-generated MP4 to browser-compatible H.264
- Display the final tracking result inside Google Colab

---

## Why Synthetic Detections Are Used

The input video used in this practical is a synthetic demonstration video created specifically for learning object-tracking concepts.

The objects are simple generated shapes rather than standard real-world YOLO classes.

Because of this, using YOLO would introduce unreliable detections that are unrelated to the main purpose of the exercise.

Instead, the practical generates detections that follow the known positions of the objects in the video.

This allows the exercise to focus directly on:

```text
Detections
    ↓
sv.Detections
    ↓
ByteTrack
    ↓
tracker_id
    ↓
Tracking Visualization
```

A future real-world tracking exercise can extend this pipeline by replacing the synthetic detections with YOLO predictions.

---

## Main Workflow

The completed practical follows this pipeline:

```text
Input Video
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
Tracking Annotations
    ↓
Object Trajectories
    ↓
Processed Video
    ↓
H.264 Conversion
    ↓
Final Browser-Compatible Video
```

---

## Input Video

The practical uses:

```text
assets/input/tracking_demo.mp4
```

Video properties:

```text
Duration: 10 seconds
Resolution: 960 × 540
Frame Rate: 30 FPS
Total Frames: 300
Format: MP4
```

The video contains three moving synthetic objects.

The objects follow different motion paths, allowing ByteTrack to demonstrate persistent identity assignment across consecutive frames.

---

## Practical Script

The implementation is located at:

```text
object_tracking_practical.py
```

The script performs the following operations:

1. Loads the input video.
2. Reads its properties.
3. Initializes ByteTrack.
4. Creates Supervision annotators.
5. Reads each video frame.
6. Generates detections corresponding to the moving objects.
7. Converts those detections into `sv.Detections`.
8. Sends the detections to ByteTrack.
9. Retrieves persistent `tracker_id` values.
10. Creates tracking labels.
11. Draws object bounding boxes.
12. Draws object trajectories.
13. Adds frame information.
14. Writes each annotated frame to the output video.
15. Releases the video-processing resources.

---

## ByteTrack

The practical uses:

```python
tracker = sv.ByteTrack(
    frame_rate=FPS
)
```

ByteTrack receives detections from consecutive frames and attempts to associate them with previously tracked objects.

Instead of treating every detection as a completely new object, ByteTrack assigns persistent IDs.

Conceptually:

```text
Frame 1 → object_a #1
Frame 2 → object_a #1
Frame 3 → object_a #1
Frame 4 → object_a #1
```

The object moves, but its tracking identity remains associated with it.

---

## Tracking IDs

The final exercise tracks three objects:

```text
object_a #1
object_b #2
object_c #3
```

These labels combine:

```text
Object Name + tracker_id
```

For example:

```python
f"{class_name} #{tracker_id}"
```

The `tracker_id` is provided by ByteTrack after the detections are associated across frames.

---

## Tracking Annotations

The practical uses Supervision annotators to visualize tracking information.

### Bounding Boxes

```python
sv.BoxAnnotator()
```

draws bounding boxes around tracked objects.

### Labels

```python
sv.LabelAnnotator()
```

displays the object name and tracking ID.

### Traces

```python
sv.TraceAnnotator()
```

visualizes the recent movement history of each tracked object.

Conceptually:

```text
● → ● → ● → ●
            ↑
     Current Position
```

---

## Generated Output

The Python tracking script initially generates:

```text
assets/output/tracked_demo.mp4
```

The tracking pipeline successfully processed:

```text
Width: 960
Height: 540
FPS: 30.0
Frames: 300
```

and completed with:

```text
Tracking completed successfully.
Output saved to: assets/output/tracked_demo.mp4
```

---

## H.264 Conversion

The initial OpenCV-generated MP4 was successfully created, but its `mp4v` encoding was not reliably playable in the Google Colab browser video player.

The video was therefore converted with FFmpeg using H.264:

```bash
ffmpeg -y \
-i assets/output/tracked_demo.mp4 \
-c:v libx264 \
-pix_fmt yuv420p \
-movflags +faststart \
assets/output/tracked_demo_h264.mp4
```

The final browser-compatible video is:

```text
assets/output/tracked_demo_h264.mp4
```

---

## Final Result

The final video successfully demonstrates:

- Three independently moving objects
- Bounding-box tracking
- Persistent tracking IDs
- Object labels
- Object trajectories
- Frame information
- Tracking across 300 consecutive frames

The final tracked objects are displayed as:

```text
object_a #1
object_b #2
object_c #3
```

The completed practical demonstrates the progression:

```text
Individual Detections
        ↓
Object Association
        ↓
ByteTrack
        ↓
Persistent tracker_id
        ↓
Tracking Visualization
        ↓
Video-Based Object Tracking
```

---

## Directory Structure

```text
practical/
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

## Google Colab Testing

The practical was tested successfully in Google Colab using:

```text
OpenCV: 5.0.0
NumPy: 2.0.2
Supervision: 0.30.0
```

Environment verification returned:

```text
Environment test: SUCCESS
```

The input video was detected successfully:

```text
Video exists: True
Video path: assets/input/tracking_demo.mp4
```

The generated output was also verified successfully:

```text
Output exists: True
Output path: assets/output/tracked_demo.mp4
```

The final H.264 video was displayed and visually inspected directly inside Google Colab.

---

## Technologies

- Python
- OpenCV
- NumPy
- Supervision
- ByteTrack
- FFmpeg
- Google Colab
- GitHub

---

## Next Extension

The next natural extension of this practical is to replace the synthetic detections with detections produced by a real object-detection model.

The extended pipeline would become:

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

This would combine the object-detection concepts from previous lessons with the tracking concepts demonstrated here.

---

## Related Course Material

This practical complements the **Object Tracking** concept notes and class recording.

[Watch the Object Tracking class recording on YouTube](https://youtu.be/UXN0l33NqF4)

---

## Repository

[View the SAM3 Learning Journey on GitHub](https://github.com/Peyman-mxli/SAM3-Learning-Journey)

---

## Author

**Peyman Miyandashti**

- [GitHub](https://github.com/Peyman-mxli)
- [LinkedIn](https://www.linkedin.com/in/peyman-mxli/)
