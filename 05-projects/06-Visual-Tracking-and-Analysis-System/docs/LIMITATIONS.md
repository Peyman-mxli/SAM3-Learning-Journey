# System Limitations and Failure Cases

This document describes the known limitations, constraints, and potential failure cases of the **Visual Tracking and Analysis System**.

The purpose is to document where the current system performs well, where results may become unreliable, and which areas could be improved in future versions.

---

## Current System

The current pipeline combines:

- YOLO object detection
- ByteTrack multi-object tracking
- SAM 3 segmentation
- Persistent tracker IDs
- SQLite persistence architecture
- Tracker-level analytics
- Trajectory analysis
- Confidence analysis
- Movement analysis
- Performance reporting
- H.264 video export

The current verified test run contains:

| Metric | Result |
|---|---:|
| Processed frames | 75 |
| Recorded observations | 246 |
| Unique tracker IDs | 6 |
| Average confidence | 0.6815 |
| Total movement distance | 693.18 px |

These results demonstrate that the complete detection, tracking, segmentation, and analytics workflow functions successfully on the current test video.

However, several limitations remain.

---

# 1. Object Detection Limitations

The tracking pipeline depends on the quality of the initial object detections.

If YOLO fails to detect an object in a frame, ByteTrack may temporarily lose the corresponding object.

Detection quality can be affected by:

- Small objects
- Motion blur
- Poor lighting
- Partial visibility
- Heavy occlusion
- Unusual viewing angles
- Low image resolution
- Objects near frame boundaries

A detection failure can propagate into later stages of the pipeline.

```text
Weak Detection
      ↓
Missing Observation
      ↓
Tracking Instability
      ↓
Incomplete Trajectory
      ↓
Less Reliable Analytics
```

---

# 2. Detection Confidence

The verified tracker-level confidence values range from:

```text
Minimum average confidence: 0.5124
Maximum average confidence: 0.8587
Overall average confidence: 0.6815
```

Lower-confidence detections may be less reliable than higher-confidence detections.

Confidence should therefore be considered when interpreting:

- Short-lived tracks
- Unexpected tracker IDs
- Temporary detections
- Potential false positives
- Tracking interruptions

Confidence scores are model estimates and should not be interpreted as guaranteed probabilities of correctness.

---

# 3. Tracker-ID Stability

ByteTrack assigns persistent IDs to detected objects across frames.

However, tracker IDs are not guaranteed to remain perfectly stable under all conditions.

Potential problems include:

- ID switches
- Lost tracks
- Reappearing objects receiving new IDs
- Multiple nearby objects being confused
- Temporary tracker fragmentation

For example:

```text
Frame 1
Person → ID 2

Frame 2
Person → ID 2

Frame 3
Occlusion

Frame 4
Person → ID 7
```

In this case, the same physical person could be represented by two different tracker IDs.

This would affect:

- Unique-object counts
- Duration calculations
- Movement calculations
- Trajectory reconstruction

---

# 4. Occlusion

Occlusion is one of the most important challenges in multi-object tracking.

Objects may become partially or completely hidden behind:

- Other people
- Vehicles
- Walls
- Furniture
- Scene structures
- Other foreground objects

Long occlusions increase the probability that a tracker will lose an object.

When the object becomes visible again, it may receive a different tracker ID.

---

# 5. Short-Lived Trackers

The current verified dataset contains trackers with significantly different observation durations.

For example, the minimum verified tracker duration is:

```text
0.20 seconds
```

while the maximum is:

```text
5.00 seconds
```

A short-lived tracker does not automatically indicate an error.

It may represent:

- An object briefly entering the frame
- An object leaving the frame
- A temporary detection
- Partial visibility
- A tracking interruption
- A possible false detection

Short tracks should therefore be interpreted together with confidence, class, and trajectory information.

---

# 6. SAM 3 Segmentation Limitations

SAM 3 provides segmentation masks associated with detected and tracked objects.

Segmentation quality may decrease when:

- Object boundaries are unclear
- Objects overlap
- The object is very small
- Motion blur is present
- The object is partially outside the frame
- Detection prompts are inaccurate
- Foreground and background have similar visual appearance

Segmentation masks should therefore be treated as model-generated estimates rather than perfect ground-truth boundaries.

---

# 7. Bounding-Box Dependency

The current SAM 3 integration depends on object information originating from the detection and tracking pipeline.

This creates a dependency:

```text
YOLO Detection
      ↓
ByteTrack
      ↓
SAM 3
```

If the original bounding box is incorrect, segmentation quality may also be affected.

Therefore, errors in earlier stages can propagate through the complete pipeline.

---

# 8. Movement Distance Is Image-Space Distance

The trajectory system calculates movement using object center coordinates.

Movement is measured in:

```text
pixels
```

The verified dataset produced:

```text
Total movement distance: 693.18 pixels
Maximum tracker movement: 203.37 pixels
Average movement per tracker: 115.53 pixels
```

These values do **not** represent meters, kilometers, or physical-world distance.

Pixel movement depends on:

- Camera resolution
- Camera angle
- Object distance from the camera
- Perspective
- Camera movement
- Lens characteristics

Physical distance estimation would require additional calibration.

---

# 9. Perspective Distortion

The same physical movement can produce different pixel distances depending on where an object appears in the scene.

For example, an object close to the camera may move many pixels while an object farther away may move only a small number of pixels despite traveling the same real-world distance.

Therefore:

```text
Pixel Distance ≠ Real-World Distance
```

Perspective calibration would be required for accurate physical measurements.

---

# 10. Camera Movement

Trajectory analysis assumes that changes in image coordinates primarily represent object movement.

If the camera itself moves, shakes, pans, tilts, or zooms, object coordinates may change even when the physical object remains stationary.

This can artificially increase calculated movement distance.

Future versions could incorporate:

- Camera-motion estimation
- Video stabilization
- Homography
- Background feature tracking

---

# 11. Frame Sampling

The current verified processing run contains:

```text
75 processed frames
```

Analytics are therefore based only on the frames processed by the pipeline.

If frames are skipped or sampled from a longer video, very short events occurring between processed frames may not be represented.

Higher temporal resolution can improve motion analysis but increases computational cost.

---

# 12. Processing Performance

The current performance report contains:

```text
Total Processing Time: N/A
Average Processing Time per Frame: N/A
Effective Processing FPS: N/A
```

The original total processing duration was not recorded during the verified run.

The project intentionally reports these values as `N/A` instead of estimating or inventing benchmark results.

The performance-analysis module supports measured processing time in future runs using:

```text
--processing-seconds
```

Once an actual processing duration is measured, the system can calculate:

- Effective processing FPS
- Average processing time per frame
- Total processing duration

---

# 13. Hardware Dependency

Computer-vision performance depends heavily on the available hardware.

Processing speed can vary depending on:

- GPU model
- Available VRAM
- CPU performance
- System memory
- Video resolution
- Model size
- Number of detected objects
- Segmentation workload

Results obtained in Google Colab may therefore have different processing speeds from local computers or production servers.

---

# 14. Colab Runtime Persistence

Google Colab runtime storage is temporary.

Files created only inside:

```text
/content/
```

can disappear when the runtime resets or disconnects.

For this reason, important project outputs should be preserved in GitHub or downloaded before the runtime ends.

The project currently preserves important analytical outputs such as:

```text
tracker_summary.csv
trajectory_summary.csv
performance_summary.csv

trajectory_visualization.png
tracker_duration_chart.png
class_observation_chart.png
movement_distance_chart.png
confidence_chart.png
performance_chart.png
```

This allows analytics to be reproduced without requiring the original temporary Colab runtime.

---

# 15. SQLite Runtime Persistence

The project includes SQLite persistence as part of the tracking architecture.

However, a database created only inside a temporary Colab runtime can be lost when that runtime resets.

The current performance-analysis stage therefore uses the preserved CSV reports:

```text
tracker_summary.csv
trajectory_summary.csv
```

instead of requiring the original temporary SQLite database.

This design prevents unnecessary reruns of:

- YOLO
- ByteTrack
- SAM 3

for system-level report generation.

---

# 16. Class Recognition Errors

Object classes are determined by the object-detection model.

Possible errors include:

- Incorrect class assignment
- Similar objects being confused
- Low-confidence classifications
- Objects outside the detector's supported classes

Incorrect class predictions can affect class-based analytics and observation counts.

---

# 17. Crowded Scenes

Tracking becomes more difficult when many similar objects appear close together.

Crowded scenes can increase the probability of:

- ID switches
- Overlapping bounding boxes
- Occlusion
- Tracker fragmentation
- Segmentation overlap
- Incorrect trajectory assignment

More complex scenes may therefore require additional tracker tuning.

---

# 18. Entry and Exit Events

The current system can identify when tracker IDs first and last appear, but it does not yet implement formal scene-entry or scene-exit event detection.

A tracker disappearing does not necessarily mean that the object physically exited the scene.

It could also mean:

- Detection failure
- Occlusion
- Tracker loss
- Object leaving the camera view

Formal entry/exit detection would require defined regions or scene boundaries.

---

# 19. No Physical Speed Estimation

The system currently measures movement in pixels.

Therefore, it does not currently provide reliable physical speed measurements such as:

```text
meters/second
kilometers/hour
miles/hour
```

Physical speed estimation would require:

- Camera calibration
- Real-world scale information
- Perspective correction
- Accurate frame timing

---

# 20. No Ground-Truth Accuracy Benchmark

The current project verifies that the complete pipeline operates successfully, but it does not currently include manually annotated ground-truth data.

Therefore, the project does not yet report formal metrics such as:

- Detection precision
- Detection recall
- mAP
- Tracking MOTA
- Tracking MOTP
- IDF1
- HOTA
- Segmentation IoU

These metrics would require a labeled evaluation dataset.

The current analytics describe the system's observed behavior rather than formal benchmark accuracy.

---

# Failure-Case Summary

| Failure Case | Possible Effect |
|---|---|
| Missed detection | Lost tracker observation |
| False detection | Incorrect tracker |
| Occlusion | Track interruption |
| ID switch | Incorrect object history |
| Tracker fragmentation | Duplicate object identities |
| Weak bounding box | Poor segmentation |
| Poor segmentation | Inaccurate object mask |
| Camera movement | Inflated movement distance |
| Perspective distortion | Misleading pixel-distance comparison |
| Low confidence | Less reliable observation |
| Frame sampling | Missed short events |
| Runtime reset | Loss of temporary files |
| Crowded scene | Increased tracking instability |

---

# Error Propagation

An important characteristic of the system is that errors can propagate between pipeline stages.

```text
Detection Error
      ↓
Tracking Error
      ↓
Segmentation Error
      ↓
Persistence Error
      ↓
Analytics Error
```

For this reason, analytical results should always be interpreted in the context of the underlying detection and tracking quality.

---

# Current Verified Strengths

Despite these limitations, the current Project 06 implementation successfully demonstrates:

- End-to-end image processing
- Recorded-video processing
- YOLO detection
- ByteTrack tracking
- Persistent tracker IDs
- SAM 3 segmentation integration
- H.264 video generation
- Structured observation persistence
- Tracker-level analytics
- Trajectory reconstruction
- Movement analysis
- Confidence analysis
- Multiple analytical visualizations
- System-level performance reporting
- Reusable CSV-based analytics

The current verified run processed:

```text
75 frames
246 observations
6 tracker IDs
```

without requiring the complete computer-vision pipeline to be rerun for subsequent analytics.

---

# Future Improvements

Potential improvements include:

- Ground-truth evaluation datasets
- MOTA and IDF1 tracking metrics
- Segmentation IoU evaluation
- ID-switch detection
- Tracker-fragmentation analysis
- Camera-motion compensation
- Perspective calibration
- Physical distance estimation
- Speed estimation
- Zone analytics
- Entry and exit detection
- Dwell-time analysis
- Region-of-interest monitoring
- Processing-time benchmarking
- GPU performance benchmarking
- Automated failure-case detection

These improvements could extend the current system from a functional computer-vision analytics project toward a more complete evaluation and monitoring platform.

---

# Related Documentation

- [Project README](../README.md)
- [Reports Documentation](../reports/README.md)
- [Analytics Documentation](../analytics/README.md)
- [Tracker Summary](../reports/tracker_summary.csv)
- [Trajectory Summary](../reports/trajectory_summary.csv)
- [Performance Summary](../reports/performance_summary.csv)
- [Performance Chart](../reports/performance_chart.png)

---

# Author

**Peyman Miyandashti**

GitHub: [Peyman-mxli](https://github.com/Peyman-mxli)

LinkedIn: [Peyman Miyandashti](https://www.linkedin.com/in/peyman-mxli/)
