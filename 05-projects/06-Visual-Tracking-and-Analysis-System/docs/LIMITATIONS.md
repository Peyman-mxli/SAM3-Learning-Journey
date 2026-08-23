# System Limitations and Failure Cases

This document describes the known limitations, constraints, evaluation results, and potential failure cases of the **Visual Tracking and Analysis System**.

The purpose is to document where the current system performs well, where results may become unreliable, and which conditions can affect detection, tracking, segmentation, and analytics.

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
- Ground-truth segmentation evaluation
- H.264 video export

The original verified tracking run contains:

| Metric | Result |
|---|---:|
| Processed frames | 75 |
| Recorded observations | 246 |
| Unique tracker IDs | 6 |
| Average confidence | 0.6815 |
| Total movement distance | 693.18 px |

The project has also completed a dedicated ground-truth evaluation using manually annotated `person` masks:

| Evaluation Metric | Result |
|---|---:|
| Evaluated images | 20 |
| Ground-truth instances | 424 |
| Predicted instances | 472 |
| True positives | 381 |
| False positives | 91 |
| False negatives / omissions | 43 |
| Precision | 0.8072 |
| Recall | 0.8986 |
| Average IoU | 0.7969 |
| Average Dice | 0.8829 |

These results demonstrate that the complete detection, tracking, segmentation, analytics, and evaluation workflow functions successfully on the current project data.

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

Occlusion is one of the most important challenges in multi-object tracking and segmentation.

Objects may become partially or completely hidden behind:

- Other people
- Vehicles
- Walls
- Furniture
- Scene structures
- Other foreground objects

Long occlusions increase the probability that a tracker will lose an object.

When the object becomes visible again, it may receive a different tracker ID.

Occlusion can also reduce segmentation quality because only part of the target object may remain visible.

In crowded scenes, overlapping people can make instance boundaries more difficult to separate.

Possible consequences include:

```text
Partial or Complete Occlusion
            ↓
Reduced Visible Object Area
            ↓
Detection or Segmentation Failure
            ↓
Missing Observation
            ↓
Tracker Interruption or New ID
```

Occlusion therefore remains a documented failure condition of the current system.

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

SAM 3 provides segmentation masks for objects identified through prompt-based segmentation.

Segmentation quality may decrease when:

- Object boundaries are unclear
- Objects overlap
- The object is very small
- Motion blur is present
- The object is partially outside the frame
- Detection prompts are inaccurate
- Foreground and background have similar visual appearance

The completed ground-truth evaluation confirms that segmentation is not perfect.

Across 20 manually annotated evaluation images:

```text
Ground-truth instances: 424
Predicted instances: 472

True positives: 381
False positives: 91
False negatives / omissions: 43
```

The measured segmentation overlap was:

```text
Average IoU:  0.7969
Average Dice: 0.8829
```

These results demonstrate substantial agreement with the manually annotated masks while also documenting measurable segmentation errors.

Segmentation masks should therefore continue to be treated as model-generated estimates rather than perfect object boundaries.

---

# 7. Bounding-Box and Detection Dependency

The complete Project 06 pipeline combines detection, tracking, and segmentation components.

This creates dependencies between processing stages:

```text
YOLO Detection
      ↓
ByteTrack Tracking
      ↓
SAM 3 Segmentation
      ↓
Analytics
```

Errors originating in detection or tracking can therefore affect later analytical results.

For example, an incorrect or missing detection can result in:

- Missing tracking observations
- Tracker fragmentation
- Incorrect class information
- Incomplete trajectories
- Less reliable analytics

For this reason, final analytical results should be interpreted together with the quality of the underlying model predictions.

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

The original verified processing run contains:

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

The ground-truth evaluation was executed using a CUDA-enabled Google Colab environment with a Tesla T4 GPU.

Flash Attention was unavailable on the T4 because of its GPU architecture, but standard SAM 3 inference completed successfully.

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

The project currently preserves important analytical and evaluation outputs such as:

```text
tracker_summary.csv
trajectory_summary.csv
performance_summary.csv
evaluation_metrics.csv
evaluation_summary.json

trajectory_visualization.png
tracker_duration_chart.png
class_observation_chart.png
movement_distance_chart.png
confidence_chart.png
performance_chart.png
```

This allows later analysis without requiring the original temporary Colab runtime.

---

# 15. SQLite Runtime Persistence

The project includes SQLite persistence as part of the tracking architecture.

However, a database created only inside a temporary Colab runtime can be lost when that runtime resets.

The current performance-analysis stage therefore uses preserved CSV reports such as:

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

The current ground-truth segmentation evaluation specifically evaluates the class:

```text
person
```

It should not be interpreted as a formal accuracy benchmark for every object class supported by the detection model.

---

# 17. Crowded Scenes

Tracking and segmentation become more difficult when many similar objects appear close together.

Crowded scenes can increase the probability of:

- ID switches
- Overlapping bounding boxes
- Occlusion
- Tracker fragmentation
- Segmentation overlap
- Incorrect trajectory assignment
- False positives
- Missed instances

The evaluation dataset includes frames from the more crowded Session 002 environment.

Several Session 002 images contain substantially more ground-truth person instances than the relatively sparse Session 001 frames.

For example, individual evaluated Session 002 frames contain dozens of manually annotated people.

The measured evaluation errors demonstrate that more complex scenes can produce both false positives and omissions.

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

# 20. Ground-Truth Evaluation Scope

Project 06 now includes a manually annotated ground-truth evaluation dataset.

The dataset contains:

```text
20 evaluation images
424 manually annotated person instances
```

The images were selected from:

```text
Session 001
Session 002
```

The annotations were created using Roboflow Instance Segmentation and exported in COCO Segmentation format.

The evaluation produced:

| Metric | Result |
|---|---:|
| Ground-truth instances | 424 |
| Predicted instances | 472 |
| True positives | 381 |
| False positives | 91 |
| False negatives / omissions | 43 |
| Precision | 0.8072 |
| Recall | 0.8986 |
| Average IoU | 0.7969 |
| Average Dice | 0.8829 |

The pixel-level confusion matrix contains:

| Pixel Classification | Count |
|---|---:|
| True-positive pixels | 1,604,567 |
| False-positive pixels | 341,396 |
| False-negative pixels | 229,371 |
| True-negative pixels | 20,864,666 |

This evaluation replaces the project's previous limitation of having no manually annotated ground-truth benchmark.

However, the benchmark still has limitations.

It evaluates:

- 20 images
- One target class: `person`
- Two recorded sessions
- The environmental conditions represented by those sessions

It does **not** provide formal benchmarks for:

- Every YOLO class
- MOTA
- MOTP
- IDF1
- HOTA
- Large-scale benchmark datasets
- Every possible environmental condition

Therefore, the current evaluation provides meaningful project-level evidence without claiming universal model accuracy.

---

# 21. Lighting Conditions

Lighting is a documented environmental limitation of the current system.

Computer-vision predictions may become less reliable when scenes contain:

- Low illumination
- Strong shadows
- Backlighting
- Overexposure
- Underexposure
- Sudden lighting changes
- Low contrast between people and the background

Possible effects include:

```text
Lighting Degradation
        ↓
Reduced Visual Features
        ↓
Lower Detection Confidence
        ↓
Missing or Incorrect Predictions
        ↓
Tracking / Segmentation Errors
```

The current evaluation dataset contains real recorded imagery, but it was not designed as a controlled lighting benchmark.

Therefore, the project documents lighting as a known failure condition without inventing a separate numerical lighting-accuracy result.

A future controlled evaluation could divide images into lighting categories and measure Precision, Recall, IoU, and Dice independently for each category.

---

# 22. Object Scale

Object scale is another documented limitation.

People far from the camera occupy fewer pixels than people close to the camera.

Small or distant objects provide less visual information for:

- Detection
- Classification
- Tracking
- Segmentation

Scale differences can therefore produce:

- Missed detections
- Incomplete masks
- Lower segmentation overlap
- Short-lived trackers
- Increased tracker fragmentation

The Session 002 evaluation data includes people appearing at different apparent sizes within the street scene.

However, the current evaluation does not separate results into formal small-, medium-, and large-object scale categories.

The project therefore documents scale sensitivity as a known limitation while preserving the measured overall evaluation results.

---

# 23. Out-of-Sample Data

The current system relies on pretrained models and has been evaluated using the visual conditions represented by the Project 06 datasets.

Out-of-sample data refers to images or videos that differ substantially from the environments represented during current testing.

Examples may include:

- Unusual camera viewpoints
- Extreme lighting conditions
- Unusual image resolutions
- Severe image compression
- Highly unusual backgrounds
- Very small or distant people
- Extreme crowd density
- Strong motion blur
- Infrared or non-standard imagery
- Environments substantially different from the tested street scenes

Performance on such data cannot be assumed to match the current evaluation results.

Possible effects include:

```text
Out-of-Sample Input
        ↓
Different Visual Distribution
        ↓
Reduced Model Reliability
        ↓
Detection / Segmentation Errors
        ↓
Tracking and Analytics Errors
```

The current Precision, Recall, IoU, and Dice values apply to the evaluated Project 06 dataset.

They should **not** be interpreted as guaranteed performance on arbitrary unseen environments.

Custom training or additional validation should only be considered if future target environments are not adequately handled by the pretrained models.

---

# 24. False Positives

The ground-truth evaluation measured:

```text
91 false-positive person predictions
```

across the complete 20-image evaluation dataset.

False positives occur when the system produces a predicted instance that does not satisfy the ground-truth matching requirement.

Possible causes include:

- Ambiguous visual structures
- Crowded regions
- Overlapping people
- Partial objects
- Model uncertainty
- Difficult object boundaries

The measured overall precision is:

```text
Precision: 0.8072
```

False positives are therefore a measured limitation rather than only a theoretical possibility.

---

# 25. False Negatives and Omissions

The ground-truth evaluation measured:

```text
43 false negatives / omissions
```

from:

```text
424 ground-truth person instances
```

The measured recall is:

```text
Recall: 0.8986
```

Possible causes of omissions include:

- Small people
- Distant people
- Occlusion
- Partial visibility
- Crowded scenes
- Difficult boundaries
- Insufficient visual information

These omissions demonstrate that the current system does not identify every manually annotated person instance.

---

# Failure-Case Summary

| Failure Case | Possible Effect |
|---|---|
| Missed detection | Lost tracker observation |
| False detection | Incorrect tracker or prediction |
| False-positive segmentation | Extra predicted instance |
| False negative / omission | Missing person instance |
| Poor lighting | Reduced detection or segmentation reliability |
| Small object scale | Missed or incomplete prediction |
| Occlusion | Track interruption or incomplete mask |
| Out-of-sample input | Reduced model reliability |
| ID switch | Incorrect object history |
| Tracker fragmentation | Duplicate object identities |
| Weak bounding box | Less reliable downstream analysis |
| Poor segmentation | Inaccurate object mask |
| Camera movement | Inflated movement distance |
| Perspective distortion | Misleading pixel-distance comparison |
| Low confidence | Less reliable observation |
| Crowded scene | Increased tracking and segmentation difficulty |
| Frame sampling | Missed short events |
| Runtime reset | Loss of temporary files |

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

For this reason, analytical results should always be interpreted in the context of the underlying detection, tracking, and segmentation quality.

The ground-truth evaluation provides quantitative evidence for segmentation performance, but it does not eliminate error propagation elsewhere in the pipeline.

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
- Historical session comparison
- Manual ground-truth annotation
- COCO Segmentation evaluation data
- SAM 3 ground-truth evaluation
- Precision measurement
- Recall measurement
- IoU measurement
- Dice measurement
- False-positive measurement
- Omission measurement
- Pixel-level confusion-matrix reporting

The original verified tracking run processed:

```text
75 frames
246 observations
6 tracker IDs
```

The dedicated ground-truth evaluation processed:

```text
20 evaluation images
424 ground-truth person instances
472 predicted instances
381 true positives
91 false positives
43 false negatives / omissions
```

without relying on estimated evaluation values.

---

# Future Improvements

Potential improvements include:

- Larger ground-truth evaluation datasets
- Additional environmental-condition datasets
- Controlled lighting evaluation
- Formal object-scale evaluation
- Additional out-of-sample testing
- MOTA and IDF1 tracking metrics
- HOTA tracking evaluation
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

Ground-truth evaluation, segmentation IoU, Dice, Precision, Recall, false-positive analysis, omission analysis, and the pixel confusion matrix are **no longer future work** because they have already been implemented and verified.

These future improvements could extend the current system from a validated computer-vision analytics project toward a broader evaluation and monitoring platform.

---

# Related Documentation

- [Project README](../README.md)
- [Project Results](./RESULTS.md)
- [Project Proposal](./PROJECT-PROPOSAL.md)
- [Reports Documentation](../reports/README.md)
- [Analytics Documentation](../analytics/README.md)
- [Evaluation Documentation](../evaluation/README.md)
- [Ground-Truth Documentation](../evaluation/ground_truth/README.md)
- [Tracker Summary](../reports/tracker_summary.csv)
- [Trajectory Summary](../reports/trajectory_summary.csv)
- [Performance Summary](../reports/performance_summary.csv)
- [Evaluation Metrics](../evaluation/evaluation_metrics.csv)
- [Evaluation Summary](../evaluation/evaluation_summary.json)
- [COCO Ground Truth](../evaluation/ground_truth/roboflow_export/_annotations.coco.json)
- [Performance Chart](../reports/performance_chart.png)

---

# Author

**Peyman Miyandashti**

GitHub: [Peyman-mxli](https://github.com/Peyman-mxli)

LinkedIn: [Peyman Miyandashti](https://www.linkedin.com/in/peyman-mxli/)
