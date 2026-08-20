# Zone Annotation and Visualization

## Introduction

Detecting, tracking, and counting objects produces useful data, but visualization makes those results much easier to understand.

In this session, **Supervision annotators** are used to display:

- Object bounding boxes
- Tracker IDs
- Polygon boundaries
- Current polygon occupancy
- Counting lines
- Directional crossing counts

The main visualization components are:

```python
sv.BoxAnnotator
sv.LabelAnnotator
sv.PolygonZoneAnnotator
sv.LineZoneAnnotator
```

Together, these components transform raw computer vision results into an understandable video analytics output.

---

# 1. Why Visualization Matters

A computer vision pipeline may internally know that:

```text
Object ID:7 → inside Zone A
Object ID:12 → outside Zone A
Current occupancy → 4
Crossings in → 15
Crossings out → 9
```

However, without visualization, it can be difficult to verify whether these results correspond correctly to the video.

Annotations allow us to visually inspect the system.

---

# 2. Visualization Pipeline

The general process is:

```text
Original Frame
      ↓
Object Detection
      ↓
Tracking
      ↓
Zone Analysis
      ↓
Counting
      ↓
Annotation
      ↓
Visual Output
```

The analytics happen first.

The visualization is then drawn on top of the frame.

---

# 3. BoxAnnotator

`BoxAnnotator` draws bounding boxes around detected or tracked objects.

It can be created using:

```python
box_annotator = sv.BoxAnnotator()
```

Then applied with:

```python
annotated = box_annotator.annotate(
    scene=frame.copy(),
    detections=detections
)
```

Conceptually:

```text
Detection
    ↓
Bounding Box
    ↓
Visible Object Location
```

---

# 4. Why Use `frame.copy()`?

A common pattern is:

```python
scene=frame.copy()
```

Instead of modifying the original frame directly, a copy is created for annotation.

This allows the original frame to remain unchanged if it is needed elsewhere in the pipeline.

Example:

```python
annotated = box_annotator.annotate(
    scene=frame.copy(),
    detections=detections
)
```

---

# 5. Drawing Only Objects Inside a Polygon

After polygon filtering:

```python
inside_zone = zone.trigger(
    detections=detections
)

detections_inside_zone = detections[
    inside_zone
]
```

we can draw only the selected objects:

```python
annotated = box_annotator.annotate(
    scene=frame.copy(),
    detections=detections_inside_zone
)
```

This produces a cleaner visualization focused on the monitored region.

---

# 6. LabelAnnotator

Bounding boxes show object location, but labels can provide additional information.

Supervision provides:

```python
sv.LabelAnnotator
```

It can be created with:

```python
label_annotator = sv.LabelAnnotator()
```

Labels can then be added to tracked detections.

---

# 7. Tracker ID Labels

After tracking, detections may contain:

```python
detections.tracker_id
```

Labels can be generated using:

```python
labels = [
    f"ID:{tracker_id}"
    for tracker_id in detections_inside_zone.tracker_id
]
```

Example labels:

```text
ID:2
ID:7
ID:11
```

These labels make it possible to visually follow the same object across multiple frames.

---

# 8. Drawing Labels

The labels can be applied with:

```python
annotated = label_annotator.annotate(
    scene=annotated,
    detections=detections_inside_zone,
    labels=labels
)
```

Now each tracked object can display its persistent identity.

Conceptually:

```text
┌──────────────────────┐
│ Car                  │
│ ID:7                 │
│                      │
└──────────────────────┘
```

---

# 9. PolygonZoneAnnotator

The polygon itself can be visualized using:

```python
sv.PolygonZoneAnnotator
```

Example:

```python
zone_annotator = sv.PolygonZoneAnnotator(
    zone=zone,
    color=sv.Color.RED,
    thickness=4
)
```

This creates an annotator associated with the polygon zone.

---

# 10. Drawing the Polygon

The polygon can be drawn using:

```python
annotated = zone_annotator.annotate(
    scene=annotated
)
```

The final frame now contains the visual boundary of the monitored area.

---

# 11. Polygon Count Visualization

`PolygonZoneAnnotator` can also display the zone's current count.

The count comes from:

```python
zone.current_count
```

which is updated when:

```python
zone.trigger(
    detections=detections
)
```

is called.

Therefore, the correct conceptual order is:

```text
Trigger Zone
     ↓
Update current_count
     ↓
Annotate Zone
     ↓
Display Current Count
```

---

# 12. Trigger Before Annotation

The zone should be triggered before drawing its count.

Example:

```python
inside_zone = zone.trigger(
    detections=detections
)

annotated = zone_annotator.annotate(
    scene=annotated
)
```

This ensures that the displayed count corresponds to the current frame.

---

# 13. Verifying Polygon Position

Before processing an entire video, the polygon can be visualized on the first frame.

Example:

```python
cap = cv2.VideoCapture(
    "assets/vehicles.mp4"
)

ret, first_frame = cap.read()

cap.release()
```

Then:

```python
frame_with_zone = zone_annotator.annotate(
    scene=first_frame.copy()
)
```

This makes it possible to verify the zone coordinates visually.

---

# 14. Displaying the Frame with Matplotlib

OpenCV uses BGR channel order while Matplotlib normally expects RGB.

Therefore:

```python
cv2.cvtColor(
    frame_with_zone,
    cv2.COLOR_BGR2RGB
)
```

can be used before displaying the frame.

Example:

```python
plt.figure(
    figsize=(12, 6)
)

plt.imshow(
    cv2.cvtColor(
        frame_with_zone,
        cv2.COLOR_BGR2RGB
    )
)

plt.axis("off")

plt.title(
    "Defined Zone"
)

plt.show()
```

---

# 15. Why BGR to RGB Conversion Matters

OpenCV normally represents image channels as:

```text
BGR
```

Matplotlib expects:

```text
RGB
```

Without conversion, colors may appear incorrect.

The conversion:

```python
cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)
```

ensures correct visualization.

---

# 16. LineZoneAnnotator

Supervision also provides:

```python
sv.LineZoneAnnotator
```

for visualizing counting lines.

Example from this session:

```python
line_zone_annotator = sv.LineZoneAnnotator(
    thickness=4,
    text_scale=1.5,
    custom_in_text="Crossings Down",
    custom_out_text="Crossings Up"
)
```

This annotator can display:

- The counting line
- One directional count
- The opposite directional count

---

# 17. Custom Counter Text

The labels displayed by the line annotator can be customized.

For example:

```python
custom_in_text="Crossings Down"
```

and:

```python
custom_out_text="Crossings Up"
```

This makes the meaning of the counters easier to understand in the final video.

---

# 18. LineZone Annotation API

An important implementation detail is that `LineZoneAnnotator` uses:

```python
frame=
```

and:

```python
line_counter=
```

Example:

```python
annotated = line_zone_annotator.annotate(
    frame=annotated,
    line_counter=line_zone
)
```

This differs from annotators that commonly use:

```python
scene=
```

---

# 19. Triggering Before Drawing Line Counts

Like polygon counting, the line trigger should occur before the annotation.

Correct order:

```python
line_zone.trigger(
    detections=detections
)

annotated = line_zone_annotator.annotate(
    frame=annotated,
    line_counter=line_zone
)
```

Conceptually:

```text
Tracked Objects
      ↓
LineZone.trigger()
      ↓
Update Crossing Counters
      ↓
LineZoneAnnotator
      ↓
Display Counters
```

---

# 20. Annotation Order

When several annotators are used, their order matters because each one modifies the frame produced by the previous step.

For example:

```python
annotated = box_annotator.annotate(
    scene=frame.copy(),
    detections=detections
)

annotated = zone_annotator.annotate(
    scene=annotated
)

annotated = line_zone_annotator.annotate(
    frame=annotated,
    line_counter=line_zone
)
```

The same `annotated` frame is passed from one annotator to the next.

---

# 21. Layered Visualization

The annotation process can be understood as layers.

```text
Original Frame
      ↓
Bounding Boxes
      ↓
Labels
      ↓
Polygon Zone
      ↓
Polygon Count
      ↓
Counting Line
      ↓
Crossing Counters
      ↓
Final Frame
```

Each layer adds additional information.

---

# 22. Combined Polygon and Line Visualization

A single frame can display both types of spatial analytics.

Conceptually:

```text
┌─────────────────────────────────────────┐
│                                         │
│       POLYGON ZONE                      │
│      ┌───────────────┐                  │
│      │ Car ID:7      │                  │
│      │ Car ID:11     │   Count: 2       │
│      └───────────────┘                  │
│                                         │
│══════════ COUNTING LINE ════════════════│
│                                         │
│ Crossings Down: 15                      │
│ Crossings Up: 9                         │
│                                         │
└─────────────────────────────────────────┘
```

This allows occupancy and flow to be understood simultaneously.

---

# 23. Complete Combined Annotation Pattern

A combined callback may follow this structure:

```python
def callback(frame: np.ndarray, _: int) -> np.ndarray:

    results = model(
        frame,
        verbose=False
    )[0]

    detections = sv.Detections.from_ultralytics(
        results
    )

    detections = tracker.update(
        detections
    )

    zone.trigger(
        detections=detections
    )

    line_zone.trigger(
        detections=detections
    )

    annotated = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )

    annotated = zone_annotator.annotate(
        scene=annotated
    )

    annotated = line_zone_annotator.annotate(
        frame=annotated,
        line_counter=line_zone
    )

    return annotated
```

This is the central idea behind the extension challenge from the session.

---

# 24. Processing the Video

Supervision provides:

```python
sv.process_video()
```

which can execute the callback for every frame.

Example:

```python
sv.process_video(
    source_path="assets/vehicles.mp4",
    target_path="assets/vehicles_output.mp4",
    callback=callback,
    show_progress=True
)
```

Conceptually:

```text
Input Video
     ↓
Frame 1 → callback()
Frame 2 → callback()
Frame 3 → callback()
...
     ↓
Annotated Frames
     ↓
Output Video
```

---

# 25. Why Video Annotation Is Useful

A final annotated video allows us to inspect:

- Detection quality
- Tracking stability
- Zone placement
- Tracker IDs
- Polygon occupancy
- Line crossing behavior
- Counter accuracy

This makes visualization an important debugging tool as well as a presentation tool.

---

# 26. Visualization as Debugging

Suppose the counter produces unexpected results.

Without visualization, it may be difficult to determine why.

With annotations, we can inspect whether:

```text
The bounding box is incorrect

The tracker changed IDs

The polygon is positioned incorrectly

The counting line is poorly placed

The object never actually crossed the line

The zone trigger is behaving differently than expected
```

Visualization therefore helps diagnose problems throughout the pipeline.

---

# 27. Visualization as Communication

Annotated videos are also useful when presenting a computer vision project.

Instead of showing only:

```text
current_count = 4
in_count = 15
out_count = 9
```

we can visually show where those numbers come from.

This makes the system easier to understand for people who did not build the code.

---

# 28. Real-World Dashboard Concept

The same information displayed directly on the video could eventually be sent to a dashboard.

For example:

```text
Camera 01

Current Occupancy: 12
Vehicles Entered: 148
Vehicles Exited: 136
```

The video annotations help validate the analytics before they are integrated into larger systems.

---

# 29. Important Annotators from This Session

The main annotators are:

| Annotator | Purpose |
|---|---|
| `BoxAnnotator` | Draw object bounding boxes |
| `LabelAnnotator` | Draw object labels and tracker IDs |
| `PolygonZoneAnnotator` | Draw polygon zone and current count |
| `LineZoneAnnotator` | Draw counting line and crossing counts |

Together they provide a complete visualization layer for the zone-counting pipeline.

---

# 30. Key Takeaways

The most important concepts are:

1. Visualization makes detection and tracking results easier to understand.
2. `BoxAnnotator` draws object bounding boxes.
3. `LabelAnnotator` can display tracker IDs.
4. `PolygonZoneAnnotator` visualizes polygon boundaries and current occupancy.
5. `LineZoneAnnotator` visualizes counting lines and directional crossing counts.
6. Zone triggers should run before their counters are visualized.
7. Annotation order determines how information is layered on the frame.
8. OpenCV BGR images should be converted to RGB when displayed with Matplotlib.
9. `LineZoneAnnotator` uses `frame=` and `line_counter=`.
10. Multiple annotators can operate on the same output frame.
11. Visualization is useful for both debugging and presenting computer vision systems.
12. `sv.process_video()` can apply the complete annotation pipeline to every frame.

---

# Summary

Zone annotation converts internal analytics into visible information.

The complete visualization process is:

```text
Detect
   ↓
Track
   ↓
Trigger Zones
   ↓
Update Counts
   ↓
Draw Bounding Boxes
   ↓
Draw Tracker Labels
   ↓
Draw Polygon
   ↓
Draw Counting Line
   ↓
Display Counts
   ↓
Annotated Video
```

This makes it possible to visually understand **what objects are detected, which objects are being tracked, where they are located, and how they move through the monitored environment**.

---

## Related Concepts

- [Concepts Overview](./README.md)
- [PolygonZone](./01-PolygonZone.md)
- [LineZone](./02-LineZone.md)
- [Occupancy vs Flow](./03-Occupancy-vs-Flow.md)
- [Tracking with Zones](./04-Tracking-with-Zones.md)
- [PolygonZone Trigger and Filtering](./05-PolygonZone-Trigger-and-Filtering.md)
- [Combining PolygonZone and LineZone](./07-Combining-PolygonZone-and-LineZone.md)

---

## Author

**Peyman Miyandashti**

- [GitHub](https://github.com/Peyman-mxli)
- [LinkedIn](https://www.linkedin.com/in/peyman-mxli/)
