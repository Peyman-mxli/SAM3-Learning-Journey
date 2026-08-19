# Tracking Annotations

After detecting and tracking objects, we need a clear way to visualize the results.

In this lesson, Supervision uses three annotators:

```python
sv.BoxAnnotator()
sv.LabelAnnotator()
sv.TraceAnnotator()
```

Together, they allow us to visualize:

- Where an object is
- Which object it is
- How the object has moved

The tracking pipeline therefore becomes:

```text
Detection
    ↓
Tracking
    ↓
tracker_id
    ↓
Annotation
    ↓
Visual Tracking Result
```

---

## Creating the Annotators

The lesson creates three annotators:

```python
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()
```

Each annotator has a different responsibility.

```text
BoxAnnotator
     ↓
Where is the object?

LabelAnnotator
     ↓
Which object is it?

TraceAnnotator
     ↓
Where has it moved?
```

---

## `BoxAnnotator`

The first annotator is:

```python
sv.BoxAnnotator()
```

Create it with:

```python
box_annotator = sv.BoxAnnotator()
```

Its purpose is to draw bounding boxes around detected objects.

For example:

```text
┌───────────────────┐
│                   │
│       CAR         │
│                   │
└───────────────────┘
```

The bounding box shows the current position of the object.

---

## Applying `BoxAnnotator`

After detection and tracking:

```python
detections = tracker.update_with_detections(
    detections
)
```

we can draw the boxes:

```python
annotated = box_annotator.annotate(
    scene=frame.copy(),
    detections=detections
)
```

The lesson uses:

```python
frame.copy()
```

so that the annotations are drawn on a copy of the original frame.

---

## `LabelAnnotator`

The second annotator is:

```python
sv.LabelAnnotator()
```

Create it with:

```python
label_annotator = sv.LabelAnnotator()
```

Labels allow us to display information about each tracked object.

For example:

```text
ID:1
ID:2
ID:3
```

---

## Creating Labels from `tracker_id`

After ByteTrack processes the detections:

```python
detections = tracker.update_with_detections(
    detections
)
```

each tracked detection contains a:

```python
tracker_id
```

The lesson creates labels using:

```python
labels = [
    f"ID:{tid}"
    for tid in detections.tracker_id
]
```

For example, if:

```python
detections.tracker_id
```

contains:

```text
[1, 2, 3]
```

the labels become:

```text
ID:1
ID:2
ID:3
```

---

## Why Use `tracker_id` in the Label?

The lesson makes an important distinction:

```text
tracker_id = who the object is

class_id = what type of object it is
```

For example:

```text
class_id
   ↓
car
```

while:

```text
tracker_id
   ↓
car #7
```

The tracker ID distinguishes one car from another.

---

## Applying `LabelAnnotator`

After creating the labels:

```python
labels = [
    f"ID:{tid}"
    for tid in detections.tracker_id
]
```

they can be drawn using:

```python
annotated = label_annotator.annotate(
    scene=annotated,
    detections=detections,
    labels=labels
)
```

Now each bounding box can display its tracking identity.

---

## Box + Tracking ID

Using the box and label annotators together produces a result conceptually similar to:

```text
      ID:5
┌───────────────────┐
│                   │
│       CAR         │
│                   │
└───────────────────┘
```

If several vehicles are visible:

```text
ID:1          ID:2           ID:3
┌──────┐      ┌──────┐       ┌──────┐
│ car  │      │ car  │       │truck │
└──────┘      └──────┘       └──────┘
```

The bounding box shows the object's position.

The label shows the object's tracking identity.

---

## `TraceAnnotator`

The third annotator introduced in the lesson is:

```python
sv.TraceAnnotator()
```

Create it with:

```python
trace_annotator = sv.TraceAnnotator()
```

Its purpose is to draw the trajectory of each tracked object.

The lesson specifically notes that `TraceAnnotator` needs:

```python
tracker_id
```

to work.

---

## Why Does `TraceAnnotator` Need `tracker_id`?

A trace represents movement across multiple frames.

For example:

```text
Frame 1 → Object at Position A

Frame 2 → Object at Position B

Frame 3 → Object at Position C
```

To connect those positions, Supervision must know that all three detections belong to the same object.

That information comes from:

```python
tracker_id
```

For example:

```text
Frame 1 → ID 4 → Position A
Frame 2 → ID 4 → Position B
Frame 3 → ID 4 → Position C
```

Because all three detections belong to:

```text
ID 4
```

their positions can form one trajectory.

---

## Visualizing a Trace

Conceptually, a trace may look like:

```text
•
 \
  •
   \
    •
     \
      •
       \
        [CAR #4]
```

The line represents the recent path followed by the tracked object.

---

## Applying `TraceAnnotator`

After drawing the boxes and labels, the lesson applies:

```python
annotated = trace_annotator.annotate(
    scene=annotated,
    detections=detections
)
```

The complete annotation sequence is therefore:

```python
annotated = box_annotator.annotate(
    scene=frame.copy(),
    detections=detections
)

annotated = label_annotator.annotate(
    scene=annotated,
    detections=detections,
    labels=labels
)

annotated = trace_annotator.annotate(
    scene=annotated,
    detections=detections
)
```

---

## Complete Tracking Annotation Pipeline

The full process used in the lesson is:

```text
Frame
  ↓
YOLO
  ↓
sv.Detections
  ↓
ByteTrack
  ↓
tracker_id
  ↓
Create Labels
  ↓
BoxAnnotator
  ↓
LabelAnnotator
  ↓
TraceAnnotator
  ↓
Annotated Frame
```

---

## Complete Example

The lesson uses a callback similar to:

```python
def procesar_frame(
    frame: np.ndarray,
    frame_idx: int
) -> np.ndarray:

    results = model(
        frame,
        verbose=False
    )[0]

    detections = sv.Detections.from_ultralytics(
        results
    )

    detections = tracker.update_with_detections(
        detections
    )

    labels = [
        f"ID:{tid}"
        for tid in detections.tracker_id
    ]

    annotated = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )

    annotated = label_annotator.annotate(
        scene=annotated,
        detections=detections,
        labels=labels
    )

    annotated = trace_annotator.annotate(
        scene=annotated,
        detections=detections
    )

    return annotated
```

This combines detection, tracking, and visualization in one frame-processing function.

---

## Showing Class and ID Together

The lesson also explores a more informative label.

Instead of displaying only:

```text
ID:3
```

we can display:

```text
car #3
```

or:

```text
truck #7
```

This combines:

```python
class_id
```

with:

```python
tracker_id
```

---

## Creating Class + ID Labels

The lesson creates these labels using:

```python
labels = [
    f"{results.names[c]} #{tid}"
    for c, tid in zip(
        detections.class_id,
        detections.tracker_id
    )
]
```

The resulting labels may look like:

```text
car #1
car #2
truck #3
bus #4
```

---

## Why Combine Class and ID?

Showing only the class:

```text
car
```

answers:

```text
What is this object?
```

Showing only the tracker ID:

```text
#5
```

answers:

```text
Which tracked object is this?
```

Combining them:

```text
car #5
```

answers both questions.

```text
WHAT?
car

WHICH ONE?
#5
```

This can make the visualization more informative.

---

## Experiment — Class and ID Together

The lesson defines a callback specifically for this experiment:

```python
def callback_clase_id(
    frame: np.ndarray,
    _: int
) -> np.ndarray:

    results = model(
        frame,
        verbose=False
    )[0]

    det = sv.Detections.from_ultralytics(
        results
    )

    det = tracker.update_with_detections(
        det
    )

    labels = [
        f"{results.names[c]} #{tid}"
        for c, tid in zip(
            det.class_id,
            det.tracker_id
        )
    ]

    scene = box_annotator.annotate(
        scene=frame.copy(),
        detections=det
    )

    return label_annotator.annotate(
        scene=scene,
        detections=det,
        labels=labels
    )
```

The video is then processed using:

```python
sv.process_video(
    source_path="assets/vehicles.mp4",
    target_path="assets/vehicles_clase_id.mp4",
    callback=callback_clase_id,
    show_progress=True
)
```

The resulting video is saved as:

```text
assets/vehicles_clase_id.mp4
```

---

## Choosing the Right Label

The lesson asks an important question:

```text
Which label is most useful for your application?
```

There are several possibilities.

### Class Only

```text
car
```

Useful when we mainly care about object categories.

### Tracker ID Only

```text
ID:4
```

Useful when we mainly care about following individual objects.

### Class + Tracker ID

```text
car #4
```

Useful when we need both category and identity.

The best choice depends on the purpose of the computer vision application.

---

## Annotation After Filtering

Annotations also work with filtered detections.

For example, the lesson filters only cars:

```python
CLASE_OBJETIVO = 2
```

Then:

```python
det = det[
    det.class_id == CLASE_OBJETIVO
]
```

After filtering, the detections are tracked:

```python
det = tracker.update_with_detections(
    det
)
```

Labels are created:

```python
labels = [
    f"auto #{tid}"
    for tid in det.tracker_id
]
```

Then the filtered tracked detections are annotated:

```python
scene = box_annotator.annotate(
    scene=frame.copy(),
    detections=det
)

scene = label_annotator.annotate(
    scene=scene,
    detections=det,
    labels=labels
)
```

---

## Annotation Order

The lesson follows this order:

```text
Detection
    ↓
Optional Filtering
    ↓
Tracking
    ↓
Create Labels
    ↓
Draw Bounding Boxes
    ↓
Draw Labels
    ↓
Draw Traces
```

This order is important because the labels and traces use information produced by the tracker.

---

## Why Tracking Comes Before Trace Annotation

Before tracking:

```python
detections.tracker_id
```

is:

```text
None
```

Without tracking information, the trace annotator cannot know which historical positions belong to which object.

After:

```python
detections = tracker.update_with_detections(
    detections
)
```

the detections contain tracking identities.

Now:

```python
sv.TraceAnnotator()
```

can associate movement with individual objects.

---

## Three Levels of Visualization

The three annotators can be understood as three levels of information.

### Level 1 — Position

```python
sv.BoxAnnotator()
```

Shows:

```text
Where is the object now?
```

### Level 2 — Identity

```python
sv.LabelAnnotator()
```

Shows:

```text
Which object is this?
```

### Level 3 — Movement

```python
sv.TraceAnnotator()
```

Shows:

```text
Where has this object moved?
```

Together:

```text
Position
   +
Identity
   +
Movement
   =
Tracking Visualization
```

---

## Relationship Between Detection and Annotation

The detector produces the information that describes objects:

```text
Bounding Boxes
Class IDs
Confidence
```

The tracker adds:

```text
tracker_id
```

The annotators then visualize that information.

```text
YOLO
  ↓
Detection Information
  ↓
ByteTrack
  ↓
Tracking Information
  ↓
Supervision Annotators
  ↓
Human-Readable Visualization
```

---

## Practical Example

Suppose a vehicle is detected and tracked as:

```text
class_id = car

tracker_id = 12
```

The bounding box tells us:

```text
where car #12 currently is
```

The label tells us:

```text
car #12
```

The trace tells us:

```text
where car #12 has recently moved
```

Together, the result provides much more information than a normal object detection box.

---

## Key Takeaways

- Supervision annotators visualize detection and tracking results.
- `BoxAnnotator` draws bounding boxes.
- `LabelAnnotator` displays information about tracked objects.
- `TraceAnnotator` draws object trajectories.
- `tracker_id` can be displayed directly in labels.
- `class_id` and `tracker_id` can be combined into labels such as `car #3`.
- `TraceAnnotator` requires tracking information to associate positions with the same object.
- Detection and tracking should happen before tracking annotations are created.
- Filtering can be performed before tracking and annotation.
- The choice between class labels, ID labels, or combined labels depends on the application.
- Boxes show position, labels show identity, and traces show movement.

---

## Next Concept

The next concept focuses on combining the previous lesson with object tracking:

```text
Detection
    ↓
Filtering
    ↓
Tracking
```

We will examine why detections should be filtered **before** being passed to ByteTrack and how to track only the classes relevant to an application.
