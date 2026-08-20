# PolygonZone Trigger and Filtering

## Introduction

One of the most important operations when working with `PolygonZone` is:

```python
zone.trigger()
```

This method evaluates detections and determines which objects are currently inside the polygon.

The result is a **Boolean mask** that can be used to filter `sv.Detections`.

The basic idea is:

```text
Tracked Detections
        ↓
PolygonZone.trigger()
        ↓
Boolean Mask
        ↓
Filter Detections
        ↓
Objects Inside Zone
```

This pattern connects spatial analysis with the detection filtering concepts covered earlier in the course.

---

# 1. Starting with Detections

Before checking a zone, the object detector first processes the current frame.

Example:

```python
results = model(
    frame,
    verbose=False
)[0]
```

The YOLO results are then converted into Supervision detections:

```python
detections = sv.Detections.from_ultralytics(
    results
)
```

Conceptually:

```text
Video Frame
    ↓
YOLO
    ↓
YOLO Results
    ↓
sv.Detections
```

---

# 2. Adding Tracking

In this session, tracking is applied before the polygon trigger.

```python
detections = tracker.update(
    detections
)
```

Now the detections can contain persistent tracker IDs.

The pipeline becomes:

```text
Frame
  ↓
YOLO
  ↓
sv.Detections
  ↓
ByteTrack
  ↓
Tracked Detections
```

These tracked detections are then passed to the polygon.

---

# 3. Calling `trigger()`

The main operation is:

```python
inside_zone = zone.trigger(
    detections=detections
)
```

The polygon checks the detections and determines which objects belong to the zone.

The returned value is a Boolean array.

---

# 4. Boolean Mask

Suppose the frame contains five tracked detections.

Conceptually:

```text
Detection 0 → Car ID:2
Detection 1 → Car ID:4
Detection 2 → Truck ID:7
Detection 3 → Car ID:9
Detection 4 → Bus ID:11
```

The polygon might return:

```text
[True, False, True, True, False]
```

This means:

```text
Detection 0 → inside
Detection 1 → outside
Detection 2 → inside
Detection 3 → inside
Detection 4 → outside
```

Therefore:

```text
Total detections = 5
Inside zone      = 3
Outside zone     = 2
```

---

# 5. Understanding `True` and `False`

Each Boolean value corresponds to one detection.

```text
True
 ↓
Detection satisfies the zone condition

False
 ↓
Detection does not satisfy the zone condition
```

This is the same general idea used in many NumPy and computer vision filtering operations.

---

# 6. Filtering `sv.Detections`

The Boolean mask can be applied directly to the detections:

```python
detections_inside_zone = detections[
    inside_zone
]
```

If:

```text
inside_zone =
[True, False, True, True, False]
```

then:

```text
detections_inside_zone =
Detection 0
Detection 2
Detection 3
```

The other detections are excluded from the filtered result.

---

# 7. Visual Representation

The process can be represented as:

```text
ALL DETECTIONS

0 → ID:2
1 → ID:4
2 → ID:7
3 → ID:9
4 → ID:11

        ↓

POLYGON TRIGGER

[True, False, True, True, False]

        ↓

FILTER

        ↓

DETECTIONS INSIDE ZONE

ID:2
ID:7
ID:9
```

---

# 8. Counting the Boolean Mask

Because Boolean values behave like `1` and `0` when summed:

```text
True  → 1
False → 0
```

we can calculate the number of objects inside the polygon using:

```python
inside_zone.sum()
```

For example:

```python
inside_zone = np.array([
    True,
    False,
    True,
    True,
    False
])

print(inside_zone.sum())
```

Result:

```text
3
```

---

# 9. `current_count`

`PolygonZone` also maintains:

```python
zone.current_count
```

This represents the current number of objects inside the zone.

After:

```python
inside_zone = zone.trigger(
    detections=detections
)
```

we can inspect:

```python
print(inside_zone.sum())
print(zone.current_count)
```

For the same frame, these values should represent the same occupancy.

---

# 10. Two Ways to Inspect the Count

Conceptually:

```text
inside_zone.sum()
        ↓
Count True values in returned mask

zone.current_count
        ↓
Current occupancy maintained by PolygonZone
```

Both help us understand the current state of the polygon.

---

# 11. Why Filtering Is Useful

Suppose YOLO detects 20 vehicles in a frame.

Only five are inside the region we care about.

Without filtering:

```text
20 detections
```

With polygon filtering:

```text
5 relevant detections
```

This allows later processing to focus only on spatially relevant objects.

---

# 12. Annotation After Filtering

Instead of drawing every detected object:

```python
annotated = box_annotator.annotate(
    scene=frame.copy(),
    detections=detections
)
```

we can draw only objects inside the zone:

```python
annotated = box_annotator.annotate(
    scene=frame.copy(),
    detections=detections_inside_zone
)
```

The output becomes easier to interpret.

---

# 13. Filtering Tracker IDs

Because tracking was performed before filtering, the filtered detections retain their tracker IDs.

Example:

```python
detections_inside_zone.tracker_id
```

might contain:

```text
[2, 7, 9]
```

These IDs correspond only to objects currently inside the polygon.

---

# 14. Creating Labels

Tracker IDs can be converted into labels:

```python
labels = [
    f"ID:{tracker_id}"
    for tracker_id in detections_inside_zone.tracker_id
]
```

Result:

```text
ID:2
ID:7
ID:9
```

These labels can then be drawn using:

```python
label_annotator.annotate()
```

---

# 15. Complete Filtering Pattern

The central pattern from this session is:

```python
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

inside_zone = zone.trigger(
    detections=detections
)

detections_inside_zone = detections[
    inside_zone
]
```

This sequence combines:

```text
Detection
    ↓
Tracking
    ↓
Spatial Condition
    ↓
Boolean Filtering
```

---

# 16. Connection to Previous Detection Filtering

Earlier detection filtering concepts used conditions such as:

```text
Confidence
Class
Bounding-box size
Position
```

For example:

```python
mask = detections.confidence > 0.5
detections = detections[mask]
```

Polygon filtering follows the same general pattern.

Instead of:

```text
confidence > threshold
```

the condition becomes:

```text
object inside polygon
```

---

# 17. General Filtering Pattern

Many computer vision filters can be represented as:

```text
Detections
    ↓
Condition
    ↓
Boolean Mask
    ↓
Filtered Detections
```

Examples:

```text
Confidence Filter
        ↓
confidence > 0.50
```

```text
Class Filter
        ↓
class_id == target_class
```

```text
Position Filter
        ↓
x > frame_width / 2
```

```text
Polygon Filter
        ↓
zone.trigger()
```

The underlying idea is very similar.

---

# 18. Combining Filters

Polygon filtering can also be combined with other detection filters.

For example, suppose we only want vehicles with high confidence inside the zone.

Conceptually:

```text
All Detections
      ↓
Confidence Filter
      ↓
Vehicle Class Filter
      ↓
Tracking
      ↓
Polygon Trigger
      ↓
Objects of Interest Inside Zone
```

This allows increasingly specific analytics.

---

# 19. Example: Confidence + Polygon

A possible pattern could be:

```python
detections = sv.Detections.from_ultralytics(
    results
)

confidence_mask = detections.confidence >= 0.5

detections = detections[
    confidence_mask
]

detections = tracker.update(
    detections
)

zone_mask = zone.trigger(
    detections=detections
)

detections_inside_zone = detections[
    zone_mask
]
```

Now the result contains detections that satisfy both conditions.

---

# 20. Spatial Filtering

Polygon filtering is a form of:

```text
Spatial Filtering
```

Instead of selecting objects based only on what they are, we select them based on **where they are**.

This is an important progression in computer vision.

```text
Class Filtering
      ↓
"What type of object?"

Confidence Filtering
      ↓
"How certain is the model?"

Spatial Filtering
      ↓
"Where is the object?"
```

---

# 21. Dynamic Frame-by-Frame Filtering

The polygon remains fixed, but objects move.

Therefore, the Boolean mask can change every frame.

Example:

```text
Frame 1

[True, False, False]
```

Then:

```text
Frame 2

[True, True, False]
```

Then:

```text
Frame 3

[False, True, True]
```

This reflects objects entering and leaving the region.

---

# 22. Example Object Movement

Suppose `ID:5` moves toward the polygon.

```text
Frame 10 → ID:5 outside
Frame 11 → ID:5 outside
Frame 12 → ID:5 inside
Frame 13 → ID:5 inside
Frame 14 → ID:5 inside
Frame 15 → ID:5 outside
```

The corresponding zone condition changes over time:

```text
False
False
True
True
True
False
```

This is how occupancy changes dynamically.

---

# 23. Empty Zone

If no objects are inside the polygon, the mask might contain:

```text
[False, False, False, False]
```

Then:

```python
inside_zone.sum()
```

returns:

```text
0
```

and:

```python
zone.current_count
```

should also represent zero current occupancy.

---

# 24. All Objects Inside

If every detected object is inside the polygon:

```text
[True, True, True, True]
```

then all detections are preserved.

```text
Total detections = 4
Inside zone      = 4
```

---

# 25. Why Boolean Masks Are Powerful

Boolean masks provide a simple way to create complex processing pipelines.

For example:

```text
Detections
    ↓
Confidence Mask
    ↓
Class Mask
    ↓
Zone Mask
    ↓
Final Relevant Objects
```

Each stage reduces the data to the objects needed for the application.

---

# 26. Real-World Example

Imagine a traffic camera detecting:

```text
12 vehicles
```

Only vehicles in the left lane matter.

A polygon is placed over that lane.

The trigger returns a mask selecting:

```text
5 vehicles
```

Now the system can calculate:

```text
Left-lane occupancy = 5
```

without being affected by vehicles elsewhere in the frame.

---

# 27. From Raw Detection to Useful Information

The transformation looks like:

```text
Raw Frame
    ↓
YOLO
    ↓
12 Vehicle Detections
    ↓
ByteTrack
    ↓
12 Tracked Vehicles
    ↓
PolygonZone
    ↓
5 Vehicles Inside Left Lane
    ↓
Occupancy = 5
```

This demonstrates how filtering converts general detections into application-specific information.

---

# 28. Key Takeaways

The most important concepts are:

1. `zone.trigger()` evaluates detections against a polygon.
2. The method returns a Boolean mask.
3. `True` represents a detection associated with the zone.
4. `False` represents a detection outside the zone.
5. Boolean masks can directly filter `sv.Detections`.
6. `mask.sum()` counts the number of `True` values.
7. `zone.current_count` represents current polygon occupancy.
8. Tracking should occur before the zone trigger in this pipeline.
9. Filtered detections retain useful information such as tracker IDs.
10. Polygon filtering is a form of spatial detection filtering.
11. Zone filtering can be combined with confidence, class, size, and other filters.
12. The Boolean mask changes dynamically as objects move through the scene.

---

# Summary

The central operation is:

```python
zone_mask = zone.trigger(
    detections=detections
)

detections_inside_zone = detections[
    zone_mask
]
```

Conceptually:

```text
Tracked Objects
      ↓
Where are they?
      ↓
PolygonZone.trigger()
      ↓
Boolean Mask
      ↓
Select Relevant Objects
      ↓
Count and Visualize
```

This simple filtering mechanism is one of the key building blocks for creating **spatial computer vision analytics systems**.

---

## Related Concepts

- [Concepts Overview](./README.md)
- [PolygonZone](./01-PolygonZone.md)
- [LineZone](./02-LineZone.md)
- [Occupancy vs Flow](./03-Occupancy-vs-Flow.md)
- [Tracking with Zones](./04-Tracking-with-Zones.md)
- [Zone Annotation and Visualization](./06-Zone-Annotation-and-Visualization.md)

---

## Author

**Peyman Miyandashti**

- [GitHub](https://github.com/Peyman-mxli)
- [LinkedIn](https://www.linkedin.com/in/peyman-mxli/)
