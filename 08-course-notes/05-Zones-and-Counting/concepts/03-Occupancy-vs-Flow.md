# Occupancy vs Flow

## Introduction

One of the most important concepts in zone-based computer vision analytics is understanding the difference between **occupancy** and **flow**.

These two measurements answer different questions:

```text
Occupancy
    ↓
"How many objects are here right now?"

Flow
    ↓
"How many objects have passed through?"
```

In this session:

```text
PolygonZone → Occupancy
LineZone    → Flow
```

Understanding this distinction is essential when designing traffic monitoring, retail analytics, parking, security, and industrial computer vision systems.

---

# 1. What Is Occupancy?

**Occupancy** measures how many objects are currently present inside a defined region.

It represents the state of that region at a particular moment.

For example:

```text
Frame 1 → 2 vehicles inside
Frame 2 → 3 vehicles inside
Frame 3 → 5 vehicles inside
Frame 4 → 4 vehicles inside
Frame 5 → 1 vehicle inside
```

The value can increase or decrease as objects enter and leave the area.

---

## Occupancy with PolygonZone

In Supervision, occupancy can be measured using:

```python
sv.PolygonZone
```

The current number of objects inside the polygon is available through:

```python
zone.current_count
```

For example:

```python
inside_zone = zone.trigger(
    detections=detections
)

print(zone.current_count)
```

If three tracked objects are currently inside the polygon:

```text
zone.current_count = 3
```

---

# 2. Occupancy Is Instantaneous

Occupancy represents the current state.

Consider this example:

```text
10:00:01 → 3 cars inside
10:00:02 → 4 cars inside
10:00:03 → 4 cars inside
10:00:04 → 2 cars inside
```

The system is not asking how many cars have visited the area in total.

It is asking:

> How many cars are inside the area at this moment?

This is why `PolygonZone` can be compared to a virtual occupancy sensor.

---

# 3. What Is Flow?

**Flow** measures movement through a boundary.

Instead of asking how many objects are currently present, it counts how many objects have crossed a particular line.

For example:

```text
Vehicle 1 crosses → total = 1
Vehicle 2 crosses → total = 2
Vehicle 3 crosses → total = 3
Vehicle 4 crosses → total = 4
```

The count accumulates as the video is processed.

---

## Flow with LineZone

In Supervision, flow can be measured using:

```python
sv.LineZone
```

The directional crossing counters are:

```python
line_zone.in_count
line_zone.out_count
```

For example:

```text
in_count  = 15
out_count = 9
```

This means that crossing events were detected in both directions.

---

# 4. Flow Is Accumulated

Unlike occupancy, flow normally increases as new crossing events occur.

For example:

```text
Frame 1   → crossings = 0
Frame 40  → crossings = 1
Frame 95  → crossings = 2
Frame 150 → crossings = 3
Frame 220 → crossings = 4
```

The system remembers previous crossing events.

This allows us to answer:

> How many objects have passed through this location?

---

# 5. Direct Comparison

The core difference can be summarized as:

| Feature | Occupancy | Flow |
|---|---|---|
| Supervision tool | `PolygonZone` | `LineZone` |
| Measures | Current presence | Crossing events |
| Time behavior | Changes frame by frame | Accumulates over time |
| Main property | `current_count` | `in_count` / `out_count` |
| Spatial representation | Area | Line |
| Main question | How many are here now? | How many passed? |
| Example | Cars currently in parking area | Cars entering parking lot |

---

# 6. Simple Analogy

A useful analogy is a building.

Imagine a room with a door.

```text
             ROOM

     ┌─────────────────────┐
     │                     │
     │   Person A          │
     │   Person B          │
     │   Person C          │
     │                     │
     └─────────┬───────────┘
               │
              DOOR
```

A polygon covering the room answers:

```text
How many people are currently inside?

Answer: 3
```

This is **occupancy**.

A line across the door answers:

```text
How many people crossed the door today?

Answer: 27
```

This is **flow**.

---

# 7. Parking Lot Example

A parking system demonstrates why both measurements can be useful.

Suppose a camera monitors a parking lot.

## PolygonZone

A polygon can cover the parking area.

```text
PolygonZone
     ↓
Vehicles currently inside
     ↓
Occupancy
```

Example:

```text
Current vehicles = 18
```

---

## LineZone

A line can be placed across the entrance.

```text
LineZone
    ↓
Vehicles crossing entrance
    ↓
Traffic Flow
```

Example:

```text
Vehicles entered = 52
Vehicles exited  = 34
```

These are different measurements describing the same environment.

---

# 8. Traffic Example

Consider a road with a polygon covering one lane.

```text
┌───────────────────────────────┐
│                               │
│        POLYGON ZONE           │
│                               │
│    Car A      Car B           │
│                               │
└───────────────────────────────┘

════════ COUNTING LINE ═════════
```

The polygon might report:

```text
current_count = 2
```

This means:

> Two vehicles are currently inside the monitored area.

The line might report:

```text
in_count = 147
```

This means:

> 147 crossing events have been recorded in that direction.

Both numbers can be correct at the same time.

---

# 9. Why These Values Should Not Be Confused

Suppose a store had:

```text
500 customers enter during the day
```

but currently only:

```text
25 customers are inside
```

Then:

```text
Flow      = 500 entries
Occupancy = 25 current customers
```

Reporting the flow value as occupancy would incorrectly suggest that 500 people are currently inside the store.

Therefore, the metric must match the question being asked.

---

# 10. Choosing the Correct Zone

Before implementing a counting system, ask:

> Do I care about the current state of an area or movement through a boundary?

If the question is:

```text
How many objects are inside this area?
```

use:

```python
sv.PolygonZone
```

If the question is:

```text
How many objects crossed this boundary?
```

use:

```python
sv.LineZone
```

---

# 11. Occupancy Use Cases

Occupancy analysis is useful when monitoring the current state of an area.

Examples include:

### Parking

```text
How many parking spaces are occupied?
```

### Traffic

```text
How many vehicles are waiting at the intersection?
```

### Retail

```text
How many customers are currently in this section?
```

### Security

```text
Is anyone currently inside the restricted area?
```

### Industrial Safety

```text
How many workers are inside the hazardous zone?
```

---

# 12. Flow Use Cases

Flow analysis is useful when measuring movement.

Examples include:

### Traffic

```text
How many vehicles passed this road?
```

### Retail

```text
How many customers entered the store?
```

### Buildings

```text
How many people crossed the entrance?
```

### Parking

```text
How many vehicles entered and exited?
```

### Industrial Systems

```text
How many products passed the inspection point?
```

---

# 13. Combining Occupancy and Flow

More advanced systems often need both measurements.

For example, a parking management system may need:

```text
Vehicles currently parked
        +
Vehicles entering
        +
Vehicles leaving
```

This can be implemented using:

```text
PolygonZone
     +
LineZone
```

---

# 14. Same Detections, Different Analysis

Both zones can analyze the same tracked detections.

Example:

```python
zone.trigger(
    detections=detections
)

line_zone.trigger(
    detections=detections
)
```

The same object tracking pipeline therefore supports two different spatial measurements.

Conceptually:

```text
             Tracked Detections
                    │
          ┌─────────┴─────────┐
          │                   │
          ↓                   ↓
     PolygonZone          LineZone
          │                   │
          ↓                   ↓
      Occupancy              Flow
```

---

# 15. Why Tracking Matters

Tracking improves spatial analysis by maintaining object identities.

For example:

```text
Frame 100 → Vehicle ID 5
Frame 101 → Vehicle ID 5
Frame 102 → Vehicle ID 5
Frame 103 → Vehicle ID 5
```

The system understands that these detections represent the same vehicle.

This is especially important for crossing analysis because a `LineZone` must determine how a specific object moves relative to the line.

---

# 16. Temporal Difference

Another way to understand occupancy and flow is through time.

## Occupancy

Occupancy describes:

```text
STATE AT TIME t
```

For example:

```text
Occupancy(t) = number of objects currently inside
```

---

## Flow

Flow describes:

```text
EVENTS OVER A TIME INTERVAL
```

For example:

```text
Flow(t₁ → t₂) = number of crossings during the interval
```

This distinction is useful when designing analytics systems.

---

# 17. Occupancy Can Increase and Decrease

Suppose a polygon currently contains four objects.

```text
Occupancy = 4
```

Two objects leave.

```text
Occupancy = 2
```

Three new objects enter.

```text
Occupancy = 5
```

Therefore, occupancy can move in either direction:

```text
2 → 5 → 3 → 7 → 1
```

---

# 18. Flow Counters Accumulate

A line counter behaves differently.

Suppose:

```text
in_count = 10
```

Another object crosses in the same direction:

```text
in_count = 11
```

Another crosses:

```text
in_count = 12
```

The counter records accumulated crossing events rather than current presence.

---

# 19. Spatial Analytics

Occupancy and flow are two forms of **spatial video analytics**.

The progression is:

```text
Object Detection
        ↓
Identify Objects

Object Tracking
        ↓
Maintain Identity

Spatial Zones
        ↓
Understand Location

Occupancy / Flow
        ↓
Generate Metrics
```

This transforms raw video into structured information.

---

# 20. From Computer Vision to Business Information

Computer vision models produce detections.

For example:

```text
car
person
truck
bus
```

Tracking adds identities:

```text
car ID:4
car ID:8
truck ID:12
```

Zones add spatial context:

```text
ID:4 inside Zone A
ID:8 crossed Line B
```

Counting produces useful metrics:

```text
Zone A occupancy = 7
Line B crossings = 125
```

This is how computer vision becomes useful for real-world analytics.

---

# 21. Key Takeaways

The most important concepts are:

1. Occupancy measures current presence.
2. Flow measures accumulated movement.
3. `PolygonZone` is primarily used for occupancy.
4. `LineZone` is primarily used for flow.
5. `zone.current_count` represents current polygon occupancy.
6. `line_zone.in_count` represents crossings in one direction.
7. `line_zone.out_count` represents crossings in the opposite direction.
8. Occupancy can increase or decrease every frame.
9. Crossing counters accumulate as events occur.
10. The same tracked detections can support both measurements.
11. Tracking provides persistent object identities.
12. Occupancy and flow should be selected according to the real-world question being answered.

---

# Summary

The central idea of this concept is:

```text
PolygonZone
     ↓
CURRENT STATE
     ↓
Occupancy

LineZone
     ↓
ACCUMULATED EVENTS
     ↓
Flow
```

A simple way to remember the difference is:

> **PolygonZone:** How many are here?

> **LineZone:** How many passed?

Combining both allows a computer vision system to understand not only **where objects currently are**, but also **how objects move through an environment**.

---

## Related Concepts

- [Concepts Overview](./README.md)
- [PolygonZone](./01-PolygonZone.md)
- [LineZone](./02-LineZone.md)
- [Tracking with Zones](./04-Tracking-with-Zones.md)

---

## Author

**Peyman Miyandashti**

- [GitHub](https://github.com/Peyman-mxli)
- [LinkedIn](https://www.linkedin.com/in/peyman-mxli/)
