# Class Recording — Zones and Counting

This page contains the video recording for **Session 06 — Zones and Counting** of my SAM3 Computer Vision Learning Journey.

---

## Session

**Session 06 — Module 02**

**Topic:** Zones and Counting

**Notebook:** `02_c_zonas_conteo.ipynb`

---

## Video Recording

The practical session is available on YouTube:

[Watch the full Zones and Counting session on YouTube](https://youtu.be/43i0z9b81Z4)

---

## Topics Covered

The session covers the implementation of spatial video analytics using object detection, tracking, and counting zones.

Main topics include:

- YOLOv8 object detection
- Supervision `sv.Detections`
- ByteTrack object tracking
- Persistent tracker IDs
- Polygon zones
- Line zones
- Current occupancy
- Directional crossing counts
- Zone triggering
- Detection filtering
- Zone annotation
- Video processing
- Combining `PolygonZone` and `LineZone`

---

## PolygonZone

`PolygonZone` is used to determine how many tracked objects are currently inside a defined region.

Conceptually:

```text
Tracked Objects
      ↓
PolygonZone
      ↓
Objects Inside Region
      ↓
Current Occupancy
```

The current occupancy is available through:

```python
zone.current_count
```

---

## LineZone

`LineZone` is used to detect objects crossing a virtual line.

Conceptually:

```text
Tracked Objects
      ↓
LineZone
      ↓
Crossing Detection
      ↓
Directional Counts
```

The accumulated counts are available through:

```python
line_zone.in_count
line_zone.out_count
```

---

## Combined Practical

The final practical combines both systems:

```text
Input Video
     ↓
YOLOv8
     ↓
Detections
     ↓
ByteTrack
     ↓
Tracked Objects
     │
     ├──────────────┐
     ↓              ↓
PolygonZone      LineZone
     ↓              ↓
Occupancy         Flow
     │              │
     └──────┬───────┘
            ↓
       Visualization
            ↓
      Output Video
```

---

## Practical Results

The tested practical processed:

```text
Resolution: 3840 × 2160
FPS: 25
Total Frames: 538
```

The final combined pipeline produced:

```text
Final polygon occupancy: 1
Crossings Down: 3
Crossings Up: 3
```

The final generated demonstration is stored at:

```text
practical/assets/output/vehicles_combined.mp4
```

---

## Practical Implementation

The complete tested Python implementation is available here:

[`practical/zones_and_counting_practical.py`](./practical/zones_and_counting_practical.py)

The practical uses:

```text
Python
YOLOv8
Supervision
ByteTrack
OpenCV
NumPy
```

---

## Related Documentation

- [Session README](./README.md)
- [Concepts Overview](./concepts/README.md)
- [Practical README](./practical/README.md)
- [PolygonZone](./concepts/01-PolygonZone.md)
- [LineZone](./concepts/02-LineZone.md)
- [Occupancy vs Flow](./concepts/03-Occupancy-vs-Flow.md)
- [Tracking with Zones](./concepts/04-Tracking-with-Zones.md)
- [Combining PolygonZone and LineZone](./concepts/07-Combining-PolygonZone-and-LineZone.md)

---

## Video

**YouTube:**  
[https://youtu.be/43i0z9b81Z4](https://youtu.be/43i0z9b81Z4)

---

## Author

**Peyman Miyandashti**

- [GitHub](https://github.com/Peyman-mxli)
- [LinkedIn](https://www.linkedin.com/in/peyman-mxli/)
