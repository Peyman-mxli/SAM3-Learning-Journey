# Session 06 — Zones and Counting

This session introduces **spatial zones and object counting** using computer vision, object detection, and object tracking.

The main objective is to understand how detected and tracked objects can be analyzed based on their position inside specific regions of a video.

Two important Supervision components are introduced:

- `PolygonZone`
- `LineZone`

These tools make it possible to answer two different questions:

> How many objects are inside a specific area right now?

and

> How many objects have crossed a specific line over time?

---

## Session Information

**Course:** SAM3 — Computer Vision with Segment Anything Model 3  
**Session:** 06  
**Module:** 02  
**Notebook:** `02_c_zonas_conteo.ipynb`  
**Topic:** Zones and Counting  
**Estimated notebook time:** 30 minutes  
**Estimated class duration:** 1 hour

---

## Learning Objectives

By the end of this session, I learned how to:

- Define polygonal regions inside an image or video
- Detect whether tracked objects are inside a region
- Count objects currently present inside a zone
- Create virtual counting lines
- Count objects crossing a line
- Understand the difference between occupancy and accumulated flow
- Combine object detection with object tracking
- Combine tracking with `PolygonZone`
- Combine tracking with `LineZone`
- Visualize zones and counters directly on video frames
- Use multiple counting mechanisms in the same video pipeline

---

# 1. Why Use Zones?

Object detection tells us:

> What objects are visible?

Object tracking tells us:

> Which object is which across multiple frames?

Zones add another layer of information:

> Where are those objects located?

This allows a computer vision system to understand activity inside specific areas.

For example, a traffic monitoring system could define different regions for:

- Left lane
- Right lane
- Parking area
- Intersection
- Pedestrian crossing
- Building entrance
- Restricted area

The system can then analyze only the objects interacting with those regions.

---

# 2. Two Types of Zones

In this session, two main types of zones are used:

1. `PolygonZone`
2. `LineZone`

Although both are used for spatial analysis, they answer different questions.

---

## PolygonZone

A `PolygonZone` represents an area of the image.

It answers:

> How many objects are inside this area right now?

The count can change from frame to frame.

Conceptually:

```text
PolygonZone → current presence / occupancy
