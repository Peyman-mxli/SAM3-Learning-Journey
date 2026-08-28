# Architecture

```text
Input Image
    ↓
YOLO Object Discovery
    ↓
Bounding-Box Centers → Positive Point Prompts
    ↓
SAM 3 Object Masks → Area and Confidence Analytics
    ↓
Negative Point Refinement for First Object
    ↓
PNG + CSV + JSON Evidence
```

YOLO supplies repeatable coordinates. SAM 3 supplies mask predictions. The analytical layer records how the mask changes when an exclusion point is added.
