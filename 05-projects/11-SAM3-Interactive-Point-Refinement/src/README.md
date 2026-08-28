# Source

`point_refinement_pipeline.py` discovers objects with YOLO, segments their center points with SAM 3, refines the first mask with a negative point, and exports visual and analytical evidence.

```bash
python src/point_refinement_pipeline.py
```
