# Generated Outputs

The practical script writes generated artifacts to this directory.

| File | Description |
|---|---|
| `vehicles_sam.mp4` | Full YOLO + ByteTrack + SAM 3 visualization |
| `vehicles_sam_zone.mp4` | Segmentation restricted to the polygon zone |
| `vehicles_sam_opacity.mp4` | Confidence-controlled mask opacity |
| `vehicles_sam_areas.mp4` | Mask visualization used during area analysis |
| `vehicles_text_prompts.mp4` | Direct semantic video segmentation |
| `mask_area_chart.png` | Temporal mask-area chart |
| `mask_areas.json` | Frame-level mask-area observations by tracker ID |

Generated files should be documented only after their execution and visual verification.


---

## Football-Pitch Homography Visuals

| File | Description |
|---|---|
| [`football_pitch_four_points.svg`](./football_pitch_four_points.svg) | Perspective pitch with four ordered source points |
| [`football_pitch_top_down.svg`](./football_pitch_top_down.svg) | Normalized top-down homography result |
| [`04_detected_players.svg`](./04_detected_players.svg) | Three detected robots with bounding boxes |
| [`05_player_anchor_points.svg`](./05_player_anchor_points.svg) | Bottom-center ground-contact points |
| [`06_football_minimap.svg`](./06_football_minimap.svg) | FIFA-style minimap using transformed coordinates |
| [`homography_matrix.json`](./homography_matrix.json) | Matrix H, source points, anchors, and field points |
| [`transformed_coordinates.csv`](./transformed_coordinates.csv) | Original image coordinates and transformed field coordinates |
| `01_original_field.png` | Runtime source image |
| `02_four_selected_points.png` | Runtime four-point visualization |
| `03_top_down_field.png` | Runtime warped field |
| `04_detected_players.png` | Runtime YOLO/demonstration detections |
| `05_player_anchor_points.png` | Runtime foot-anchor visualization |
| `06_football_minimap.png` | Runtime minimap |
