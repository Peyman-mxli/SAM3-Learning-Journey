# Validated Output Assets

The scripts generate:

```text
01_basic_positive_point_output.png
02_multiple_positive_points_output.png
03_positive_negative_refinement_output.png
04_prompt_type_comparison_output.png
05_text_to_point_refinement_output.png
06_three_object_point_challenge_output.png
```

All six images were generated successfully by the standalone scripts in Google Colab using a Tesla T4 GPU.

## Verified Results

| Output | Evidence |
|---|---|
| `01_basic_positive_point_output.png` | One mask from point `[413, 494]`; area 27,808 px²; confidence 0.683 |
| `02_multiple_positive_points_output.png` | Side-by-side comparison using one, two, and three positive points |
| `03_positive_negative_refinement_output.png` | Positive point `[413, 494]` and negative point `[353, 414]` |
| `04_prompt_type_comparison_output.png` | Text: 6 people; point: 1 object; YOLO boxes: 4 objects |
| `05_text_to_point_refinement_output.png` | Text found 6 people; point-refined center `[281, 628]` |
| `06_three_object_point_challenge_output.png` | Three YOLO-discovered objects segmented with point prompts |

These are real runtime outputs, not placeholders or notebook screenshots.
