# Output Assets — SAM3 Text Prompts

This directory contains visual evidence extracted from the successfully executed Session 08 notebook.

```text
01_basic_text_prompt_output.png
02_concept_comparison_output.png
04_filter_text_detections_output.png
05_specific_wheel_prompt_output.png
06_reusable_prompt_comparison_output.png
```

## Output Mapping

- `01_basic_text_prompt_output.png` — basic `person` text-prompt segmentation
- `02_concept_comparison_output.png` — comparison of `vehicle`, `bus`, and `person`
- `04_filter_text_detections_output.png` — five reliable person masks after filtering
- `05_specific_wheel_prompt_output.png` — two detected bus wheels
- `06_reusable_prompt_comparison_output.png` — final four-prompt comparison

Example 03 prints six confidence and mask-area measurements and intentionally generates no image.

Validated counts:

```text
vehicle: 1
bus:     1
person:  6
wheel:   2
filtered persons: 5
```

**Status: Five visual outputs preserved.**
