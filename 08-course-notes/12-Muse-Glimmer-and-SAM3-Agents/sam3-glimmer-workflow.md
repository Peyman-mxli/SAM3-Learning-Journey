# Proposed SAM 3 + Muse Glimmer Workflow

## Goal

Design a local multimodal agent that can turn a natural-language visual task into validated SAM 3 operations and structured outputs.

## Example Task

> Find all vehicles in the image, segment them, calculate their visible pixel area, annotate the image, and export a CSV summary.

## Expected Workflow

```text
Natural-Language Goal
        ↓
Muse Glimmer Planning
        ↓
inspect_media
        ↓
segment_with_sam3
        ↓
validate_segmentation
        ↓
measure_masks
        ↓
render_annotations
        ↓
export_results
        ↓
Muse Glimmer Summary
```

## Tool Contracts

### inspect_media

Input:

```json
{
  "media_path": "assets/input/traffic.jpg"
}
```

Output:

```json
{
  "type": "image",
  "width": 1920,
  "height": 1080,
  "channels": 3,
  "valid": true
}
```

### segment_with_sam3

Input:

```json
{
  "media_path": "assets/input/traffic.jpg",
  "prompt_type": "text",
  "prompt": "vehicle",
  "confidence": 0.25
}
```

Output:

```json
{
  "request_id": "seg-001",
  "detections": [
    {
      "detection_id": 0,
      "label": "vehicle",
      "confidence": 0.94,
      "bbox_xyxy": [120, 240, 510, 690],
      "mask_path": "assets/output/masks/vehicle-000.png"
    }
  ],
  "warnings": []
}
```

### measure_masks

Input:

```json
{
  "request_id": "seg-001",
  "measurements": ["pixel_area", "image_coverage", "centroid"]
}
```

Output:

```json
{
  "measurements": [
    {
      "detection_id": 0,
      "pixel_area": 84210,
      "image_coverage": 0.0406,
      "centroid_xy": [315.2, 471.8]
    }
  ]
}
```

## Agent Logic

Conceptual pseudocode:

```python
goal = receive_user_goal()

plan = glimmer.plan(
    goal=goal,
    available_tools=tool_schemas,
)

media = inspect_media(plan.media_path)
assert media["valid"]

segmentation = segment_with_sam3(
    media_path=plan.media_path,
    prompt_type=plan.prompt_type,
    prompt=plan.prompt,
    confidence=plan.confidence,
)

if not segmentation["detections"]:
    revised = glimmer.refine_prompt(
        original_goal=goal,
        tool_result=segmentation,
    )
    segmentation = segment_with_sam3(**revised)

validated = validate_segmentation(segmentation)

measurements = measure_masks(
    request_id=validated["request_id"],
    measurements=["pixel_area", "image_coverage", "centroid"],
)

rendered = render_annotations(
    media_path=plan.media_path,
    detections=validated["detections"],
)

artifacts = export_results(
    detections=validated["detections"],
    measurements=measurements,
    rendered_media=rendered,
    formats=["json", "csv", "png"],
)

final_response = glimmer.summarize(
    goal=goal,
    results=measurements,
    artifacts=artifacts,
)
```

This is architectural pseudocode, not a tested implementation.

## Prompt Strategy

The agent should choose among prompt types based on the task.

| Situation | Preferred prompt |
|---|---|
| Semantic category known | Text prompt |
| User selects a visible location | Positive/negative point prompts |
| Detector already found an object | Bounding-box prompt |
| Mask needs correction | Additional positive/negative points |
| Video object must persist | Video segmentation/tracking workflow |

## Validation Rules

Before accepting results:

- Detection count must be non-negative and internally consistent.
- Every bounding box must fall within image dimensions.
- Every mask must match the media dimensions or include a documented transform.
- Confidence must remain in the expected numeric range.
- Mask files must exist and be readable.
- Pixel-area calculations must come from code.
- Exported rows must map to stable detection IDs.
- Empty results must be explicitly reported.
- Retried prompts and parameters must be recorded.

## Retry Policy

Example bounded policy:

```text
Attempt 1: Original semantic prompt
Attempt 2: More specific semantic prompt
Attempt 3: Alternate prompt type or request user guidance
Stop: Return partial results and diagnostic information
```

The agent should never silently change the user's goal.

## First Practical Milestone

The first implementation should remain deliberately small:

```text
One image
One text prompt
One SAM 3 call
One measurement function
One annotated PNG
One JSON result
No autonomous file deletion
No external side effects
```

Success criteria:

- Schema-valid tool call
- SAM 3 produces at least one valid mask
- Mask measurement is verified independently
- Output paths exist
- Final explanation matches the structured result
- Hardware and runtime metrics are recorded

## Future Extension

After the image prototype is validated:

```text
Static Image Agent
      ↓
Multiple Objects
      ↓
Prompt Refinement
      ↓
Zones and Counting
      ↓
Video Segmentation
      ↓
Tracking
      ↓
Temporal Analytics
      ↓
Complete Vision Agent Project
```

The future practical project can then be created under:

```text
05-projects/
└── 12-SAM3-Muse-Glimmer-Agent/
```
