# Output Assets

This folder contains the visually verified outputs from Session 08.

## `sam3_person_text_prompt_filtered.png`

Shows the five reliable `person` masks retained after applying:

```text
confidence >= 0.50
mask area  >= 1,000 px²
```

## `sam3_text_prompts_comparison.png`

Shows the final four-prompt comparison:

```text
vehicle: 1
bus:     1
person:  6
wheel:   2
```

Both outputs were generated and visually inspected in Google Colab.
