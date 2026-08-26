# Project 09 Results

## Validated Execution

The Session 08 notebook successfully processed `bus.jpg` with four semantic prompts.

| Prompt | Objects detected |
|---|---:|
| vehicle | 1 |
| bus | 1 |
| person | 6 |
| wheel | 2 |

## Person Analysis

| Object | Confidence | Mask area | Reliable |
|---:|---:|---:|:---:|
| 1 | 0.972 | 32,208 px² | Yes |
| 2 | 0.970 | 45,738 px² | Yes |
| 3 | 0.966 | 21,145 px² | Yes |
| 4 | 0.943 | 11,440 px² | Yes |
| 5 | 0.762 | 2,803 px² | Yes |
| 6 | 0.301 | 547 px² | No |

Filtering retained five of six person masks using confidence `>= 0.50` and area `>= 1,000 px²`.

## Preserved Evidence

```text
data/output/bus_filtered_person.png
data/output/bus_prompt_comparison.png
results/json/bus_semantic_analysis.json
results/csv/bus_detections.csv
results/csv/bus_prompt_summary.csv
```

The new standalone pipeline passed syntax validation. A complete standalone Colab rerun remains required before marking the pipeline itself runtime-validated.
