# Project 09 Results

## Validated Standalone Execution

The complete Project 09 pipeline successfully processed `bus.jpg` in Google Colab using a Tesla T4 GPU and four semantic prompts.

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

## Final Artifact Validation

Original Colab artifacts before repository image optimization:

```text
data/output/bus_filtered_person.png       1,689,776 bytes ✅
data/output/bus_prompt_comparison.png     3,654,198 bytes ✅
results/json/bus_semantic_analysis.json       2,922 bytes ✅
results/csv/bus_detections.csv                  667 bytes ✅
results/csv/bus_prompt_summary.csv               116 bytes ✅
```

The JSON report contains 10 object records. The detection CSV contains 10 rows, and the prompt-summary CSV contains four rows.

## Runtime Notes

Ultralytics automatically adjusted the inference size from `640` to `644` to satisfy the model stride. It also displayed a deprecation warning for `half`; neither warning interrupted execution.

```text
Images processed: 1
Object records:   10
Runtime errors:    0
```

**Status: COMPLETE — standalone pipeline validated successfully.**
