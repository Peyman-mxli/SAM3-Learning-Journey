# Project 06 — MVP Completion Matrix

This document maps the official **Sistema de Seguimiento y Análisis Visual**
proposal to concrete implementation evidence in this repository.

## Guiding question

> How do variations in lighting and occlusion affect the accuracy of object
> tracking in recorded videos?

The project now includes a dedicated, reproducible evaluation module for this
question:

- `evaluation/environmental_condition_analysis.py`
- `evaluation/environmental_conditions_template.csv`

The evaluator intentionally requires real condition-labeled observations. It
does not fabricate lighting or occlusion results when the original runs did
not record those labels.

## MVP requirements

| Proposal requirement | Implementation evidence | Status |
|---|---|---|
| Load recorded image/video and preserve source/date | `src/process_video_session.py`, `data/session_history.csv` | Complete |
| Detect entities | YOLO integration in `src/detector.py` and video pipeline | Complete |
| Track entities through frames | ByteTrack integration in `src/tracker.py` | Complete |
| Maintain temporal identity | Persistent tracker IDs and tracker summaries | Complete |
| Optional segmentation extension | SAM 3 integration in `src/segmenter.py` | Complete |
| Store ID, timestamp/frame, medium, result, confidence, notes | SQLite/data persistence layer and exported observation CSVs | Complete |
| Query historical observations | `src/database.py`, session registry and analytics | Complete |
| Compare sessions | `analytics/session_comparison.py` and `reports/session_comparison_summary.csv` | Complete |
| Show evolution/evidence in a digital application/dashboard | `app.py` Streamlit dashboard | Complete |
| Export structured evidence | CSV reports in `reports/` and `evaluation/` | Complete |
| Precision / Recall | Ground-truth evaluator and `evaluation/evaluation_summary.json` | Complete |
| IoU / Dice | `src/metrics.py` and ground-truth evaluation | Complete |
| Confusion matrix evidence | `evaluation/evaluation_summary.json` | Complete |
| Document failures and operating limits | `docs/LIMITATIONS.md`, `docs/RESULTS.md` | Complete |
| Evaluate lighting and occlusion explicitly | Dedicated environmental-condition evaluator | Implementation complete; real condition-labeled observations required |
| 20–50 images or 2–5 short videos for technical validation | 20-image ground-truth evaluation plus multiple recorded-video sessions | Complete |
| 100–200 varied observations | Existing video sessions exceed the requested observation count | Complete |
| Exportable performance report | `reports/performance_summary.csv` and analytics modules | Complete |
| Reproducible portfolio documentation | Root README, reports, evaluation docs, Colab workflow | Complete |

## Existing verified evidence

The repository already preserves two verified recorded-video sessions:

- `session_001`: 75 frames, 246 observations, 6 tracker IDs.
- `session_002`: 75 frames, 720 observations, 52 tracker IDs.

The ground-truth evaluation currently preserves:

- 20 evaluated images
- 424 ground-truth instances
- 472 predicted instances
- 381 true positives
- 91 false positives
- 43 false negatives
- Precision: 0.8072
- Recall: 0.8986
- Average IoU: 0.7969
- Average Dice: 0.8829

These are measured project results, not estimates.

## Environmental-condition protocol

To answer the guiding question with evidence rather than assumptions:

1. Copy `evaluation/environmental_conditions_template.csv` to
   `evaluation/environmental_conditions.csv`.
2. Use the same model/configuration across condition tests.
3. Record each real test case with:
   - lighting condition,
   - occlusion condition,
   - TP / FP / FN,
   - ID switches,
   - tracked-ID opportunities,
   - average confidence.
4. Run:

```bash
python evaluation/environmental_condition_analysis.py
```

5. The module generates:

```text
reports/environmental_condition_results.csv
reports/lighting_condition_summary.csv
reports/occlusion_condition_summary.csv
reports/environmental_condition_chart.png
```

This closes the implementation gap in the official proposal while preserving
scientific integrity: the repository is ready to calculate and compare the
effect of lighting and occlusion as soon as real labeled condition trials are
recorded.

## Definition of done

The software MVP is complete when the pipeline can process recorded media,
preserve historical tracking evidence, export metrics, compare sessions, and
present results through the dashboard.

The **empirical answer** to the lighting/occlusion research question is a
separate validation artifact and must be based on measured, condition-labeled
data. The code required to produce that answer is now included.
