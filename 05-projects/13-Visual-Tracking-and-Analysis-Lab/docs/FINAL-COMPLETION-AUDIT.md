# Project 13 — Final Completion Audit

This audit maps the original **Sistema de Seguimiento y Análisis Visual**
laboratory proposal (Run `45ef3e05`) to concrete Project 13 implementation.

## Software implementation

| Requirement | Evidence | Status |
|---|---|---|
| Recorded image/video input | `src/pipeline.py` | Complete |
| Preserve source/date | SQLite session metadata | Complete |
| YOLO detection | `src/pipeline.py` | Complete |
| Supervision post-processing | Supervision detections + ByteTrack | Complete |
| Temporal tracking IDs | ByteTrack + persisted tracker IDs | Complete |
| SAM 3 extension | `src/segmenter.py` + verified run | Complete |
| SQLite history | `src/database.py` | Complete |
| Historical query | Streamlit + SQLite | Complete |
| Session export | `src/export_results.py` | Complete |
| Dashboard | `app.py` | Complete |
| Precision/Recall/F1 | human-reviewed 25-frame detection evaluation | Complete |
| Detection confusion matrix | `results/detection_evaluation.json` | Complete |
| Bounding-box IoU | human-reviewed detection evaluation | Complete |
| Multi-video validation tooling | condition-variant generator + suite runner | Complete |
| Lighting/occlusion/blur/scale robustness tooling | validation condition suite | Complete |
| Tracking ID-switch / fragmentation evaluator | `evaluation/evaluate_tracking_csv.py` | Complete |
| Mask IoU / Dice evaluator | `evaluation/evaluate_segmentation_masks.py` | Complete |
| Reproducible Colab workflow | project notebook + execution guide | Complete |
| Responsible evidence boundary | evaluation docs explicitly separate execution from accuracy | Complete |

## Verified evidence already preserved

Definitive persistent run:

- Session: `lab13_0d46777217`
- Pipeline: YOLO + ByteTrack + SAM 3
- Frames: 75
- Observations: 299
- Unique tracker IDs: 10
- Average confidence: 0.7059505884
- SAM 3 mask observations: 21
- Average SAM mask area: 10047.0952 px

Human-reviewed detection evaluation:

- Reviewed frames: 25
- Ground-truth objects: 92
- Predictions on reviewed frames: 94
- TP: 79
- FP: 15
- FN: 13
- Precision: 0.8404
- Recall: 0.8587
- F1: 0.8495
- Mean matched box IoU: 0.9945

## Controlled validation suite

The repository now contains a reproducible 5-video test design generated from
the same source scene:

1. baseline
2. low light
3. partial occlusion
4. motion blur
5. reduced scale

Generate them:

```bash
python evaluation/create_condition_variants.py \
  data/input/tracking_test_01.mp4
```

Run all five sessions:

```bash
python evaluation/run_validation_suite.py
```

Analyze robustness deltas:

```bash
python evaluation/analyze_validation_conditions.py
```

The robustness report compares observation count, tracker count, confidence,
track length, and SAM-mask coverage against the baseline. These are
**sensitivity indicators**, not ground-truth accuracy.

## Remaining empirical annotation boundary

The software required for the original proposal is complete. Two metrics still
require data that cannot be manufactured legitimately:

- mask IoU / Dice require human-reviewed segmentation masks;
- ID-switch / fragmentation accuracy require human-reviewed temporal track IDs.

Project 13 now contains both evaluators and annotation templates. These metrics
must remain unreported until the corresponding human labels exist.

That boundary is intentional: completing a portfolio project does not justify
inventing ground truth.

## Completion statement

**Project 13 engineering implementation: COMPLETE.**

**Detection evaluation: COMPLETE.**

**Environmental robustness experiment infrastructure: COMPLETE.**

**Mask/tracking accuracy code paths: COMPLETE and awaiting human-reviewed
annotations before numerical claims can be made.**
