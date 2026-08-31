# Project Architecture

```text
Muse Glimmer Server
OpenAI-compatible HTTP
        ↓
GlimmerClient
        ↓
One Tool Call per Turn
        ↓
segment_with_sam3
        ↓
RealSAM3Adapter
        ↓
SAM3SemanticPredictor
        ↓
Masks + Boxes + Confidence
        ↓
Deterministic Area Calculation
        ↓
JSON + CSV + Annotated Image
        ↓
Muse Glimmer Final Summary
```

## Design Principles

1. Muse Glimmer chooses and sequences tools.
2. SAM 3 generates pixel-level masks.
3. NumPy calculates mask measurements.
4. Schemas validate every result.
5. The runtime limits retries.
6. Mock outputs are labeled and isolated.
7. Real outputs include runtime metadata.
8. The agent cannot delete or overwrite arbitrary files.

## Backends

- `mock`: tests contracts and exports without models.
- `sam3`: runs the Ultralytics semantic-predictor pattern validated in Project 09.
- `glimmer-sam3`: asks an OpenAI-compatible Muse Glimmer endpoint to call SAM 3, then returns a grounded final summary.
