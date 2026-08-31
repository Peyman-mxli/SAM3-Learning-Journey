# Project 12 — SAM 3 + Muse Glimmer Vision Agent

A validation-first project scaffold for combining **Muse Glimmer** as an agentic reasoning and tool-orchestration model with **SAM 3** as a specialized visual segmentation tool.

> **Current status:** Architecture and executable mock pipeline created. Real Muse Glimmer inference and real SAM 3 inference are not yet validated in this project.

## Objective

Transform a natural-language visual goal into a controlled sequence of validated tool calls:

```text
User Goal
   ↓
Agent Planner
   ↓
Media Inspection
   ↓
SAM 3 Segmentation Tool
   ↓
Deterministic Mask Analysis
   ↓
Structured JSON Result
   ↓
Final Agent Summary
```

## Why Mock First?

A mock-first pipeline validates orchestration, schemas, paths, error handling, and exports without requiring the 30B Muse Glimmer model or SAM 3 checkpoint to be loaded simultaneously.

It does not simulate model quality and must not be treated as model validation.

## Project Structure

```text
12-SAM3-Muse-Glimmer-Agent/
├── README.md
├── config.example.json
├── requirements.txt
├── docs/
│   └── VALIDATION.md
├── src/
│   ├── __init__.py
│   ├── agent.py
│   ├── main.py
│   ├── sam3_adapter.py
│   └── schemas.py
├── tests/
│   └── test_pipeline.py
├── data/
│   ├── input/
│   │   └── README.md
│   └── output/
│       └── README.md
└── results/
    └── README.md
```

## Components

- `agent.py` — bounded orchestration and result verification.
- `sam3_adapter.py` — stable interface with mock and real-backend boundary.
- `schemas.py` — validated dataclasses and serialization.
- `main.py` — command-line entry point.
- `test_pipeline.py` — standard-library unit tests for the mock pipeline.
- `config.example.json` — explicit runtime configuration.
- `docs/VALIDATION.md` — evidence checklist for real execution.

## Run the Mock Pipeline

From this project folder:

```bash
python -m src.main \
  --media data/input/example.jpg \
  --goal "Segment every vehicle and measure its visible area" \
  --prompt vehicle \
  --backend mock \
  --output results/mock-result.json
```

The mock backend validates control flow only. The input image does not need to exist in mock mode.

## Run Tests

```bash
python -m unittest discover -s tests -v
```

## Real Backend Boundary

The `RealSAM3Adapter` intentionally raises `NotImplementedError`. Completing it requires:

1. Selecting the verified SAM 3 runtime used by this repository.
2. Loading the checkpoint without exposing credentials.
3. Converting real predictions to the stable `SegmentationResult` schema.
4. Recording GPU, runtime, package, and checkpoint versions.
5. Testing Glimmer and SAM 3 separately.
6. Connecting the real Muse Glimmer tool-call interface.
7. Saving actual evidence and updating the validation document.

## Success Criteria for the Future Real Run

- Muse Glimmer produces a schema-valid tool request.
- SAM 3 generates at least one inspected mask.
- Mask area is calculated deterministically.
- Detection IDs remain stable across exports.
- Output JSON matches the schema.
- GPU memory and latency are recorded.
- Retry behavior is bounded and observable.
- Generated artifacts are inspected.
- Documentation distinguishes measured results from assumptions.

## Related Documentation

- [Course Module 12](../../08-course-notes/12-Muse-Glimmer-and-SAM3-Agents/)
- [Architecture](../../08-course-notes/12-Muse-Glimmer-and-SAM3-Agents/architecture.md)
- [Hardware Requirements](../../08-course-notes/12-Muse-Glimmer-and-SAM3-Agents/hardware-requirements.md)
- [Integration Workflow](../../08-course-notes/12-Muse-Glimmer-and-SAM3-Agents/sam3-glimmer-workflow.md)
