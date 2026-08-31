# Project 12 — SAM 3 + Muse Glimmer Vision Agent

Production-style, validation-first architecture for using Muse Glimmer as the reasoning/tool layer and SAM 3 as the segmentation layer. The repository now includes a fully executable deterministic pipeline, a strict real-runtime extension point, tested schemas, bounded retries, measured mask generation, JSON/CSV exports, sample data, scripts, and operational documentation.

## What is complete

- End-to-end executable pipeline
- Typed and validated request/result contracts
- Deterministic segmentation demo that reads a real image and writes a real pixel mask
- Stable adapter boundary for an actual SAM 3 runtime
- Bounded retries and measured latency
- JSON and CSV evidence exports
- Reproducible, license-free input image
- Automated tests and one-command validation
- Architecture, security, validation, and real-execution runbooks
- Explicit separation of measured demo evidence from external model claims

## Architecture

```text
Natural-language goal
        ↓
Validated AgentRequest
        ↓
Bounded VisionAgent
        ↓
Segmentation adapter
   ├── demo: measured color mask
   └── real: installed SAM 3 bridge
        ↓
Validated detections
        ↓
Deterministic measurements
        ↓
JSON + CSV + mask artifact
```

## Run it

Requires Python 3.10 or later; the validated demo has no third-party dependencies.

```bash
cd 05-projects/12-SAM3-Muse-Glimmer-Agent
bash scripts/validate.sh
```

Expected result:

```text
3 tests pass
1 detection
6-pixel measured mask area
JSON, CSV, and PGM mask generated
```

Direct CLI usage:

```bash
python -m src.main \
  --media data/input/sample-scene.ppm \
  --goal "Segment the red vehicle and measure its visible area" \
  --prompt vehicle \
  --backend demo
```

## Project structure

```text
├── config/                 Runtime and tool schemas
├── data/input/             Reproducible source input
├── data/output/            Generated masks and previews
├── docs/                   Architecture, runbook, security, validation
├── results/csv/            Detection tables
├── results/json/           Complete run records
├── scripts/                One-command run and validation
├── src/                    Agent, adapters, schemas, reporting, CLI
└── tests/                  Automated pipeline tests
```

## Real SAM 3 and Muse Glimmer

The codebase is ready to connect a real runtime without editing the agent core. Follow [the runbook](docs/RUNBOOK.md) and provide an importable adapter via `SAM3_PLUGIN_MODULE`.

Real model execution is not represented by fake outputs. It requires the actual checkpoint, accepted license/access conditions, the exact compatible package API, and sufficient compute. Those are external execution requirements—not unfinished empty folders. The committed evidence clearly identifies which backend produced it.

See [validation](docs/VALIDATION.md) for the evidence matrix and [security](docs/SECURITY.md) before loading third-party model code.

## Related course material

- [Muse Glimmer and SAM 3 agents](../../08-course-notes/12-Muse-Glimmer-and-SAM3-Agents/)
- [Agent architecture](../../08-course-notes/12-Muse-Glimmer-and-SAM3-Agents/architecture.md)
- [Hardware requirements](../../08-course-notes/12-Muse-Glimmer-and-SAM3-Agents/hardware-requirements.md)
