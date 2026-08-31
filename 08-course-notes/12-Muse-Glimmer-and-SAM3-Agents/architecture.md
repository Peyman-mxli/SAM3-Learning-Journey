# Architecture — Muse Glimmer with SAM 3

## Purpose

This document defines a proposed architecture for combining Muse Glimmer's agentic reasoning with SAM 3's visual segmentation capabilities.

The central design rule is separation of responsibilities:

```text
Reasoning and Orchestration → Muse Glimmer
Visual Segmentation         → SAM 3
Numeric Computation         → Deterministic Python
Storage and Export          → Application Services
Policy and Permissions      → Agent Runtime
```

## Main Components

### 1. User Interface

Accepts the user's goal, input images or videos, desired output format, and optional constraints.

Examples:

- “Segment every vehicle.”
- “Count people inside this zone.”
- “Track the red car and report its mask area.”
- “Export masks and a CSV summary.”

### 2. Muse Glimmer Agent

Responsible for:

- Understanding the goal
- Decomposing it into steps
- Selecting tools
- Constructing schema-valid arguments
- Inspecting tool responses
- Deciding whether to retry
- Producing a final explanation

It should not fabricate segmentation masks or numeric measurements.

### 3. Tool Registry

Exposes narrowly scoped functions to the agent.

Suggested tools:

| Tool | Responsibility |
|---|---|
| `inspect_media` | Validate image/video type, resolution, frames, and metadata |
| `segment_with_sam3` | Run SAM 3 with text, point, or box prompts |
| `filter_detections` | Apply confidence, class, area, and spatial filters |
| `measure_masks` | Calculate area, coverage, centroid, and bounding boxes |
| `track_video_objects` | Maintain identities across frames |
| `render_annotations` | Create visual overlays |
| `export_results` | Save JSON, CSV, masks, images, or video |

Each tool should have a strict input and output schema.

### 4. SAM 3 Service

The SAM 3 service receives a validated prompt and media reference, performs inference, and returns structured results.

Suggested response contract:

```json
{
  "request_id": "seg-001",
  "image": {
    "width": 1920,
    "height": 1080
  },
  "detections": [
    {
      "id": 0,
      "label": "vehicle",
      "confidence": 0.94,
      "bbox_xyxy": [120, 240, 510, 690],
      "mask_reference": "masks/seg-001-0.png"
    }
  ],
  "warnings": []
}
```

Large masks should be stored as files or compact binary representations rather than inserted directly into the model context.

### 5. Deterministic Analysis Layer

Operations such as these should be calculated in code:

- Pixel area
- Percentage coverage
- Bounding-box dimensions
- Centroids
- Zone occupancy
- Line crossings
- Counts
- Confidence thresholds
- Data validation

The language model may explain these results but should not replace deterministic calculations.

### 6. Artifact and Metadata Store

Stores:

- Original inputs
- Generated masks
- Annotated media
- JSON and CSV results
- Tool-call records
- Runtime configuration
- Model and checkpoint identifiers
- Errors and warnings

## Control Flow

```text
Request
  ↓
Input Validation
  ↓
Agent Planning
  ↓
Permission Check
  ↓
Tool Call
  ↓
Schema Validation
  ↓
SAM 3 or Analysis Execution
  ↓
Result Validation
  ├── Valid → Continue
  └── Invalid → Diagnose → Retry or Escalate
  ↓
Artifact Export
  ↓
Final Response
```

## Failure Recovery

A retry policy should distinguish recoverable from terminal failures.

Recoverable examples:

- No object found with a vague prompt
- Temporary out-of-memory condition with adjustable parameters
- Invalid output path
- Low-confidence result
- Tool timeout

Terminal or escalation examples:

- Unsupported media
- Missing checkpoint
- Insufficient hardware after safe fallback attempts
- Permission denied
- Repeated invalid tool arguments
- Corrupted model weights

Every retry should be bounded. The agent must not loop indefinitely.

## Safety and Security

- Treat images, documents, and tool outputs as untrusted input.
- Do not allow visual content to override system instructions.
- Restrict tools to allowlisted operations and paths.
- Require confirmation for destructive or external actions.
- Validate every tool argument before execution.
- Avoid exposing credentials, local secrets, or private file contents.
- Record model version, tool version, parameters, and errors.
- Use least-privilege access for files, databases, and external services.

## Observability

A professional pipeline should log:

- Request ID
- Timestamp
- Model/checkpoint version
- Quantization and runtime
- Prompt type and prompt value
- Tool name and validated arguments
- Execution time
- GPU memory usage
- Detection count
- Confidence statistics
- Retry count
- Output artifact paths
- Warnings and failures

## Architecture Decision

Muse Glimmer should orchestrate SAM 3 through a stable tool interface rather than importing SAM internals directly into the agent logic. This keeps the vision system independently testable and allows the orchestration model or SAM implementation to be replaced later.
