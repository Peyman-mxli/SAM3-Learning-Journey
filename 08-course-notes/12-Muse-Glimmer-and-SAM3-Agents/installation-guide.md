# Installation Guide

## Scope

This guide defines a safe, validation-first installation strategy for Muse Glimmer. Runtime support around a newly released model can change quickly, so exact commands should be taken from the current official model documentation for the chosen backend.

This repository does not yet claim a validated local installation.

## Supported Deployment Paths Announced by Meta

Meta identifies several intended deployment ecosystems, including:

- Hugging Face Transformers
- llama.cpp
- MLX
- ExecuTorch
- vLLM
- SGLang
- Ollama
- LM Studio
- Unsloth
- Hosted partners for users who do not need a fully local deployment

Availability and exact support may differ by model artifact, quantization, operating system, and release date.

## Step 1 — Define the Target

Before installing, record:

```text
Operating system:
CPU:
GPU:
GPU/unified memory:
System RAM:
Available disk:
Target runtime:
Target quantization:
Local-only requirement:
SAM 3 execution location:
```

## Step 2 — Review the License and Model Card

Read:

1. The Meta announcement
2. The official developer page
3. The Hugging Face model card
4. The Apache 2.0 license
5. Safety, intended-use, and limitation sections
6. Runtime-specific documentation

Do not download a large checkpoint before confirming that the selected artifact fits the target hardware.

## Step 3 — Create an Isolated Environment

Example:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the backend only after selecting it from the current official instructions.

## Step 4 — Authenticate Only When Required

Some model hosts may require accepting terms or authenticating before downloading files.

Security rules:

- Never commit access tokens.
- Store tokens in environment variables or an approved credential manager.
- Do not paste secrets into notebooks saved in GitHub.
- Restrict token scope.
- Rotate exposed credentials immediately.

## Step 5 — Download the Correct Artifact

Select:

- Full or quantized weights
- Backend-compatible format
- Vision/perception components
- Tokenizer and configuration
- Optional speculative-decoding drafter
- Checksums when published

Avoid mixing artifacts from incompatible releases.

## Step 6 — Validate Glimmer Independently

Begin with:

- One short text prompt
- One small image
- Short context
- Batch size 1
- No SAM 3 integration
- GPU memory monitoring enabled

Validate:

```text
[ ] Model loads
[ ] Text generation works
[ ] Image input works
[ ] Tool-call format is parseable
[ ] Peak memory is recorded
[ ] Latency is recorded
[ ] Output is reproducible enough for evaluation
```

## Step 7 — Validate SAM 3 Independently

Use the existing SAM 3 environment and confirm:

- Checkpoint loads
- Image inference works
- Prompt type works
- Masks are returned
- Result schema is stable
- Memory and latency are measured

## Step 8 — Add a Tool Adapter

Wrap SAM 3 behind a deterministic function.

Conceptual interface:

```python
def segment_with_sam3(
    media_path: str,
    prompt_type: str,
    prompt: dict,
    confidence: float = 0.25,
) -> dict:
    """Return schema-validated segmentation metadata."""
```

The adapter should validate paths, prompt type, thresholds, empty results, and output shape.

## Step 9 — Connect the Agent Runtime

Expose the adapter through the chosen agent framework or function-calling interface.

The model should receive:

- Tool name
- Tool description
- Strict argument schema
- Structured result schema
- Clear failure messages
- Bounded retry policy

## Step 10 — Test the Smallest End-to-End Case

Recommended first test:

```text
Input: one image
Goal: segment one clearly visible object class
Tool calls: one
Output: one annotated image + JSON summary
```

Only after this works should the experiment expand to video, tracking, multiple tools, or autonomous retries.

## Common Failure Categories

| Failure | Likely cause | Response |
|---|---|---|
| Out of memory | Model/context/image too large | Reduce context or image size, choose supported quantization, or change hardware |
| Model does not load | Incompatible artifact/runtime | Verify format, version, and official instructions |
| Image rejected | Incorrect preprocessing or unsupported interface | Use backend-specific processor |
| Invalid tool arguments | Weak or ambiguous schema | Tighten schema and validate before execution |
| Empty SAM result | Prompt mismatch or threshold too high | Refine prompt or use another prompt type |
| Slow generation | Hardware or backend limitation | Profile runtime, quantization, cache, and speculative decoding |
| Repeated retries | Missing termination condition | Enforce retry and time limits |

## Documentation Rule

When this repository performs the first real installation, record exact commands, versions, hardware, observed VRAM, runtime, outputs, and errors. Until then, this file remains a deployment plan rather than a tested installation transcript.
