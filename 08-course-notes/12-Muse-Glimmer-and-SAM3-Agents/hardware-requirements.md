# Hardware Requirements

## Why Hardware Planning Matters

Muse Glimmer contains approximately 30 billion parameters. “Runs locally” and “runs on a single GPU” do not mean that it will run on every laptop or entry-level GPU.

Model weights, the KV cache, the perception encoder, temporary tensors, the agent runtime, and SAM 3 must all fit within available memory.

## Memory Overview

According to Meta's release information:

- Full-precision operation requires more than 55 GB of memory.
- Quantized weights reduce the language-model footprint to under 20 GB.
- Meta targets 24 GB and 32 GB memory envelopes for practical local deployment.
- Additional memory is needed for context, image perception, speculative decoding, and runtime overhead.

Exact requirements depend on:

- Quantization format
- Context length
- Image size and count
- Runtime implementation
- KV-cache precision
- Batch size
- Whether SAM 3 shares the same GPU
- Operating-system and display memory usage

## Practical Hardware Tiers

| Hardware | Expected suitability |
|---|---|
| 8–12 GB GPU | Generally insufficient for the complete Glimmer model |
| 16 GB GPU / Colab T4 | Likely insufficient for the standard full local stack; experiments may require offloading or aggressive optimization |
| 24 GB GPU | Target range for optimized 4-bit deployment, with careful memory management |
| 32 GB unified/GPU memory | More practical room for model, context, and perception components |
| More than 55 GB memory | Appropriate for less-compressed variants, depending on runtime |
| CPU-only | Technically runtime-dependent but expected to be slow for interactive agent workflows |

These are planning guidelines, not benchmark guarantees.

## Running Muse Glimmer and SAM 3 Together

The most important architectural question is whether both models must reside on the same GPU.

### Option A — Sequential GPU Loading

Load one model, run inference, release memory, and then load the other.

Advantages:

- Lower peak GPU memory

Trade-offs:

- High latency
- Model reload overhead
- More complicated resource management

### Option B — CPU/GPU Offloading

Place selected layers or models in system RAM.

Advantages:

- May enable constrained hardware experiments

Trade-offs:

- Slower inference
- Higher RAM requirements
- Runtime-specific configuration
- Potential instability or unsupported combinations

### Option C — Separate Processes or Devices

Run Glimmer and SAM 3 on separate GPUs or machines.

Advantages:

- Clear isolation
- Better concurrency
- Easier performance profiling

Trade-offs:

- More infrastructure
- Network or IPC design
- Increased operational complexity

### Option D — Remote SAM 3 Tool

Keep Glimmer local while exposing SAM 3 through an internal service.

Advantages:

- Local reasoning and data-policy control
- Independent scaling of the vision service

Trade-offs:

- Media may leave the local machine
- Requires authentication and transport security
- No longer fully offline

## Colab T4 Assessment

A typical Colab T4 provides 16 GB of VRAM. It should not be presented as a validated target for running the complete optimized Muse Glimmer stack together with SAM 3.

The correct workflow is:

1. Check the exact published artifact and runtime.
2. Measure free VRAM.
3. Start with the smallest supported quantization.
4. Use a short context and one image.
5. Monitor peak memory.
6. Test Glimmer alone.
7. Test SAM 3 alone.
8. Only then attempt orchestration.
9. Record measured results rather than assumptions.

## Validation Checklist

```text
[ ] GPU model recorded
[ ] GPU memory recorded
[ ] System RAM recorded
[ ] Runtime and version recorded
[ ] Model artifact recorded
[ ] Quantization recorded
[ ] Context length recorded
[ ] Image size recorded
[ ] Glimmer-only peak memory measured
[ ] SAM-only peak memory measured
[ ] Combined workflow peak memory measured
[ ] Generation speed measured
[ ] Tool-call latency measured
[ ] Failure behavior documented
```
