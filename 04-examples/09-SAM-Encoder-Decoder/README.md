# 09 — SAM Encoder-Decoder

This directory contains four focused Python examples explaining the encoder-decoder architecture behind promptable segmentation and why image-embedding caching enables fast interaction.

The code is an educational simulation. It does not load a SAM checkpoint, process real pixels, or measure production model performance.

## Structure

```text
09-SAM-Encoder-Decoder/
├── 01_encoder_decoder_components.py
├── 02_interactive_prompt_workflow.py
├── 03_caching_comparison.py
├── 04_reusable_caching_benchmark.py
├── README.md
├── requirements.txt
└── assets/
    ├── README.md
    └── output/
        ├── README.md
        └── 03_caching_comparison_output.png
```

No `assets/input/` folder is included because none of these examples reads an image. Adding one would be unnecessary.

## Examples

### 01 — Encoder-Decoder Components

Models the three responsibilities separately: the Image Encoder creates a reusable image embedding, the Prompt Encoder converts a point into a prompt embedding, and the Mask Decoder combines both representations.

### 02 — Interactive Prompt Workflow

Encodes the simulated image once and reuses its embedding for three point prompts. It reports each interaction time and generated mask dimensions.

### 03 — Caching Comparison

Measures the same workflow with and without caching, prints time saved and speedup, and writes a comparison chart.

### 04 — Reusable Caching Benchmark

Provides a small command-line benchmark with configurable prompt count and delays. It validates arguments and returns a structured result.

## Validated Colab Results

```text
Initial image encoding: 2.02 seconds
Prompt 1 interaction:   0.105 seconds
Prompt 2 interaction:   0.105 seconds
Prompt 3 interaction:   0.103 seconds

Without caching:        6.352 seconds
With caching:           2.322 seconds
Time saved:             4.031 seconds
Speedup:                2.74x
```

Exact values vary slightly because `sleep` and runtime scheduling are not perfectly precise.

## Run

```bash
python -m pip install -r requirements.txt
python 01_encoder_decoder_components.py
python 02_interactive_prompt_workflow.py
python 03_caching_comparison.py
python 04_reusable_caching_benchmark.py --prompts 3
```

No GPU, model checkpoint, or external image is required.

## Learning Progression

```text
Component Anatomy
       ↓
Interactive Prompting
       ↓
Caching Comparison
       ↓
Reusable Benchmark
```

## Scope

The example demonstrates architectural control flow and relative caching behavior. It does not evaluate segmentation quality, real GPU latency, image embeddings, or mask accuracy.

**Status: Complete — source validated and Colab evidence preserved.**
