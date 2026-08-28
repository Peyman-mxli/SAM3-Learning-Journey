# 09 — SAM Encoder-Decoder Architecture

This session explains the **promptable encoder-decoder architecture behind Segment Anything Model (SAM)** and why image embeddings can be cached to support fast interactive segmentation.

The lesson uses a lightweight Python simulation to isolate the computational roles of the image encoder, prompt encoder, and mask decoder without requiring a SAM checkpoint.

---

# Session Objective

The objective of this lesson is to understand how SAM separates expensive image processing from lightweight prompt interaction.

The session explores:

- The promptable segmentation paradigm
- Image, prompt, and mask embeddings
- The three principal SAM components
- Image-embedding caching
- Interactive segmentation latency
- Reuse of one image embedding across multiple prompts
- Why video requires a different temporal strategy
- The conceptual evolution from SAM to SAM 2.1 and SAM 3

---

# Topics Covered

- Foundation models for segmentation
- Promptable segmentation
- Point prompts
- Bounding-box prompts
- Text prompts
- Mask prompts
- Vision Transformer image encoders
- Prompt encoders
- Mask decoders
- Image embeddings
- Prompt embeddings
- Caching and reuse
- Interactive latency
- Frame-by-frame video processing
- Spatiotemporal feature reuse

---

# Session Structure

```text
09-SAM-Encoder-Decoder/
├── README.md
├── CLASS-RECORDING.md
└── 04_b_sam_encoder_decoder.ipynb
```

The practical implementation and its input/output evidence will be added after the notebook is executed, extended, and validated.

---

# Original Class Notebook

The original class notebook is preserved as:

[04_b_sam_encoder_decoder.ipynb](./04_b_sam_encoder_decoder.ipynb)

The notebook contains the original Spanish lesson content and a computational simulation of encoder-decoder latency.

---

# The Promptable Segmentation Paradigm

Traditional task-specific segmentation systems normally require a labeled dataset and training for a fixed category. SAM changes the interaction model: a user provides a prompt that identifies what should be segmented.

Supported prompt concepts include:

```text
Point
Bounding Box
Text
Previous Mask
```

The model combines the image representation with the prompt representation to generate a segmentation mask.

---

# SAM Architecture

SAM is organized into three principal components.

## 1. Image Encoder

The image encoder processes the complete image and produces a high-dimensional image embedding.

```text
Image
  ↓
Image Encoder
  ↓
Image Embedding
```

It is the computationally expensive component. The central optimization is to run it once per image and cache its output.

## 2. Prompt Encoder

The prompt encoder converts a point, box, text instruction, or mask into a mathematical prompt embedding.

```text
User Prompt
    ↓
Prompt Encoder
    ↓
Prompt Embedding
```

It is designed to be lightweight so the user can provide new prompts interactively.

## 3. Mask Decoder

The mask decoder combines the cached image embedding with the new prompt embedding and predicts the final mask.

```text
Image Embedding + Prompt Embedding
                  ↓
             Mask Decoder
                  ↓
          Segmentation Mask
```

---

# Complete Inference Flow

```text
Image ──→ Image Encoder ──→ Cached Image Embedding
                                      │
Prompt ─→ Prompt Encoder ─→ Prompt Embedding
                                      │
                                      ↓
                                Mask Decoder
                                      ↓
                             Segmentation Mask
```

The expensive image representation remains fixed while the lightweight prompt path can be executed repeatedly.

---

# Notebook Simulation

The notebook defines three educational components:

```python
DummyImageEncoder
DummyPromptEncoder
DummyMaskDecoder
```

They simulate the relative cost of each architectural stage.

| Component | Simulated latency | Output |
|---|---:|---|
| Image Encoder | 2.00 seconds | `(1, 256, 64, 64)` image embedding |
| Prompt Encoder | 0.05 seconds | `(1, 256)` prompt embedding |
| Mask Decoder | 0.05 seconds | `(500, 500)` binary mask |

These arrays are randomly generated educational stand-ins. They are not predictions from a trained SAM model.

---

# Interactive Latency Demonstration

The notebook encodes the image once:

```python
image_embedding = image_encoder.encode(image)
```

It then reuses that embedding for three prompts:

```text
Point(100, 150)
Point(200, 200)
Point(400, 50)
```

For every interaction, only the lightweight components run:

```python
prompt_embedding = prompt_encoder.encode(prompt)
mask = mask_decoder.decode(
    image_embedding,
    prompt_embedding
)
```

Expected conceptual timing:

```text
Initial image encoding: approximately 2.00 s

Each prompt:
Prompt Encoder: 0.05 s
Mask Decoder:   0.05 s
Total:          approximately 0.10 s
```

The exact printed values may vary slightly because operating-system scheduling and notebook overhead affect `time.time()` measurements.

---

# Why Caching Matters

Without caching, every user click would repeat the expensive image encoder.

For three prompts:

```text
Without caching:
3 × (2.00 + 0.05 + 0.05) ≈ 6.30 s

With caching:
2.00 + 3 × (0.05 + 0.05) ≈ 2.30 s
```

In this simulation, caching avoids approximately four seconds of redundant encoder work.

---

# Image Segmentation versus Video

Caching is straightforward for a static image because its embedding remains valid while the pixels do not change.

Video introduces a different challenge:

```text
Frame 1 → Image changes
Frame 2 → Image changes
Frame 3 → Image changes
```

A basic frame-independent approach must encode every frame. Modern video-aware SAM workflows use memory and spatiotemporal information to propagate object representations and reduce redundant computation while maintaining temporal consistency.

The notebook presents this as the conceptual motivation for SAM 2.1 and SAM 3 video capabilities.

---

# Required Libraries

The lesson simulation only requires:

```python
import time
import numpy as np
```

No GPU, model checkpoint, image file, Ultralytics installation, or network download is required for the original six-cell notebook.

---

# Validation Requirements

This lesson will be considered complete after verifying:

- Notebook opens successfully
- NumPy imports successfully
- Dummy encoder-decoder classes initialize
- Image embedding is calculated once
- Three prompts reuse the cached embedding
- One binary mask is returned per prompt
- Image-encoder and interaction times are printed
- The latency comparison is analyzed
- A practical extension is implemented
- Practical outputs are saved and visually inspected
- Session documentation is finalized

---

# Expected Learning Outcomes

After completing this lesson, the learner should be able to:

- Explain promptable segmentation
- Identify the three principal SAM components
- Distinguish image and prompt embeddings
- Explain why the image encoder is expensive
- Explain why prompt interaction can be fast
- Reuse a cached image embedding
- Interpret an encoder-decoder latency simulation
- Estimate the benefit of caching
- Explain why video requires spatiotemporal memory
- Connect this architecture to interactive segmentation applications

---

# Relationship to Previous Sessions

```text
Session 07
Advanced Mask Visualization
          ↓
Session 08
Natural-Language Text Prompts
          ↓
Session 09
How SAM Encodes Images and Decodes Prompts
```

Session 09 asks:

```text
Why can SAM respond quickly to many prompts
after an image has been encoded once?
```

---

# Technologies

```text
Python
Google Colab
NumPy
SAM Architecture
Encoder-Decoder Models
Image Embeddings
Prompt Embeddings
Interactive Segmentation
```

---

# Current Progress

Completed:

- Lesson 09 folder design
- Original class notebook reviewed
- Lesson README
- Class-recording documentation
- Original notebook preserved

Pending:

- Notebook execution and timing validation
- Practical extension
- Input and output asset documentation
- Visual output evidence
- Course index update after all classes
