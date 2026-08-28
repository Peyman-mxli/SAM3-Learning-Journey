# Architecture

Promptable segmentation separates expensive image understanding from lightweight interaction:

```text
Image → Image Encoder → Cached Image Embedding
                              ↓
Prompt → Prompt Encoder → Mask Decoder → Mask
```

Without caching, the image encoder runs once per prompt. With caching, it runs once per image and the embedding is reused.

The benchmark evaluates how this design changes total latency as the prompt count grows.
