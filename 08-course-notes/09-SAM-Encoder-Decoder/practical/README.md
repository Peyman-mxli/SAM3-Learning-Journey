# Lesson 09 Practical — Encoder-Decoder Caching

This practical demonstrates why a promptable segmentation system should cache the image embedding when multiple prompts are applied to the same image.

The implementation simulates three SAM components:

- Image Encoder
- Prompt Encoder
- Mask Decoder

It compares two execution strategies:

```text
Without caching
Encode Image → Encode Prompt → Decode Mask
Encode Image → Encode Prompt → Decode Mask
Encode Image → Encode Prompt → Decode Mask
```

```text
With caching
Encode Image Once
        ↓
Reuse Image Embedding for Three Prompts
```

## Run

Install the declared dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the practical:

```bash
python sam_encoder_decoder_caching.py
```

The script requires Python, NumPy, and Matplotlib. It does not require a GPU or SAM checkpoint because it is an architectural latency simulation.

## Outputs

The script prints the measured execution times and creates:

```text
assets/output/sam_encoder_decoder_caching_comparison.png
```

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

Exact times may vary slightly between executions because of runtime scheduling and plotting overhead.

## Files

```text
practical/
├── README.md
├── requirements.txt
├── sam_encoder_decoder_caching.py
└── assets/
    ├── README.md
    └── output/
        ├── README.md
        └── sam_encoder_decoder_caching_comparison.png
```

## Scope and Honesty

This practical measures a controlled latency simulation. It does **not** measure real SAM accuracy, GPU inference speed, image-embedding quality, or mask quality. The generated arrays are deterministic educational stand-ins created with a fixed random seed.
