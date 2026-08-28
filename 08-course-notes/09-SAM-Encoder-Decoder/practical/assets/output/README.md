# Output Assets

## Validated Output

```text
sam_encoder_decoder_caching_comparison.png
```

The chart compares the total time required to process three prompts when the image embedding is recalculated for every prompt versus calculated once and reused.

Validated Google Colab measurements:

| Strategy | Total time |
|---|---:|
| Without caching | 6.352 seconds |
| With caching | 2.322 seconds |

```text
Time saved: 4.031 seconds
Speedup:    2.74x
```

The PNG in this directory is the actual output downloaded from the validated Colab execution.

