"""Compare simulated encoder-decoder latency with and without caching."""

from pathlib import Path
from time import perf_counter, sleep

import matplotlib.pyplot as plt


OUTPUT_PATH = Path(__file__).parent / "assets/output/03_caching_comparison_output.png"
PROMPTS = ["Point(100, 150)", "Point(200, 200)", "Point(400, 50)"]


def image_encoder() -> None:
    sleep(2.0)


def prompt_encoder_and_mask_decoder(prompt: str) -> None:
    print(f"Processing {prompt}")
    sleep(0.10)


def without_cache() -> float:
    start = perf_counter()
    for prompt in PROMPTS:
        image_encoder()
        prompt_encoder_and_mask_decoder(prompt)
    return perf_counter() - start


def with_cache() -> float:
    start = perf_counter()
    image_encoder()
    for prompt in PROMPTS:
        prompt_encoder_and_mask_decoder(prompt)
    return perf_counter() - start


def save_chart(no_cache: float, cached: float) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    labels = ["Without caching", "With caching"]
    values = [no_cache, cached]
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(labels, values, color=["#d9534f", "#2ca02c"])
    axis.set_ylabel("Total time (seconds)")
    axis.set_title("SAM Encoder-Decoder: caching impact (simulation)")
    axis.set_ylim(0, max(values) * 1.15)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.05, f"{value:.2f} s", ha="center")
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    no_cache = without_cache()
    cached = with_cache()
    saved = no_cache - cached
    speedup = no_cache / cached
    save_chart(no_cache, cached)

    print("\nCACHING RESULTS")
    print(f"Without caching: {no_cache:.3f} seconds")
    print(f"With caching:    {cached:.3f} seconds")
    print(f"Time saved:      {saved:.3f} seconds")
    print(f"Speedup:         {speedup:.2f}x")
    print(f"Output:          {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
