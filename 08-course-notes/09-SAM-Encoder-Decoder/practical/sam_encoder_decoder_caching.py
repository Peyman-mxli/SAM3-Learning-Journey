"""Compare promptable encoder-decoder latency with and without caching.

This is an educational architecture simulation. It does not load an image,
run a trained SAM checkpoint, or claim model-inference performance.
"""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np


RANDOM_SEED = 42
IMAGE_EMBEDDING_SHAPE = (1, 256, 64, 64)
PROMPT_EMBEDDING_SHAPE = (1, 256)
MASK_SHAPE = (500, 500)


class DummyImageEncoder:
    """Simulate an expensive image encoder."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def encode(self, image_reference: str) -> np.ndarray:
        print(f"[Image Encoder] Processing: {image_reference}")
        time.sleep(2.0)
        return self.rng.random(IMAGE_EMBEDDING_SHAPE)


class DummyPromptEncoder:
    """Simulate a lightweight prompt encoder."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def encode(self, prompt: str) -> np.ndarray:
        print(f"[Prompt Encoder] Encoding: {prompt}")
        time.sleep(0.05)
        return self.rng.random(PROMPT_EMBEDDING_SHAPE)


class DummyMaskDecoder:
    """Simulate a lightweight binary-mask decoder."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def decode(
        self,
        image_embedding: np.ndarray,
        prompt_embedding: np.ndarray,
    ) -> np.ndarray:
        _ = image_embedding, prompt_embedding
        print("[Mask Decoder] Generating mask")
        time.sleep(0.05)
        return self.rng.random(MASK_SHAPE) > 0.5


def run_without_cache(
    image_reference: str,
    prompts: list[str],
    image_encoder: DummyImageEncoder,
    prompt_encoder: DummyPromptEncoder,
    mask_decoder: DummyMaskDecoder,
) -> float:
    """Encode the image again for every prompt."""
    start = time.perf_counter()

    for prompt in prompts:
        image_embedding = image_encoder.encode(image_reference)
        prompt_embedding = prompt_encoder.encode(prompt)
        mask_decoder.decode(image_embedding, prompt_embedding)

    return time.perf_counter() - start


def run_with_cache(
    image_reference: str,
    prompts: list[str],
    image_encoder: DummyImageEncoder,
    prompt_encoder: DummyPromptEncoder,
    mask_decoder: DummyMaskDecoder,
) -> float:
    """Encode the image once and reuse its embedding."""
    start = time.perf_counter()
    image_embedding = image_encoder.encode(image_reference)

    for prompt in prompts:
        prompt_embedding = prompt_encoder.encode(prompt)
        mask_decoder.decode(image_embedding, prompt_embedding)

    return time.perf_counter() - start


def save_chart(
    without_cache: float,
    with_cache: float,
    output_path: Path,
) -> None:
    """Save the execution-time comparison chart."""
    labels = ["Sin caching", "Con caching"]
    values = [without_cache, with_cache]

    _, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(labels, values, color=["#d9534f", "#2ca02c"])
    axis.set_ylabel("Tiempo total (segundos)")
    axis.set_title("SAM Encoder-Decoder: impacto del caching")
    axis.set_ylim(0, max(values) * 1.15)

    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.05,
            f"{value:.2f} s",
            ha="center",
            fontweight="bold",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    image_reference = "simulated_static_image"
    prompts = [
        "Point(100, 150)",
        "Point(200, 200)",
        "Point(400, 50)",
    ]

    rng = np.random.default_rng(RANDOM_SEED)
    image_encoder = DummyImageEncoder(rng)
    prompt_encoder = DummyPromptEncoder(rng)
    mask_decoder = DummyMaskDecoder(rng)

    print("\nWITHOUT CACHING")
    without_cache = run_without_cache(
        image_reference,
        prompts,
        image_encoder,
        prompt_encoder,
        mask_decoder,
    )

    print("\nWITH CACHING")
    with_cache = run_with_cache(
        image_reference,
        prompts,
        image_encoder,
        prompt_encoder,
        mask_decoder,
    )

    saved = without_cache - with_cache
    speedup = without_cache / with_cache

    print("\nRESULTS")
    print(f"Without caching: {without_cache:.3f} seconds")
    print(f"With caching:    {with_cache:.3f} seconds")
    print(f"Time saved:      {saved:.3f} seconds")
    print(f"Speedup:         {speedup:.2f}x")

    output_path = (
        Path(__file__).parent
        / "assets"
        / "output"
        / "sam_encoder_decoder_caching_comparison.png"
    )
    save_chart(without_cache, with_cache, output_path)
    print(f"Chart saved: {output_path}")


if __name__ == "__main__":
    main()
