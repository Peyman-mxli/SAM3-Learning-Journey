"""Encode one image once, then answer several interactive prompts quickly."""

from time import perf_counter, sleep

import numpy as np


def encode_image(rng: np.random.Generator) -> np.ndarray:
    sleep(2.0)
    return rng.random((1, 256, 64, 64))


def answer_prompt(
    image_embedding: np.ndarray,
    prompt: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    _ = image_embedding
    start = perf_counter()
    print(f"[Prompt Encoder] {prompt}")
    sleep(0.05)
    print("[Mask Decoder] Generating mask")
    sleep(0.05)
    mask = rng.random((500, 500)) > 0.5
    return mask, perf_counter() - start


def main() -> None:
    prompts = ["Point(100, 150)", "Point(200, 200)", "Point(400, 50)"]
    rng = np.random.default_rng(42)

    start = perf_counter()
    image_embedding = encode_image(rng)
    encoder_time = perf_counter() - start
    print(f"Image encoded once in {encoder_time:.3f} seconds\n")

    for prompt in prompts:
        mask, interaction_time = answer_prompt(image_embedding, prompt, rng)
        print(
            f"{prompt}: {interaction_time:.3f} seconds, "
            f"foreground pixels={int(mask.sum()):,}\n"
        )

    print("Interactive prompt workflow completed.")


if __name__ == "__main__":
    main()
