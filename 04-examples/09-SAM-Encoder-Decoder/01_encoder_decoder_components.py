"""Demonstrate the three conceptual components of promptable SAM systems."""

import time

import numpy as np


SEED = 42


class ImageEncoder:
    """Educational stand-in for SAM's expensive image encoder."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def encode(self, image_reference: str) -> np.ndarray:
        print(f"[Image Encoder] Processing {image_reference}")
        time.sleep(0.20)
        return self.rng.random((1, 256, 64, 64))


class PromptEncoder:
    """Educational stand-in for SAM's lightweight prompt encoder."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def encode(self, prompt: str) -> np.ndarray:
        print(f"[Prompt Encoder] Encoding {prompt}")
        time.sleep(0.01)
        return self.rng.random((1, 256))


class MaskDecoder:
    """Educational stand-in for SAM's mask decoder."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def decode(self, image_embedding: np.ndarray, prompt_embedding: np.ndarray) -> np.ndarray:
        _ = image_embedding, prompt_embedding
        print("[Mask Decoder] Generating a binary mask")
        time.sleep(0.01)
        return self.rng.random((500, 500)) > 0.5


def main() -> None:
    rng = np.random.default_rng(SEED)
    image_embedding = ImageEncoder(rng).encode("simulated_static_image")
    prompt_embedding = PromptEncoder(rng).encode("Point(100, 150)")
    mask = MaskDecoder(rng).decode(image_embedding, prompt_embedding)

    print("\nCOMPONENT OUTPUTS")
    print(f"Image embedding shape:  {image_embedding.shape}")
    print(f"Prompt embedding shape: {prompt_embedding.shape}")
    print(f"Mask shape:             {mask.shape}")
    print("Educational component example completed.")


if __name__ == "__main__":
    main()
