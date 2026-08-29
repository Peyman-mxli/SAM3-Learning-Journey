"""Verify authenticated access to the official SAM 3 Hub repository."""

from huggingface_hub import HfApi


MODEL_ID = "facebook/sam3"


def main() -> None:
    info = HfApi().model_info(MODEL_ID)
    print(f"Repository: {info.id}")
    print(f"Revision SHA: {info.sha}")
    print("Access verified. No token value was displayed.")


if __name__ == "__main__":
    main()
