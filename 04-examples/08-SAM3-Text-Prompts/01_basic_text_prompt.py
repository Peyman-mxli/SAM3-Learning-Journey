from pathlib import Path

import cv2
import supervision as sv
import torch
from ultralytics.models.sam import SAM3SemanticPredictor

BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "assets" / "input" / "bus.jpg"
OUTPUT_PATH = BASE_DIR / "assets" / "output" / "01_basic_text_prompt_output.png"
SAM_MODEL_PATH = Path("/content/drive/MyDrive/SAM3-Models/sam3.pt")

if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Input image not found: {IMAGE_PATH}")
if not SAM_MODEL_PATH.exists():
    raise FileNotFoundError(f"SAM 3 model not found: {SAM_MODEL_PATH}")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
image = cv2.imread(str(IMAGE_PATH))
if image is None:
    raise RuntimeError(f"Could not load image: {IMAGE_PATH}")

overrides = {"conf": 0.25, "task": "segment", "mode": "predict", "model": str(SAM_MODEL_PATH)}
if torch.cuda.is_available():
    overrides["half"] = True

predictor = SAM3SemanticPredictor(overrides=overrides)
predictor.set_image(image)
result = predictor(text=["person"])[0]
detections = sv.Detections.from_ultralytics(result)

if detections.mask is None:
    raise RuntimeError("SAM 3 did not return segmentation masks.")

scene = sv.MaskAnnotator(opacity=0.6).annotate(image.copy(), detections)
scene = sv.BoxAnnotator().annotate(scene, detections)
if not cv2.imwrite(str(OUTPUT_PATH), scene):
    raise RuntimeError(f"Could not save: {OUTPUT_PATH}")

print("Example 01 — Basic Text Prompt")
print(f"Prompt: person")
print(f"Objects detected: {len(detections)}")
print(f"Output: {OUTPUT_PATH}")
print("Basic text-prompt example completed.")
