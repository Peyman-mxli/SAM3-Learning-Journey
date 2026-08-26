from pathlib import Path

import cv2
import supervision as sv
import torch
from ultralytics.models.sam import SAM3SemanticPredictor

BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "assets" / "input" / "bus.jpg"
OUTPUT_PATH = BASE_DIR / "assets" / "output" / "05_specific_wheel_prompt_output.png"
SAM_MODEL_PATH = Path("/content/drive/MyDrive/SAM3-Models/sam3.pt")

if not IMAGE_PATH.exists() or not SAM_MODEL_PATH.exists():
    raise FileNotFoundError("Required input image or SAM 3 checkpoint is missing.")
image = cv2.imread(str(IMAGE_PATH))
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

overrides = {"conf": 0.25, "task": "segment", "mode": "predict", "model": str(SAM_MODEL_PATH)}
if torch.cuda.is_available():
    overrides["half"] = True
predictor = SAM3SemanticPredictor(overrides=overrides)
predictor.set_image(image)
detections = sv.Detections.from_ultralytics(predictor(text=["wheel"])[0])
scene = sv.MaskAnnotator(opacity=0.6).annotate(image.copy(), detections)

if not cv2.imwrite(str(OUTPUT_PATH), scene):
    raise RuntimeError(f"Could not save: {OUTPUT_PATH}")
print("Example 05 — Specific Wheel Prompt")
print(f"Wheels detected: {len(detections)}")
print(f"Output: {OUTPUT_PATH}")
print("Specific wheel-prompt example completed.")
