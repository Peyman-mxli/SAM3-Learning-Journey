from pathlib import Path

import cv2
import supervision as sv
import torch
from ultralytics.models.sam import SAM3SemanticPredictor

BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "assets" / "input" / "bus.jpg"
SAM_MODEL_PATH = Path("/content/drive/MyDrive/SAM3-Models/sam3.pt")

if not IMAGE_PATH.exists() or not SAM_MODEL_PATH.exists():
    raise FileNotFoundError("Required input image or SAM 3 checkpoint is missing.")
image = cv2.imread(str(IMAGE_PATH))

overrides = {"conf": 0.25, "task": "segment", "mode": "predict", "model": str(SAM_MODEL_PATH)}
if torch.cuda.is_available():
    overrides["half"] = True
predictor = SAM3SemanticPredictor(overrides=overrides)
predictor.set_image(image)
detections = sv.Detections.from_ultralytics(predictor(text=["person"])[0])

if detections.mask is None or detections.confidence is None:
    raise RuntimeError("Confidence values or masks are unavailable.")

print("Example 03 — Confidence and Mask Area")
print(f"Persons detected: {len(detections)}\n")
for index, (confidence, mask) in enumerate(zip(detections.confidence, detections.mask)):
    print(f"Object {index}: confidence={confidence:.3f} area={int(mask.sum()):,} px²")
print("Confidence and mask-area inspection completed.")
