from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch
from ultralytics.models.sam import SAM3SemanticPredictor

BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "assets" / "input" / "bus.jpg"
OUTPUT_PATH = BASE_DIR / "assets" / "output" / "04_filter_text_detections_output.png"
SAM_MODEL_PATH = Path("/content/drive/MyDrive/SAM3-Models/sam3.pt")
CONFIDENCE_MIN = 0.50
AREA_MIN = 1_000

if not IMAGE_PATH.exists() or not SAM_MODEL_PATH.exists():
    raise FileNotFoundError("Required input image or SAM 3 checkpoint is missing.")
image = cv2.imread(str(IMAGE_PATH))
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

overrides = {"conf": 0.25, "task": "segment", "mode": "predict", "model": str(SAM_MODEL_PATH)}
if torch.cuda.is_available():
    overrides["half"] = True
predictor = SAM3SemanticPredictor(overrides=overrides)
predictor.set_image(image)
detections = sv.Detections.from_ultralytics(predictor(text=["person"])[0])

areas = np.array([int(mask.sum()) for mask in detections.mask])
keep = (detections.confidence >= CONFIDENCE_MIN) & (areas >= AREA_MIN)
filtered = detections[keep]
labels = [f"person {confidence:.2f}" for confidence in filtered.confidence]

scene = sv.MaskAnnotator(opacity=0.6).annotate(image.copy(), filtered)
scene = sv.LabelAnnotator().annotate(scene, filtered, labels=labels)
if not cv2.imwrite(str(OUTPUT_PATH), scene):
    raise RuntimeError(f"Could not save: {OUTPUT_PATH}")

print(f"Original detections: {len(detections)}")
print(f"Filtered detections: {len(filtered)}")
print(f"Output: {OUTPUT_PATH}")
print("Text-detection filtering example completed.")
