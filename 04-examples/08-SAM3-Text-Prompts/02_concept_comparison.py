from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import supervision as sv
import torch
from ultralytics.models.sam import SAM3SemanticPredictor

BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "assets" / "input" / "bus.jpg"
OUTPUT_PATH = BASE_DIR / "assets" / "output" / "02_concept_comparison_output.png"
SAM_MODEL_PATH = Path("/content/drive/MyDrive/SAM3-Models/sam3.pt")
PROMPTS = ["vehicle", "bus", "person"]

if not IMAGE_PATH.exists() or not SAM_MODEL_PATH.exists():
    raise FileNotFoundError("Required input image or SAM 3 checkpoint is missing.")
image = cv2.imread(str(IMAGE_PATH))
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

overrides = {"conf": 0.25, "task": "segment", "mode": "predict", "model": str(SAM_MODEL_PATH)}
if torch.cuda.is_available():
    overrides["half"] = True
predictor = SAM3SemanticPredictor(overrides=overrides)
predictor.set_image(image)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, prompt in zip(axes, PROMPTS):
    detections = sv.Detections.from_ultralytics(predictor(text=[prompt])[0])
    scene = sv.MaskAnnotator(opacity=0.6).annotate(image.copy(), detections)
    ax.imshow(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
    ax.set_title(f'"{prompt}" — {len(detections)} objects')
    ax.axis("off")
    print(f"{prompt}: {len(detections)}")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Output: {OUTPUT_PATH}")
print("Concept-comparison example completed.")
