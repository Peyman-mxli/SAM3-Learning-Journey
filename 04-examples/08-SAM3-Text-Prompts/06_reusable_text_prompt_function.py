from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import supervision as sv
import torch
from ultralytics.models.sam import SAM3SemanticPredictor

BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "assets" / "input" / "bus.jpg"
OUTPUT_PATH = BASE_DIR / "assets" / "output" / "06_reusable_prompt_comparison_output.png"
SAM_MODEL_PATH = Path("/content/drive/MyDrive/SAM3-Models/sam3.pt")
PROMPTS = ["vehicle", "bus", "person", "wheel"]

def segment_concept(predictor, image, prompt):
    result = predictor(text=[prompt])[0]
    detections = sv.Detections.from_ultralytics(result)
    scene = sv.MaskAnnotator(opacity=0.6).annotate(image.copy(), detections)
    return detections, scene

if not IMAGE_PATH.exists() or not SAM_MODEL_PATH.exists():
    raise FileNotFoundError("Required input image or SAM 3 checkpoint is missing.")
image = cv2.imread(str(IMAGE_PATH))
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

overrides = {"conf": 0.25, "task": "segment", "mode": "predict", "model": str(SAM_MODEL_PATH)}
if torch.cuda.is_available():
    overrides["half"] = True
predictor = SAM3SemanticPredictor(overrides=overrides)
predictor.set_image(image)

fig, axes = plt.subplots(2, 2, figsize=(12, 14))
for ax, prompt in zip(axes.ravel(), PROMPTS):
    detections, scene = segment_concept(predictor, image, prompt)
    ax.imshow(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
    ax.set_title(f'"{prompt}" — {len(detections)} objects')
    ax.axis("off")
    print(f"{prompt}: {len(detections)}")

plt.suptitle("SAM3 Text-Prompt Segmentation", fontsize=16)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Output: {OUTPUT_PATH}")
print("Reusable text-prompt example completed.")
