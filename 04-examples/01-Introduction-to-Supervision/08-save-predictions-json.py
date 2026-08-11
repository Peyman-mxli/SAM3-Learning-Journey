"""
08-save-predictions-json.py

Introduction to Supervision — Example 08

Goal:
Convert YOLO + Supervision detections into a
structured Python dictionary and save the
predictions as a JSON file.
"""

from pathlib import Path
import urllib.request
import json

import cv2
import supervision as sv
from ultralytics import YOLO


# --------------------------------------------------
# 1. Prepare the image
# --------------------------------------------------

Path("assets").mkdir(exist_ok=True)

image_path = "assets/bus.jpg"

if not Path(image_path).exists():
    urllib.request.urlretrieve(
        "https://ultralytics.com/images/bus.jpg",
        image_path
    )


# --------------------------------------------------
# 2. Load the image
# --------------------------------------------------

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {image_path}"
    )


# --------------------------------------------------
# 3. Load YOLO
# --------------------------------------------------

model = YOLO("yolov8n.pt")


# --------------------------------------------------
# 4. Run inference
# --------------------------------------------------

results = model(image)[0]


# --------------------------------------------------
# 5. Convert to Supervision
# --------------------------------------------------

detections = sv.Detections.from_ultralytics(
    results
)


# --------------------------------------------------
# 6. Create structured prediction data
# --------------------------------------------------

predictions = []

for i in range(len(detections)):

    x1, y1, x2, y2 = detections.xyxy[i]

    class_id = int(
        detections.class_id[i]
    )

    confidence = float(
        detections.confidence[i]
    )

    prediction = {
        "class_id": class_id,
        "class_name": results.names[class_id],
        "confidence": confidence,
        "bounding_box": {
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2)
        }
    }

    predictions.append(prediction)


# --------------------------------------------------
# 7. Create final JSON structure
# --------------------------------------------------

output = {
    "image": image_path,
    "model": "yolov8n.pt",
    "number_of_detections": len(detections),
    "predictions": predictions
}


# --------------------------------------------------
# 8. Save JSON
# --------------------------------------------------

output_path = "predictions.json"

with open(
    output_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        output,
        file,
        indent=4
    )


# --------------------------------------------------
# 9. Display result
# --------------------------------------------------

print(
    f"Predictions saved to: "
    f"{output_path}"
)

print(
    f"Number of detections: "
    f"{len(detections)}"
)

print("\nJSON preview:")

print(
    json.dumps(
        output,
        indent=4
    )
)


# --------------------------------------------------
# Pipeline
# --------------------------------------------------

"""
Image
  ↓
YOLO
  ↓
Ultralytics Results
  ↓
sv.Detections
  ↓
Extract:
  ├── class_id
  ├── class_name
  ├── confidence
  └── bounding_box
  ↓
Python Dictionary
  ↓
JSON
  ↓
predictions.json
"""
