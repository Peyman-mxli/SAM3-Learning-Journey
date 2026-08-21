# Google Colab Workflow

This document preserves the tested Google Colab workflow used to develop and validate the **Visual Tracking and Analysis System**.

It records the important commands, environment configuration, troubleshooting steps, and working SAM 3 setup used during development.

---

## Environment

The project was tested in Google Colab using:

```text
GPU: Tesla T4
GPU VRAM: 15,360 MiB
Python: 3.12.13
CUDA available: True
```

Initial project dependencies included:

```text
OpenCV: 4.14.0
NumPy: 2.0.2
Pandas: 2.2.3
Supervision: 0.30.0
Ultralytics: 8.4.124
PyTorch: 2.11.0+cu128
```

After installing the official SAM 3 repository, NumPy was changed to:

```text
NumPy: 1.26.4
```

A Colab runtime restart was required after this change to avoid binary compatibility errors.

---

# 01 — Check GPU

```python
!nvidia-smi
```

Expected environment:

```text
Tesla T4
CUDA available
Approximately 15 GB VRAM
```

---

# 02 — Clone SAM3 Learning Journey

```python
!git clone https://github.com/Peyman-mxli/SAM3-Learning-Journey.git
```

---

# 03 — Enter Project Directory

```python
%cd /content/SAM3-Learning-Journey/05-projects/06-Visual-Tracking-and-Analysis-System
```

---

# 04 — Verify Project Files

```python
!find . -maxdepth 3 -type f | sort
```

Important files include:

```text
README.md
app.py
requirements.txt

src/
├── database.py
├── detector.py
├── metrics.py
├── segmenter.py
├── tracker.py
└── visualization.py

assets/
├── README.md
├── input/README.md
└── output/README.md

data/README.md
docs/PROJECT-PROPOSAL.md
reports/README.md
notebooks/README.md
```

---

# 05 — Install Project Dependencies

```python
!pip install -q -r requirements.txt
```

---

# 06 — Environment Verification

```python
import sys
import cv2
import numpy as np
import pandas as pd
import supervision as sv
import ultralytics
import torch

print("Environment verification")
print("-" * 40)

print("Python:", sys.version.split()[0])
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Supervision:", sv.__version__)
print("Ultralytics:", ultralytics.__version__)
print("PyTorch:", torch.__version__)

print("-" * 40)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("-" * 40)
print("Environment test: SUCCESS")
```

Verified initial result:

```text
Python: 3.12.13
OpenCV: 4.14.0
NumPy: 2.0.2
Pandas: 2.2.3
Supervision: 0.30.0
Ultralytics: 8.4.124
PyTorch: 2.11.0+cu128
CUDA available: True
GPU: Tesla T4
Environment test: SUCCESS
```

---

# 07 — Verify Project Modules

```python
from src.detector import ObjectDetector
from src.segmenter import ObjectSegmenter
from src.tracker import ObjectTracker
from src.database import AnalysisDatabase
from src.visualization import TrackingVisualizer

from src.metrics import (
    calculate_iou,
    calculate_precision,
    calculate_recall,
    calculate_dice,
)

print("Project module verification")
print("-" * 40)

print("ObjectDetector: OK")
print("ObjectSegmenter: OK")
print("ObjectTracker: OK")
print("AnalysisDatabase: OK")
print("TrackingVisualizer: OK")
print("Metrics functions: OK")

print("-" * 40)
print("Project modules test: SUCCESS")
```

---

# 08 — Metrics Functional Test

```python
from src.metrics import (
    calculate_iou,
    calculate_precision,
    calculate_recall,
    calculate_dice,
)

print("Metrics functional test")
print("-" * 40)

box_a = [0, 0, 100, 100]
box_b = [50, 50, 150, 150]

iou = calculate_iou(
    box_a,
    box_b
)

precision = calculate_precision(
    true_positives=8,
    false_positives=2
)

recall = calculate_recall(
    true_positives=8,
    false_negatives=2
)

dice = calculate_dice(
    intersection=80,
    predicted_area=100,
    ground_truth_area=100
)

print(f"IoU:       {iou:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"Dice:      {dice:.4f}")

print("-" * 40)

assert abs(iou - 0.142857) < 0.001
assert precision == 0.8
assert recall == 0.8
assert dice == 0.8

print("Metrics functional test: SUCCESS")
```

Verified result:

```text
IoU:       0.1429
Precision: 0.8000
Recall:    0.8000
Dice:      0.8000

Metrics functional test: SUCCESS
```

---

# 09 — Database Functional Test

```python
from src.database import AnalysisDatabase
from pathlib import Path
import sqlite3

print("Database functional test")
print("-" * 40)

test_database = "data/test_analysis.db"

database_path = Path(
    test_database
)

if database_path.exists():
    database_path.unlink()

db = AnalysisDatabase(
    test_database
)

session_id = db.create_session(
    source="test_video.mp4",
    notes="Colab database functional test"
)

print(
    "Session created:",
    session_id
)

db.add_observation(
    session_id=session_id,
    frame_number=1,
    timestamp=0.033,
    tracker_id=1,
    class_id=0,
    class_name="person",
    confidence=0.95,
    bounding_box=[
        100,
        120,
        300,
        450
    ],
    notes="Test observation"
)

db.close()

connection = sqlite3.connect(
    test_database
)

cursor = connection.cursor()

cursor.execute(
    "SELECT COUNT(*) FROM sessions"
)

session_count = cursor.fetchone()[0]

cursor.execute(
    "SELECT COUNT(*) FROM observations"
)

observation_count = cursor.fetchone()[0]

connection.close()

print(
    "Sessions stored:",
    session_count
)

print(
    "Observations stored:",
    observation_count
)

print("-" * 40)

assert session_count == 1
assert observation_count == 1

print(
    "Database functional test: SUCCESS"
)
```

Verified result:

```text
Session created: 1
Sessions stored: 1
Observations stored: 1

Database functional test: SUCCESS
```

---

# 10 — Download YOLO Test Image

```python
import urllib.request
from pathlib import Path

image_url = (
    "https://ultralytics.com/images/bus.jpg"
)

image_path = Path(
    "assets/input/yolo_bus_test.jpg"
)

image_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

urllib.request.urlretrieve(
    image_url,
    image_path
)

print(
    "Test image:",
    image_path
)

print(
    "Exists:",
    image_path.exists()
)
```

---

# 11 — Object Detector Functional Test

```python
import cv2

from src.detector import ObjectDetector

print(
    "Object detector functional test"
)

print("-" * 40)

image = cv2.imread(
    "assets/input/yolo_bus_test.jpg"
)

assert image is not None

print(
    "Test image loaded:",
    image.shape
)

detector = ObjectDetector()

detections = detector.detect(
    image
)

print(
    "Detections found:",
    len(detections)
)

assert len(detections) > 0

print("-" * 40)

print(
    "Object detector functional test: SUCCESS"
)
```

Verified result:

```text
Test image loaded: (1080, 810, 3)
Detections found: 4

Object detector functional test: SUCCESS
```

---

# 12 — ByteTrack Functional Test

```python
import cv2

from src.detector import ObjectDetector
from src.tracker import ObjectTracker

print(
    "Object tracking functional test"
)

print("-" * 40)

image = cv2.imread(
    "assets/input/yolo_bus_test.jpg"
)

assert image is not None

detector = ObjectDetector()
tracker = ObjectTracker()

detections_1 = detector.detect(
    image
)

tracked_1 = tracker.update(
    detections_1
)

detections_2 = detector.detect(
    image
)

tracked_2 = tracker.update(
    detections_2
)

print(
    "Frame 1 detections:",
    len(tracked_1)
)

print(
    "Frame 2 detections:",
    len(tracked_2)
)

print(
    "Frame 1 tracker IDs:",
    tracked_1.tracker_id
)

print(
    "Frame 2 tracker IDs:",
    tracked_2.tracker_id
)

assert tracked_1.tracker_id is not None
assert tracked_2.tracker_id is not None

print("-" * 40)

print(
    "Object tracking functional test: SUCCESS"
)
```

Verified result:

```text
Frame 1 detections: 4
Frame 2 detections: 4

Frame 1 tracker IDs: [1 2 3 4]
Frame 2 tracker IDs: [1 2 3 4]

Object tracking functional test: SUCCESS
```

---

# 13 — Visualization Functional Test

```python
import cv2
from pathlib import Path

from src.detector import ObjectDetector
from src.tracker import ObjectTracker
from src.visualization import TrackingVisualizer

print(
    "Visualization functional test"
)

print("-" * 40)

input_path = Path(
    "assets/input/yolo_bus_test.jpg"
)

output_path = Path(
    "assets/output/yolo_bus_tracking_test.jpg"
)

image = cv2.imread(
    str(input_path)
)

assert image is not None

detector = ObjectDetector()
tracker = ObjectTracker()
visualizer = TrackingVisualizer()

detections = detector.detect(
    image
)

tracked_detections = tracker.update(
    detections
)

class_names = detector.get_class_names()

annotated_image = visualizer.annotate(
    image=image,
    detections=tracked_detections,
    class_names=class_names
)

saved = cv2.imwrite(
    str(output_path),
    annotated_image
)

assert saved
assert output_path.exists()

print(
    "Detections visualized:",
    len(tracked_detections)
)

print(
    "Output saved to:",
    output_path
)

print("-" * 40)

print(
    "Visualization functional test: SUCCESS"
)
```

---

# 14 — Display Detection and Tracking Result

```python
from IPython.display import display
from PIL import Image

output_image = Image.open(
    "assets/output/yolo_bus_tracking_test.jpg"
)

display(
    output_image
)
```

The verified output contained:

```text
#1 bus
#2 person
#3 person
#4 person
```

with bounding boxes, class names, confidence scores, and persistent tracker IDs.

---

# 15 — Install Official Meta SAM 3

Clone the official repository using a name that does not conflict with the Python `sam3` package.

```python
!git clone https://github.com/facebookresearch/sam3.git /content/sam3_repo
!pip install -q -e /content/sam3_repo
```

Important:

Do not clone directly to:

```text
/content/sam3
```

because the outer folder name can shadow the actual Python `sam3` package.

Use:

```text
/content/sam3_repo
```

instead.

---

# 16 — NumPy Compatibility

Installing SAM 3 changed NumPy from:

```text
2.0.2
```

to:

```text
1.26.4
```

This produced a binary compatibility error until the Colab runtime was restarted.

Observed error:

```text
ValueError:
numpy.dtype size changed,
may indicate binary incompatibility.
```

Solution:

```text
Runtime
→ Restart session
```

Do not use:

```text
Factory reset runtime
```

unless a full environment reset is intentionally required.

---

# 17 — Restore SAM 3 Import Path

After the restart:

```python
import sys

repo_path = "/content/sam3_repo"

if repo_path not in sys.path:
    sys.path.insert(
        0,
        repo_path
    )

from sam3.model_builder import (
    build_sam3_image_model
)

from sam3.model.sam3_image_processor import (
    Sam3Processor
)

print(
    "SAM 3 imports after restart: SUCCESS"
)
```

Verified result:

```text
SAM 3 imports after restart: SUCCESS
```

---

# 18 — Hugging Face Access

SAM 3 is a gated model.

The Hugging Face account used in Colab must be the same account that has been granted access to:

```text
facebook/sam3
```

The project initially received:

```text
403 Forbidden
```

because Colab was authenticated using a different Hugging Face account.

The correct account showed:

```text
Gated model
You have been granted access to this model
```

---

# 19 — Hugging Face Token

Store the Hugging Face token securely in:

```text
Google Colab
→ Secrets
→ HF_TOKEN
```

Do not store the token directly inside notebook code.

The token must have read permission.

---

# 20 — Authenticate with Colab Secret

```python
from google.colab import userdata

from huggingface_hub import (
    login,
    whoami
)

hf_token = userdata.get(
    "HF_TOKEN"
)

login(
    token=hf_token,
    add_to_git_credential=False
)

user = whoami(
    token=hf_token
)

print(
    "Logged in as:",
    user["name"]
)

print(
    "HF_TOKEN authentication: SUCCESS"
)
```

---

# 21 — Verify Gated SAM 3 Access

```python
from huggingface_hub import (
    hf_hub_download
)

from google.colab import userdata

config_path = hf_hub_download(
    repo_id="facebook/sam3",
    filename="config.json",
    token=userdata.get("HF_TOKEN")
)

print(
    "Downloaded:",
    config_path
)

print(
    "SAM 3 gated-file access: SUCCESS"
)
```

Verified result:

```text
SAM 3 gated-file access: SUCCESS
```

---

# 22 — Download SAM 3 Checkpoint Separately

Downloading the checkpoint separately makes the process easier to verify and prevents accidental repeated downloads.

```python
from huggingface_hub import (
    hf_hub_download
)

from google.colab import userdata
from pathlib import Path

print(
    "Downloading SAM 3 checkpoint..."
)

print("-" * 50)

checkpoint_path = hf_hub_download(
    repo_id="facebook/sam3",
    filename="sam3.pt",
    token=userdata.get("HF_TOKEN")
)

checkpoint = Path(
    checkpoint_path
)

print("-" * 50)

print(
    "Checkpoint:",
    checkpoint
)

print(
    f"Size: "
    f"{checkpoint.stat().st_size / (1024 ** 3):.2f} GB"
)

print(
    "SAM 3 checkpoint download: SUCCESS"
)
```

Verified checkpoint size:

```text
Approximately 3.21 GB
```

---

# 23 — Load SAM 3 from Local Checkpoint

```python
import torch

print(
    "Loading SAM 3 from local checkpoint..."
)

print("-" * 50)

torch.cuda.empty_cache()

print(
    f"GPU before load - allocated: "
    f"{torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB"
)

print(
    f"GPU before load - reserved: "
    f"{torch.cuda.memory_reserved() / (1024 ** 3):.2f} GB"
)

model = build_sam3_image_model(
    checkpoint_path=str(checkpoint_path),
    device="cuda",
    load_from_HF=False,
    enable_segmentation=True
)

processor = Sam3Processor(
    model
)

print("-" * 50)

print(
    "SAM 3 model loaded: SUCCESS"
)

print(
    f"GPU after load - allocated: "
    f"{torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB"
)

print(
    f"GPU after load - reserved: "
    f"{torch.cuda.memory_reserved() / (1024 ** 3):.2f} GB"
)
```

Verified result:

```text
SAM 3 model loaded: SUCCESS

GPU after load - allocated: 3.33 GB
GPU after load - reserved: 3.43 GB
```

The Tesla T4 successfully loaded SAM 3.

---

# 24 — SAM 3 BFloat16 Fix

Initial SAM 3 image inference produced:

```text
RuntimeError:
mat1 and mat2 must have the same dtype,
but got BFloat16 and Float
```

The working solution was to use CUDA autocast with:

```python
torch.bfloat16
```

during both image encoding and prompt inference.

---

# 25 — Real SAM 3 Segmentation Test

```python
from PIL import Image
import torch

print(
    "Running SAM 3 segmentation test..."
)

print("-" * 50)

image_path = (
    "/content/SAM3-Learning-Journey/"
    "05-projects/"
    "06-Visual-Tracking-and-Analysis-System/"
    "assets/input/yolo_bus_test.jpg"
)

image = Image.open(
    image_path
).convert(
    "RGB"
)

print(
    "Image loaded:",
    image.size
)

with torch.autocast(
    device_type="cuda",
    dtype=torch.bfloat16
):

    inference_state = processor.set_image(
        image
    )

    output = processor.set_text_prompt(
        state=inference_state,
        prompt="person"
    )

print("-" * 50)

print(
    "SAM 3 segmentation inference: SUCCESS"
)

print(
    "Number of masks:",
    len(output["masks"])
)

print(
    "Number of boxes:",
    len(output["boxes"])
)

print(
    "Scores:",
    output["scores"]
)
```

Verified result:

```text
SAM 3 segmentation inference: SUCCESS

Number of masks: 4
Number of boxes: 4
```

Observed confidence scores were approximately:

```text
0.9570
0.9414
0.9688
0.9688
```

---

# 26 — Visualize SAM 3 Masks

```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print(
    "Visualizing SAM 3 segmentation..."
)

print("-" * 50)

output_path = Path(
    "/content/SAM3-Learning-Journey/"
    "05-projects/"
    "06-Visual-Tracking-and-Analysis-System/"
    "assets/output/"
    "sam3_person_segmentation.jpg"
)

image_np = np.array(
    image
)

plt.figure(
    figsize=(10, 12)
)

plt.imshow(
    image_np
)

for mask, score in zip(
    output["masks"],
    output["scores"]
):

    mask_np = (
        mask
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    mask_np = np.squeeze(
        mask_np
    )

    plt.contour(
        mask_np,
        levels=[0.5],
        linewidths=2
    )

plt.axis(
    "off"
)

plt.tight_layout()

plt.savefig(
    output_path,
    bbox_inches="tight",
    pad_inches=0
)

plt.show()

print("-" * 50)

print(
    "Saved to:",
    output_path
)

print(
    "SAM 3 visualization: SUCCESS"
)
```

The resulting image successfully displayed segmentation boundaries around the people detected by SAM 3.

---

# Verified Project Components

The following components were tested successfully in Google Colab:

```text
YOLO Detection                 SUCCESS
Supervision Detections         SUCCESS
ByteTrack                      SUCCESS
Persistent Tracker IDs         SUCCESS
Visualization                  SUCCESS
SQLite Database                SUCCESS
IoU Metric                     SUCCESS
Precision Metric               SUCCESS
Recall Metric                  SUCCESS
Dice Metric                    SUCCESS
SAM 3 Import                   SUCCESS
SAM 3 Gated Access             SUCCESS
SAM 3 Checkpoint Download      SUCCESS
SAM 3 Model Loading            SUCCESS
SAM 3 Text Prompt              SUCCESS
SAM 3 Segmentation Masks       SUCCESS
SAM 3 Bounding Boxes           SUCCESS
SAM 3 Visualization            SUCCESS
Tesla T4 GPU                   SUCCESS
```

---

# Important Lessons

## Hugging Face Account

Gated-model access is associated with the Hugging Face account.

The token used in Colab must belong to the same account that has permission to access:

```text
facebook/sam3
```

---

## Do Not Expose Tokens

Never commit:

```text
HF_TOKEN
```

or any Hugging Face access token to GitHub.

Use Colab Secrets instead.

---

## SAM 3 Repository Path

Use:

```text
/content/sam3_repo
```

instead of:

```text
/content/sam3
```

to avoid Python package shadowing problems.

---

## NumPy Restart

After SAM 3 changes NumPy versions, restart the Colab session before importing libraries compiled against NumPy.

---

## SAM 3 Checkpoint

Download the model checkpoint separately before loading the model.

This makes it easier to:

- Verify the download
- Confirm file size
- Avoid unnecessary repeated downloads
- Load directly from a local checkpoint

---

## SAM 3 Autocast

For the tested Tesla T4 environment, SAM 3 image inference required:

```python
with torch.autocast(
    device_type="cuda",
    dtype=torch.bfloat16
):
```

This resolved the observed BFloat16/Float matrix multiplication mismatch.

---

# Current Status

Project 06 now has working and tested implementations for:

- Object detection
- Object tracking
- Persistent IDs
- Visualization
- Database storage
- Evaluation metrics
- Real SAM 3 segmentation
- Text-prompt segmentation
- Segmentation visualization

The next development stage is to combine these components into a complete end-to-end processing pipeline.

---

## Author

**Peyman Miyandashti**

GitHub: [Peyman-mxli](https://github.com/Peyman-mxli)

LinkedIn: [Peyman Miyandashti](https://www.linkedin.com/in/peyman-mxli/)
