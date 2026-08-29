# Meta SAM 3 on Hugging Face — Model Access and Promptable Segmentation

SAM 3 is Meta's unified foundation model for promptable segmentation in images and videos. It can detect, segment, and track objects using text or visual prompts such as points, boxes, and masks. The official `facebook/sam3` repository on Hugging Face provides the model card, gated files, examples, and integration information used by the advanced course sessions.

## Resource Summary

| Item | Details |
|---|---|
| Model | Segment Anything Model 3 |
| Publisher | Meta / Facebook |
| Hub repository | <https://huggingface.co/facebook/sam3> |
| Input | Image or video plus text or visual prompts |
| Output | Masks, boxes, scores, and temporal tracking results |
| Course association | Sessions 07, 12, and 14 |
| Access | Hugging Face account, accepted model terms, and authentication may be required |

## What Makes SAM 3 Different?

SAM 3 extends promptable visual segmentation with open-vocabulary concepts. It supports two broad forms of interaction:

- **Promptable Concept Segmentation (PCS):** identify instances matching a concept, including text prompts.
- **Promptable Visual Segmentation (PVS):** segment or track targets indicated through visual prompts.

```text
Image or video
      +
Text, point, box, mask, or exemplar prompt
      ↓
SAM 3 model and processor/predictor
      ↓
Masks + boxes + scores + optional temporal identities
```

## Image and Video Capabilities

| Capability | Example prompt | Result |
|---|---|---|
| Text-guided image segmentation | `person`, `vehicle`, `bus` | All matching instances and masks |
| Point prompt | Positive/negative coordinates | Mask refined around indicated object |
| Box prompt | Object bounding box | Segmentation constrained by region |
| Mask prompt | Previous or approximate mask | Refined segmentation |
| Video prompt | Prompt on a selected frame | Propagated masks through time |
| Open-vocabulary concept | Short phrase or exemplar | Matching instances beyond a fixed class list |

## Access Procedure

1. Create or sign in to a Hugging Face account.
2. Open <https://huggingface.co/facebook/sam3>.
3. Read and accept the model terms or request access when prompted.
4. Wait for approval if access is not automatic.
5. Create a read or fine-grained access token.
6. Authenticate the local machine or Colab environment.
7. Confirm access before attempting a large download.

Accepting model terms is separate from creating a token. Both may be required.

## Installation

Use an isolated environment and follow the exact current model-card instructions because dependencies may change.

Typical Hugging Face/Transformers environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision transformers huggingface_hub pillow
```

Google Colab:

```python
!pip install -q -U transformers huggingface_hub
```

Use a GPU runtime for practical image and especially video inference.

## Authentication

Interactive CLI:

```bash
hf auth login
```

Colab secret pattern:

```python
from google.colab import userdata
from huggingface_hub import login

login(token=userdata.get("HF_TOKEN"))
```

Do not place the raw token in the notebook, README, Git commit, screenshot, or console output.

## Verify Access Without Downloading the Full Model

The included script queries repository metadata using your authenticated Hugging Face session:

```bash
python verify_sam3_access.py
```

It prints the repository ID and access confirmation but never prints the token.

## Transformers Image Example

The official model card documents `Sam3Model` and `Sam3Processor`:

```python
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "facebook/sam3"

model = Sam3Model.from_pretrained(model_id).to(device)
processor = Sam3Processor.from_pretrained(model_id)
image = Image.open("image.jpg").convert("RGB")
```

Prompt preparation and post-processing must follow the currently installed Transformers version and the official model card. SAM-related APIs evolve, so record the exact package and checkpoint revision used for every experiment.

## Official SAM 3 Package Pattern

The official model card also documents the Meta package:

```python
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model()
processor = Sam3Processor(model)
state = processor.set_image(Image.open("image.jpg"))
output = processor.set_text_prompt(state=state, prompt="person")

masks = output["masks"]
boxes = output["boxes"]
scores = output["scores"]
```

## Video Session Pattern

```python
from sam3.model_builder import build_sam3_video_predictor

predictor = build_sam3_video_predictor()
start = predictor.handle_request({
    "type": "start_session",
    "resource_path": "video.mp4",
})

response = predictor.handle_request({
    "type": "add_prompt",
    "session_id": start["session_id"],
    "frame_index": 0,
    "text": "vehicle",
})
```

Video inference retains session state. Do not recreate the predictor or session for each frame.

## SAM 3 and YOLO

| YOLOv8 | SAM 3 |
|---|---|
| Fast closed-set detection | Promptable segmentation and tracking |
| Returns boxes/classes/scores | Returns masks/boxes/scores |
| COCO-pretrained class vocabulary | Text and visual prompting |
| Useful for candidate generation | Useful for precise object regions |

The course can combine them by using YOLO boxes as visual prompts for SAM 3, then transferring class, confidence, and tracker metadata to the resulting masks.

## Resource Requirements

- GPU memory depends on checkpoint, resolution, batch size, precision, and video length.
- Video workflows require substantially more memory and processing time than one image.
- Mixed precision may reduce memory use on compatible GPUs.
- Input frames and model tensors must use compatible device and dtype settings.
- Long videos should be processed and written incrementally.

## Common Access Problems

| Symptom | Likely cause | Resolution |
|---|---|---|
| `401 Unauthorized` | No valid authentication | Run `hf auth login` or provide a valid secret |
| `403 Forbidden` | Terms not accepted or access not approved | Open the model page while signed in and complete access request |
| Repository appears public but files fail | Model files are gated | Authenticate with the approved account |
| CUDA out of memory | Model/input exceeds available VRAM | Reduce resolution, frames, batch, or use supported precision |
| API import fails | Package version does not contain SAM 3 classes | Upgrade or use versions stated by the official model card |
| Output alignment is wrong | Resize metadata was ignored | Use the official processor post-processing path |

## Security and Reproducibility

- Never commit weights unless their license and repository policy permit redistribution.
- Never commit access tokens.
- Store the model ID, revision, dependencies, prompts, thresholds, and device.
- Save prompt coordinates in the original-image coordinate system.
- Validate masks visually and numerically.
- Separate illustrative outputs from evaluation claims.

## Course Use

The model resource is associated with Sessions 07, 12, and 14 and supports the repository's point-prompt, text-prompt, video-segmentation, tracking, and semantic-analysis work.

## Official References

- Model card: <https://huggingface.co/facebook/sam3>
- Meta SAM 3 source linked by the model card: <https://github.com/facebookresearch/sam3>
- Hugging Face gated-model documentation: <https://huggingface.co/docs/hub/models-gated>
- Transformers documentation: <https://huggingface.co/docs/transformers/>
