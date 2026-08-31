# Execution Runbook

## Mock Validation

```bash
python -m unittest discover -s tests -v
bash scripts/run_mock.sh
```

## Real SAM 3

```bash
python -m pip install -r requirements.txt
python -m src.main \
  --media ../09-SAM3-Semantic-Prompt-Analytics/data/input/bus.jpg \
  --goal "Segment every vehicle and measure its visible area" \
  --prompt vehicle \
  --backend sam3 \
  --sam3-checkpoint /content/drive/MyDrive/SAM3-Models/sam3.pt \
  --output results/json/sam3-result.json
```

## Start Muse Glimmer

```bash
python -m pip install vllm
bash scripts/serve_glimmer_vllm.sh
```

## Combined Run

```bash
python -m src.main \
  --media ../09-SAM3-Semantic-Prompt-Analytics/data/input/bus.jpg \
  --goal "Find vehicles, segment them, and report visible pixel area" \
  --backend glimmer-sam3 \
  --glimmer-base-url http://127.0.0.1:8000/v1 \
  --sam3-checkpoint /content/drive/MyDrive/SAM3-Models/sam3.pt \
  --output results/json/glimmer-sam3-result.json
```

Verify JSON, CSV, masks, annotated image, runtime metadata, counts, areas, latency, and backend markers.
