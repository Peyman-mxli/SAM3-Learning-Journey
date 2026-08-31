# Runbook

## Reproducible validation

```bash
bash scripts/validate.sh
```

## Real SAM 3 bridge

Create an importable module with `build_adapter(config)` returning an object whose `segment(request)` method returns `SegmentationResult`. Then set:

```bash
export SAM3_PLUGIN_MODULE=my_sam3_bridge
export SAM3_CONFIG=config/sam3.real.json
python -m src.main --media image.jpg --goal "Segment vehicles" --prompt vehicle --backend real
```

Record checkpoint identity, package versions, GPU model, peak memory, latency, output hashes, and inspected artifacts before claiming real validation.
