#!/usr/bin/env bash
set -euo pipefail
python -m unittest discover -s tests -v
python -m src.main --media data/input/sample-scene.ppm --goal "Segment the red vehicle and measure its visible area" --prompt vehicle --backend demo
python -m json.tool results/json/agent-result.json >/dev/null
