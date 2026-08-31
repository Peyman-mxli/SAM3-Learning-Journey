#!/usr/bin/env bash
set -euo pipefail
python -m src.main --media data/input/sample-scene.ppm --goal "Segment the red vehicle and measure its visible area" --prompt vehicle --backend demo
