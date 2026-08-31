#!/usr/bin/env bash
set -euo pipefail
python -m src.main --media data/input/example.jpg --goal "Segment every vehicle and measure its visible area" --prompt vehicle --backend mock --output results/json/mock-result.json
