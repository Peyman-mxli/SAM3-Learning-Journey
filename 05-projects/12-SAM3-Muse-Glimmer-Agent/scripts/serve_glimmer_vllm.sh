#!/usr/bin/env bash
set -euo pipefail
vllm serve "meta-models/Muse-Glimmer-30B" --host 127.0.0.1 --port 8000
