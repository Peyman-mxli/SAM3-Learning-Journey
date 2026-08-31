from __future__ import annotations

import argparse
import os

from .agent import VisionAgent
from .reporting import export_run
from .sam3_adapter import build_sam3_adapter
from .schemas import AgentRequest


def main() -> int:
    p = argparse.ArgumentParser(description="SAM 3 + Muse Glimmer agent pipeline")
    p.add_argument("--media", required=True); p.add_argument("--goal", required=True); p.add_argument("--prompt", required=True)
    p.add_argument("--backend", choices=("demo", "real"), default="demo"); p.add_argument("--confidence", type=float, default=.25); p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--json", default="results/json/agent-result.json"); p.add_argument("--csv", default="results/csv/detections.csv"); p.add_argument("--output-dir", default="data/output")
    args = p.parse_args()
    adapter = build_sam3_adapter(args.backend, args.output_dir, os.getenv("SAM3_PLUGIN_MODULE", ""), os.getenv("SAM3_CONFIG", "config/sam3.real.example.json"))
    run = VisionAgent(adapter, args.max_retries).run(AgentRequest(args.media, args.goal, args.prompt, args.confidence))
    export_run(run.to_dict(), args.json, args.csv)
    print(run.summary); print(f"JSON: {args.json}\nCSV: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
