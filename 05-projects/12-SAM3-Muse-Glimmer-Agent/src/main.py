"""Command-line entry point for Project 12."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import VisionAgent
from .sam3_adapter import build_sam3_adapter
from .schemas import AgentRequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the SAM 3 + Muse Glimmer vision-agent scaffold."
    )
    parser.add_argument("--media", required=True, help="Input image or video path")
    parser.add_argument("--goal", required=True, help="Natural-language task")
    parser.add_argument("--prompt", required=True, help="SAM 3 semantic prompt")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--backend", choices=("mock", "real"), default="mock")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--output", default="results/agent-result.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    request = AgentRequest(
        media_path=args.media,
        goal=args.goal,
        prompt=args.prompt,
        confidence=args.confidence,
    )
    adapter = build_sam3_adapter(args.backend)
    agent = VisionAgent(adapter, max_retries=args.max_retries)
    run = agent.run(request)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(run.to_dict(), indent=2),
        encoding="utf-8",
    )

    print(run.summary)
    print(f"Result saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
