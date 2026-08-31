"""Bounded vision-agent orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from .sam3_adapter import SAM3Adapter
from .schemas import AgentRequest, SegmentationResult


@dataclass
class AgentRun:
    request: AgentRequest
    segmentation: SegmentationResult
    summary: str

    def to_dict(self) -> dict:
        return {
            "request": {
                "media_path": self.request.media_path,
                "goal": self.request.goal,
                "prompt": self.request.prompt,
                "confidence": self.request.confidence,
            },
            "segmentation": self.segmentation.to_dict(),
            "summary": self.summary,
        }


class VisionAgent:
    def __init__(self, sam3: SAM3Adapter, max_retries: int = 2) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.sam3 = sam3
        self.max_retries = max_retries

    def run(self, request: AgentRequest) -> AgentRun:
        request.validate()
        last_result: SegmentationResult | None = None

        for _attempt in range(self.max_retries + 1):
            result = self.sam3.segment(request)
            result.validate()
            last_result = result
            if result.detections:
                break

        if last_result is None:
            raise RuntimeError("The segmentation adapter returned no result")

        count = len(last_result.detections)
        total_area = sum(item.pixel_area for item in last_result.detections)
        qualifier = "mock" if last_result.backend == "mock" else "model"
        summary = (
            f"The {qualifier} pipeline returned {count} detection(s) "
            f"with a combined mask area of {total_area} pixels."
        )
        return AgentRun(request=request, segmentation=last_result, summary=summary)
