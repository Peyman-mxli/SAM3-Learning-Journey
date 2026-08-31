from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

from .sam3_adapter import SAM3Adapter
from .schemas import AgentRequest, SegmentationResult


@dataclass
class AgentRun:
    request: AgentRequest
    segmentation: SegmentationResult
    summary: str
    attempts: int
    elapsed_ms: float

    def to_dict(self) -> dict:
        return {"request": asdict(self.request), "segmentation": self.segmentation.to_dict(), "summary": self.summary, "attempts": self.attempts, "elapsed_ms": round(self.elapsed_ms, 3)}


class VisionAgent:
    def __init__(self, sam3: SAM3Adapter, max_retries: int = 2) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.sam3, self.max_retries = sam3, max_retries

    def run(self, request: AgentRequest) -> AgentRun:
        request.validate()
        started, result = perf_counter(), None
        for attempt in range(1, self.max_retries + 2):
            result = self.sam3.segment(request)
            result.validate()
            if result.detections:
                break
        assert result is not None
        area = sum(x.pixel_area for x in result.detections)
        summary = f"Verified tool result: {len(result.detections)} detection(s), combined mask area {area} pixels. Backend: {result.backend}."
        return AgentRun(request, result, summary, attempt, (perf_counter() - started) * 1000)
