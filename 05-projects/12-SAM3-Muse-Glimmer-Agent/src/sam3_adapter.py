"""SAM 3 adapter boundary.

The mock implementation validates orchestration. The real adapter remains
deliberately incomplete until the actual runtime is selected and tested.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import uuid4

from .schemas import AgentRequest, Detection, SegmentationResult


class SAM3Adapter(ABC):
    @abstractmethod
    def segment(self, request: AgentRequest) -> SegmentationResult:
        """Return a schema-valid segmentation result."""


class MockSAM3Adapter(SAM3Adapter):
    def segment(self, request: AgentRequest) -> SegmentationResult:
        request.validate()
        detection = Detection(
            detection_id=0,
            label=request.prompt,
            confidence=max(request.confidence, 0.90),
            bbox_xyxy=(120.0, 240.0, 510.0, 690.0),
            mask_path="data/output/mock-mask-000.png",
            pixel_area=84210,
        )
        result = SegmentationResult(
            request_id=f"mock-{uuid4().hex[:12]}",
            backend="mock",
            detections=[detection],
            warnings=[
                "Mock output: no image was inspected and no model inference ran."
            ],
            metadata={
                "validated_inference": False,
                "purpose": "orchestration and schema testing only",
            },
        )
        result.validate()
        return result


class RealSAM3Adapter(SAM3Adapter):
    def segment(self, request: AgentRequest) -> SegmentationResult:
        request.validate()
        raise NotImplementedError(
            "Real SAM 3 inference is pending runtime and hardware validation."
        )


def build_sam3_adapter(backend: str) -> SAM3Adapter:
    normalized = backend.strip().lower()
    if normalized == "mock":
        return MockSAM3Adapter()
    if normalized == "real":
        return RealSAM3Adapter()
    raise ValueError(f"Unsupported SAM 3 backend: {backend}")
