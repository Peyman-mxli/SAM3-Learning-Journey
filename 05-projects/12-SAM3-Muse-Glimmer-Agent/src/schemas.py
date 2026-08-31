"""Stable data contracts for the vision-agent pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentRequest:
    media_path: str
    goal: str
    prompt: str
    confidence: float = 0.25

    def validate(self) -> None:
        if not self.media_path.strip():
            raise ValueError("media_path must not be empty")
        if not self.goal.strip():
            raise ValueError("goal must not be empty")
        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True)
class Detection:
    detection_id: int
    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    mask_path: str
    pixel_area: int

    def validate(self) -> None:
        x1, y1, x2, y2 = self.bbox_xyxy
        if self.detection_id < 0:
            raise ValueError("detection_id must be non-negative")
        if not self.label:
            raise ValueError("label must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("detection confidence must be between 0 and 1")
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox_xyxy must have positive width and height")
        if self.pixel_area < 0:
            raise ValueError("pixel_area must be non-negative")


@dataclass
class SegmentationResult:
    request_id: str
    backend: str
    detections: list[Detection] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if not self.backend:
            raise ValueError("backend must not be empty")
        ids: set[int] = set()
        for detection in self.detections:
            detection.validate()
            if detection.detection_id in ids:
                raise ValueError("detection IDs must be unique")
            ids.add(detection.detection_id)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)
