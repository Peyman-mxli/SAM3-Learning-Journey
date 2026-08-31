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
        if not all((self.media_path.strip(), self.goal.strip(), self.prompt.strip())):
            raise ValueError("media_path, goal, and prompt are required")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True)
class Detection:
    detection_id: int
    label: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    mask_path: str
    pixel_area: int

    def validate(self) -> None:
        x1, y1, x2, y2 = self.bbox_xyxy
        if self.detection_id < 0 or not self.label or not 0 <= self.confidence <= 1:
            raise ValueError("invalid detection identity, label, or confidence")
        if x2 <= x1 or y2 <= y1 or self.pixel_area < 0:
            raise ValueError("invalid bounding box or mask area")


@dataclass
class SegmentationResult:
    request_id: str
    backend: str
    detections: list[Detection] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.request_id or not self.backend:
            raise ValueError("request_id and backend are required")
        ids: set[int] = set()
        for item in self.detections:
            item.validate()
            if item.detection_id in ids:
                raise ValueError("detection IDs must be unique")
            ids.add(item.detection_id)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)
