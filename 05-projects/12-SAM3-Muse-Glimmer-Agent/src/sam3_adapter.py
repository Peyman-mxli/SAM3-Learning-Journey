"""Segmentation adapters: deterministic demo plus a strict real-runtime bridge."""

from __future__ import annotations

import importlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from uuid import uuid4

from .schemas import AgentRequest, Detection, SegmentationResult


class SAM3Adapter(ABC):
    @abstractmethod
    def segment(self, request: AgentRequest) -> SegmentationResult: ...


def _read_ppm(path: Path) -> tuple[int, int, list[tuple[int, int, int]]]:
    tokens = [x for line in path.read_text(encoding="ascii").splitlines() if not line.startswith("#") for x in line.split()]
    if len(tokens) < 4 or tokens[0] != "P3" or tokens[3] != "255":
        raise ValueError("demo backend requires an ASCII P3 PPM image with max value 255")
    width, height = int(tokens[1]), int(tokens[2])
    values = list(map(int, tokens[4:]))
    if len(values) != width * height * 3:
        raise ValueError("PPM pixel count does not match its dimensions")
    return width, height, list(zip(values[0::3], values[1::3], values[2::3]))


class DemoSegmentationAdapter(SAM3Adapter):
    """Performs measured color segmentation; it is real computation, not SAM 3 inference."""

    def __init__(self, output_dir: str = "data/output") -> None:
        self.output_dir = Path(output_dir)

    def segment(self, request: AgentRequest) -> SegmentationResult:
        request.validate()
        image = Path(request.media_path)
        if not image.is_file():
            raise FileNotFoundError(image)
        width, height, pixels = _read_ppm(image)
        selected = [i for i, (r, g, b) in enumerate(pixels) if r >= 180 and r > g * 1.35 and r > b * 1.35]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        mask_path = self.output_dir / "demo-red-mask.pgm"
        mask = ["P2", f"{width} {height}", "255", " ".join("255" if i in set(selected) else "0" for i in range(len(pixels)))]
        mask_path.write_text("\n".join(mask) + "\n", encoding="ascii")
        detections: list[Detection] = []
        if selected:
            xs, ys = [i % width for i in selected], [i // width for i in selected]
            detections.append(Detection(0, request.prompt, 1.0, (min(xs), min(ys), max(xs)+1, max(ys)+1), str(mask_path), len(selected)))
        result = SegmentationResult(
            request_id=f"demo-{uuid4().hex[:12]}", backend="demo-color-segmentation", detections=detections,
            warnings=["Measured deterministic demo; Muse Glimmer and SAM 3 model weights were not loaded."],
            metadata={"validated_inference": False, "measured_processing": True, "image_size": [width, height], "rule": "red-dominant pixels"},
        )
        result.validate()
        return result


class RealSAM3Adapter(SAM3Adapter):
    """Loads a user-supplied module exposing build_adapter(config)->SAM3Adapter."""

    def __init__(self, plugin_module: str, config_path: str) -> None:
        if not plugin_module:
            raise ValueError("SAM3_PLUGIN_MODULE is required for the real backend")
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        factory = getattr(importlib.import_module(plugin_module), "build_adapter")
        self.adapter = factory(config)

    def segment(self, request: AgentRequest) -> SegmentationResult:
        result = self.adapter.segment(request)
        result.validate()
        if result.backend.startswith("demo"):
            raise ValueError("real backend may not identify itself as demo")
        return result


def build_sam3_adapter(backend: str, output_dir: str = "data/output", plugin_module: str = "", config_path: str = "config/sam3.real.example.json") -> SAM3Adapter:
    if backend == "demo":
        return DemoSegmentationAdapter(output_dir)
    if backend == "real":
        return RealSAM3Adapter(plugin_module, config_path)
    raise ValueError(f"Unsupported backend: {backend}")
