"""Tests for the executable mock pipeline."""

import unittest

from src.agent import VisionAgent
from src.sam3_adapter import MockSAM3Adapter, RealSAM3Adapter
from src.schemas import AgentRequest


class VisionAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.request = AgentRequest(
            media_path="data/input/example.jpg",
            goal="Segment every vehicle and measure its visible area",
            prompt="vehicle",
            confidence=0.25,
        )

    def test_mock_pipeline_returns_valid_detection(self) -> None:
        run = VisionAgent(MockSAM3Adapter()).run(self.request)
        self.assertEqual(run.segmentation.backend, "mock")
        self.assertEqual(len(run.segmentation.detections), 1)
        self.assertEqual(run.segmentation.detections[0].pixel_area, 84210)
        self.assertFalse(run.segmentation.metadata["validated_inference"])

    def test_request_rejects_invalid_confidence(self) -> None:
        request = AgentRequest(
            media_path="image.jpg",
            goal="segment",
            prompt="vehicle",
            confidence=1.5,
        )
        with self.assertRaises(ValueError):
            request.validate()

    def test_real_backend_is_explicitly_pending(self) -> None:
        with self.assertRaises(NotImplementedError):
            RealSAM3Adapter().segment(self.request)


if __name__ == "__main__":
    unittest.main()
