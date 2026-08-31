import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.agent import VisionAgent
from src.reporting import export_run
from src.sam3_adapter import DemoSegmentationAdapter
from src.schemas import AgentRequest


ROOT = Path(__file__).parents[1]


class PipelineTests(unittest.TestCase):
    def test_demo_measures_six_pixels(self):
        with tempfile.TemporaryDirectory() as folder:
            request = AgentRequest(str(ROOT / "data/input/sample-scene.ppm"), "Find the red vehicle", "vehicle")
            run = VisionAgent(DemoSegmentationAdapter(folder)).run(request)
            self.assertEqual(run.segmentation.detections[0].pixel_area, 6)
            self.assertEqual(run.segmentation.detections[0].bbox_xyxy, (1, 1, 4, 3))
            self.assertTrue(Path(run.segmentation.detections[0].mask_path).is_file())

    def test_exports_json_and_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            run = VisionAgent(DemoSegmentationAdapter(folder)).run(AgentRequest(str(ROOT / "data/input/sample-scene.ppm"), "Measure", "vehicle"))
            jp, cp = Path(folder)/"run.json", Path(folder)/"run.csv"
            export_run(run.to_dict(), str(jp), str(cp))
            self.assertEqual(json.loads(jp.read_text())["segmentation"]["backend"], "demo-color-segmentation")
            with cp.open(encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)

    def test_request_rejects_invalid_confidence(self):
        with self.assertRaises(ValueError):
            AgentRequest("x", "y", "z", 2).validate()


if __name__ == "__main__":
    unittest.main()
