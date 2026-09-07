"""Unit test suite for Stage 8 — Output Layer."""

import json
import os
import sys
import tempfile
import unittest

# Ensure project root is in sys.path when script is executed directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from rtsp_pipeline.output import OutputWriter


class TestOutputWriter(unittest.TestCase):
    """Unit tests for OutputWriter conditional persistence."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.frames_dir = os.path.join(self.temp_dir.name, "frames")
        self.logs_dir = os.path.join(self.temp_dir.name, "logs")
        self.writer = OutputWriter(
            frames_dir=self.frames_dir,
            logs_dir=self.logs_dir,
            metadata_filename="detections.jsonl",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_discard_when_zero_detections(self) -> None:
        """Verify frame image and JSONL log are NOT written when detections list is empty."""
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = self.writer.save_detection_result(dummy_frame, detections=[], frame_timestamp=100.0)

        # Assert returns None
        self.assertIsNone(result)

        # Assert no image files saved in frames_dir
        saved_frames = os.listdir(self.frames_dir) if os.path.exists(self.frames_dir) else []
        self.assertEqual(len(saved_frames), 0)

        # Assert metadata file is either not created or 0 bytes
        if os.path.exists(self.writer.metadata_file_path):
            self.assertEqual(os.path.getsize(self.writer.metadata_file_path), 0)

    def test_save_when_detections_exist(self) -> None:
        """Verify frame image and JSONL log ARE written when detections exist."""
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = [
            {"label": "person", "confidence": 0.89, "colour": None},
            {"label": "car", "confidence": 0.94, "colour": None},
        ]

        result = self.writer.save_detection_result(dummy_frame, detections=detections, frame_timestamp=105.0)

        self.assertIsNotNone(result)
        self.assertEqual(result["detection_count"], 2)

        # Assert image file exists
        saved_frames = os.listdir(self.frames_dir)
        self.assertEqual(len(saved_frames), 1)
        self.assertTrue(saved_frames[0].endswith(".jpg"))

        # Assert JSONL file exists and contains valid line
        self.assertTrue(os.path.exists(self.writer.metadata_file_path))
        with open(self.writer.metadata_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["detection_count"], 2)
            self.assertEqual(record["detections"][0]["label"], "person")


if __name__ == "__main__":
    unittest.main()
