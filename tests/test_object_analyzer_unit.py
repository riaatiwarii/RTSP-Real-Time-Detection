"""Unit test suite for Stage 4 — Object and Crowd Analyzer."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path when script is executed directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np


class TestObjectAnalyzer(unittest.TestCase):
    """Unit tests for ObjectAnalyzer logic."""

    @patch("rtsp_pipeline.object_analyzer.YOLO")
    def test_detection_and_crowd_counting(self, mock_yolo_cls) -> None:
        """Verify object detection extraction and crowd metric derivation."""
        # Setup mock YOLO model instance
        mock_model = MagicMock()
        mock_model.names = {0: "person", 2: "car"}
        mock_yolo_cls.return_value = mock_model

        # Mock YOLO detection boxes (3 persons, 1 car)
        mock_box1 = MagicMock()
        mock_box1.cls = [0]
        mock_box1.conf = [0.92]
        mock_box1.xyxy = [MagicMock(cpu=lambda: MagicMock(numpy=lambda: np.array([10, 20, 100, 200])))]

        mock_box2 = MagicMock()
        mock_box2.cls = [0]
        mock_box2.conf = [0.85]
        mock_box2.xyxy = [MagicMock(cpu=lambda: MagicMock(numpy=lambda: np.array([150, 30, 250, 210])))]

        mock_box3 = MagicMock()
        mock_box3.cls = [0]
        mock_box3.conf = [0.88]
        mock_box3.xyxy = [MagicMock(cpu=lambda: MagicMock(numpy=lambda: np.array([300, 40, 400, 220])))]

        mock_box4 = MagicMock()
        mock_box4.cls = [2]
        mock_box4.conf = [0.95]
        mock_box4.xyxy = [MagicMock(cpu=lambda: MagicMock(numpy=lambda: np.array([50, 300, 200, 450])))]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box1, mock_box2, mock_box3, mock_box4]
        mock_model.return_value = [mock_result]
        mock_model.track.return_value = [mock_result]

        from rtsp_pipeline.object_analyzer import ObjectAnalyzer

        analyzer = ObjectAnalyzer(model_weights="yolov8n.pt", confidence_threshold=0.5, crowd_threshold=3)

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections, person_count = analyzer.analyze(dummy_frame)

        self.assertEqual(person_count, 3)
        # Should have 4 object detections + 1 crowd detection (total 5)
        self.assertEqual(len(detections), 5)

        # Check standard object detection fields
        person_dets = [d for d in detections if d["label"] == "person"]
        self.assertEqual(len(person_dets), 3)
        self.assertIn("confidence", person_dets[0])
        self.assertIn("colour", person_dets[0])
        self.assertIn("bbox", person_dets[0])
        self.assertIsNone(person_dets[0]["colour"])

        # Check derived crowd detection entry
        crowd_dets = [d for d in detections if d["label"] == "crowd"]
        self.assertEqual(len(crowd_dets), 1)
        self.assertEqual(crowd_dets[0]["confidence"], 3.0)
        self.assertIsNone(crowd_dets[0]["bbox"])


if __name__ == "__main__":
    unittest.main()
