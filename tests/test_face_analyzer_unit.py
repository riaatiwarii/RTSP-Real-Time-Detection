"""Unit test suite for Stage 5 — Face Analyzer."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path when script is executed directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np


class TestFaceAnalyzer(unittest.TestCase):
    """Unit tests for FaceAnalyzer logic."""

    @patch("rtsp_pipeline.face_analyzer.FaceAnalysis")
    def test_face_detection_parsing_and_unletterboxing(self, mock_face_analysis_cls) -> None:
        """Verify face detection extraction, schema conformity, and coordinate mapping."""
        mock_app = MagicMock()
        mock_face_analysis_cls.return_value = mock_app

        # Mock face objects returned by InsightFace app.get()
        mock_face1 = MagicMock()
        mock_face1.det_score = 0.95
        mock_face1.bbox = np.array([100, 150, 200, 250])

        mock_face2 = MagicMock()
        mock_face2.det_score = 0.30  # Below 0.5 threshold
        mock_face2.bbox = np.array([300, 350, 400, 450])

        mock_app.get.return_value = [mock_face1, mock_face2]

        from rtsp_pipeline.face_analyzer import FaceAnalyzer

        analyzer = FaceAnalyzer(model_name="buffalo_l", confidence_threshold=0.5)

        dummy_rgb = np.zeros((640, 640, 3), dtype=np.uint8)

        # 1. Test without metadata (letterbox coords)
        detections = analyzer.analyze(dummy_rgb)
        self.assertEqual(len(detections), 1)
        face_det = detections[0]
        self.assertEqual(face_det["label"], "face")
        self.assertEqual(face_det["confidence"], 0.95)
        self.assertIsNone(face_det["colour"])
        self.assertEqual(face_det["bbox"], (100, 150, 200, 250))

        # 2. Test with metadata coordinate un-scaling
        meta = {
            "original_shape": (1080, 1920),
            "target_size": (640, 640),
            "scale": 0.3333,
            "pad_top": 140,
            "pad_left": 0,
        }
        mapped_dets = analyzer.analyze(dummy_rgb, meta=meta)
        self.assertEqual(len(mapped_dets), 1)
        mapped_bbox = mapped_dets[0]["bbox"]
        # x1 = (100 - 0) / 0.3333 = 300
        # y1 = max(0, (150 - 140) / 0.3333) = 30
        self.assertEqual(mapped_bbox[0], 300)
        self.assertEqual(mapped_bbox[1], 30)


if __name__ == "__main__":
    unittest.main()
