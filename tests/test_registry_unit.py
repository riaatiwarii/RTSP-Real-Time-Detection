"""Unit test suite for Stage 7 — AnalyzerRegistry."""

import os
import sys
import unittest
from unittest.mock import MagicMock

# Ensure project root is in sys.path when script is executed directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from rtsp_pipeline.registry import AnalyzerRegistry


class TestAnalyzerRegistry(unittest.TestCase):
    """Unit tests for AnalyzerRegistry central dispatcher."""

    def test_merged_detections_and_schema_cleanup(self) -> None:
        """Verify merging object and face detections and removing internal bbox fields."""
        # Mock Object Analyzer
        mock_obj_analyzer = MagicMock()
        mock_obj_analyzer.analyze.return_value = (
            [
                {"label": "person", "confidence": 0.92, "colour": None, "bbox": (10, 10, 50, 50)},
                {"label": "car", "confidence": 0.88, "colour": None, "bbox": (100, 100, 200, 200)},
            ],
            1,
        )

        # Mock Face Analyzer
        mock_face_analyzer = MagicMock()
        mock_face_analyzer.analyze.return_value = [
            {"label": "face", "confidence": 0.96, "colour": None, "bbox": (15, 15, 45, 45)}
        ]

        registry = AnalyzerRegistry(
            object_analyzer=mock_obj_analyzer,
            face_analyzer=mock_face_analyzer,
            enable_object=True,
            enable_face=True,
            enable_colour=False,
        )

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        final_detections = registry.analyze_frame(dummy_frame)

        self.assertEqual(len(final_detections), 3)

        # Verify strict output schema: {"label": str, "confidence": float, "colour": str | None}
        for det in final_detections:
            self.assertIn("label", det)
            self.assertIn("confidence", det)
            self.assertIn("colour", det)
            self.assertIn("bbox", det)  # Retained bbox in final output record

        labels = [d["label"] for d in final_detections]
        self.assertIn("person", labels)
        self.assertIn("car", labels)
        self.assertIn("face", labels)

    def test_analyzer_toggle_flags(self) -> None:
        """Verify disabling object or face analyzer via registry flags."""
        mock_obj_analyzer = MagicMock()
        mock_obj_analyzer.analyze.return_value = ([{"label": "car", "confidence": 0.90, "colour": None, "bbox": (0, 0, 10, 10)}], 0)

        mock_face_analyzer = MagicMock()
        mock_face_analyzer.analyze.return_value = [{"label": "face", "confidence": 0.95, "colour": None, "bbox": (0, 0, 10, 10)}]

        # Enable face only
        registry_face_only = AnalyzerRegistry(
            object_analyzer=mock_obj_analyzer,
            face_analyzer=mock_face_analyzer,
            enable_object=False,
            enable_face=True,
            enable_colour=False,
        )

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = registry_face_only.analyze_frame(dummy_frame)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["label"], "face")
        mock_obj_analyzer.analyze.assert_not_called()


if __name__ == "__main__":
    unittest.main()
