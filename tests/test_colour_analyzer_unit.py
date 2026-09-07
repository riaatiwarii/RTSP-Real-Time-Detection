"""Unit test suite for Stage 6 — Colour Analyzer."""

import os
import sys
import unittest

# Ensure project root is in sys.path when script is executed directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from rtsp_pipeline.colour_analyzer import ColourAnalyzer


class TestColourAnalyzer(unittest.TestCase):
    """Unit tests for ColourAnalyzer logic."""

    def setUp(self) -> None:
        self.analyzer = ColourAnalyzer()

    def test_dominant_colour_extraction(self) -> None:
        """Verify HSV color classification on synthetic color patches."""
        # Pure Red BGR (0, 0, 255)
        red_crop = np.zeros((50, 50, 3), dtype=np.uint8)
        red_crop[:, :] = (0, 0, 255)
        self.assertEqual(ColourAnalyzer.get_dominant_colour(red_crop), "red")

        # Pure Blue BGR (255, 0, 0)
        blue_crop = np.zeros((50, 50, 3), dtype=np.uint8)
        blue_crop[:, :] = (255, 0, 0)
        self.assertEqual(ColourAnalyzer.get_dominant_colour(blue_crop), "blue")

        # Pure Green BGR (0, 255, 0)
        green_crop = np.zeros((50, 50, 3), dtype=np.uint8)
        green_crop[:, :] = (0, 255, 0)
        self.assertEqual(ColourAnalyzer.get_dominant_colour(green_crop), "green")

        # Black patch (0, 0, 0)
        black_crop = np.zeros((50, 50, 3), dtype=np.uint8)
        self.assertEqual(ColourAnalyzer.get_dominant_colour(black_crop), "black")

        # White patch (255, 255, 255)
        white_crop = np.full((50, 50, 3), 255, dtype=np.uint8)
        self.assertEqual(ColourAnalyzer.get_dominant_colour(white_crop), "white")

    def test_enrich_detections(self) -> None:
        """Verify object detections are enriched with colour while faces/crowd are untouched."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        # Paint a blue box at (10, 10, 60, 60)
        frame[10:60, 10:60] = (255, 0, 0)

        detections = [
            {"label": "car", "confidence": 0.90, "colour": None, "bbox": (10, 10, 60, 60)},
            {"label": "face", "confidence": 0.95, "colour": None, "bbox": (70, 70, 120, 120)},
            {"label": "crowd", "confidence": 4.0, "colour": None, "bbox": None},
        ]

        enriched = self.analyzer.enrich_detections(frame, detections)

        # Car object should now have colour = "blue"
        self.assertEqual(enriched[0]["label"], "car")
        self.assertEqual(enriched[0]["colour"], "blue")

        # Face should remain colour = None
        self.assertEqual(enriched[1]["label"], "face")
        self.assertIsNone(enriched[1]["colour"])

        # Crowd should remain colour = None
        self.assertEqual(enriched[2]["label"], "crowd")
        self.assertIsNone(enriched[2]["colour"])


if __name__ == "__main__":
    unittest.main()
