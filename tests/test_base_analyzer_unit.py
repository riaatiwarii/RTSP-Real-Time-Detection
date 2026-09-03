"""Unit test suite for Stage 3 — Base Analyzer (preprocessing)."""

import os
import sys
import unittest

# Ensure project root is in sys.path when script is executed directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import cv2
from rtsp_pipeline.base_analyzer import BaseAnalyzer


class TestBaseAnalyzer(unittest.TestCase):
    """Unit tests for model-specific preprocessing in BaseAnalyzer."""

    def setUp(self) -> None:
        """Create synthetic test frames."""
        # 1920x1080 BGR synthetic frame
        self.frame_1080p = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Blue rectangle at top-left
        self.frame_1080p[:100, :100] = (255, 0, 0)  # BGR: Pure Blue

        # 2592x1904 BGR synthetic frame (user's camera resolution)
        self.frame_highres = np.zeros((1904, 2592, 3), dtype=np.uint8)
        self.frame_highres[:100, :100] = (0, 0, 255)  # BGR: Pure Red

    def test_yolo_preprocessing_passthrough(self) -> None:
        """Verify YOLO path passes raw frame through completely unchanged."""
        processed = BaseAnalyzer.preprocess_yolo(self.frame_1080p)

        # Assert shape, dtype, and color channels are identical
        self.assertEqual(processed.shape, (1080, 1920, 3))
        self.assertEqual(processed.dtype, np.uint8)
        np.testing.assert_array_equal(processed, self.frame_1080p)

        # Assert identity/same memory reference or exact values
        self.assertTrue(np.array_equal(processed[:10, :10, 0], 255))  # Blue channel BGR

    def test_insightface_preprocessing_letterbox_and_rgb(self) -> None:
        """Verify InsightFace path performs letterbox, BGR->RGB, and shape target."""
        target_size = (640, 640)
        rgb_frame, meta = BaseAnalyzer.preprocess_insightface(self.frame_1080p, target_size=target_size)

        # Assert output shape matches (height, width, channels) -> (640, 640, 3)
        self.assertEqual(rgb_frame.shape, (640, 640, 3))
        self.assertEqual(rgb_frame.dtype, np.uint8)

        # Check color conversion: BGR (255, 0, 0) -> RGB (0, 0, 255)
        # The blue box is at top-left of the original frame; after padding, check inside scaled box region
        pad_top = meta["pad_top"]
        pad_left = meta["pad_left"]
        scale = meta["scale"]

        # Assert metadata properties
        self.assertEqual(meta["original_shape"], (1080, 1920))
        self.assertAlmostEqual(scale, 640 / 1920, places=4)
        self.assertEqual(meta["pad_left"], 0)
        self.assertGreater(meta["pad_top"], 0)

        # In RGB, top-left colored pixel should have Red=0, Green=0, Blue=255
        sample_pixel = rgb_frame[pad_top + 5, pad_left + 5]
        self.assertEqual(sample_pixel[0], 0)    # R
        self.assertEqual(sample_pixel[1], 0)    # G
        self.assertEqual(sample_pixel[2], 255)  # B

    def test_insightface_normalization(self) -> None:
        """Verify InsightFace normalization returns float32 array in [0.0, 1.0]."""
        norm_frame, meta = BaseAnalyzer.preprocess_insightface(
            self.frame_1080p, target_size=(640, 640), normalize=True
        )
        self.assertEqual(norm_frame.shape, (640, 640, 3))
        self.assertEqual(norm_frame.dtype, np.float32)
        self.assertLessEqual(norm_frame.max(), 1.0)
        self.assertGreaterEqual(norm_frame.min(), 0.0)

    def test_invalid_inputs(self) -> None:
        """Verify invalid frame arrays raise ValueError."""
        with self.assertRaises(ValueError):
            BaseAnalyzer.preprocess_yolo(None)
        with self.assertRaises(ValueError):
            BaseAnalyzer.preprocess_insightface(np.array([]))


if __name__ == "__main__":
    unittest.main()
