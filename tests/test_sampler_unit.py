"""Unit test for Stage 2 FrameSampler class."""

import os
import sys
import tempfile
import time
import unittest

# Ensure project root is in sys.path when script is executed directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import cv2
from rtsp_pipeline.capture import RTSPCapture
from rtsp_pipeline.sampler import FrameSampler


class TestFrameSampler(unittest.TestCase):
    """Unit tests for FrameSampler time-interval throttling logic."""

    def setUp(self) -> None:
        """Create a synthetic test video."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.video_path = os.path.join(self.temp_dir.name, "test_stream.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.video_path, fourcc, 10.0, (320, 240))
        for i in range(30):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

        self.capture = RTSPCapture(self.video_path)
        self.capture.start()
        time.sleep(0.2)

    def tearDown(self) -> None:
        self.capture.stop()
        self.temp_dir.cleanup()

    def test_sampler_throttling(self) -> None:
        """Verify sampling throttles based on target FPS."""
        sampler = FrameSampler(capture=self.capture, target_fps=5.0)  # interval = 0.2s

        accepted_count = 0
        start_t = time.time()
        while time.time() - start_t < 1.0:
            accepted, frame, ts = sampler.sample_latest()
            if accepted:
                accepted_count += 1
                self.assertIsNotNone(frame)
            time.sleep(0.02)

        # In 1.0s at 5 FPS, we expect ~5 accepted frames (+/- 1 due to timing jitter)
        self.assertGreaterEqual(accepted_count, 3)
        self.assertLessEqual(accepted_count, 7)

    def test_invalid_target_fps(self) -> None:
        """Verify ValueError raised for zero or negative target FPS."""
        with self.assertRaises(ValueError):
            FrameSampler(capture=self.capture, target_fps=0.0)
        with self.assertRaises(ValueError):
            FrameSampler(capture=self.capture, target_fps=-5.0)


if __name__ == "__main__":
    unittest.main()
