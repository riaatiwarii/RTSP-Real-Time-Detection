"""Unit test for Stage 2 FrameSampler class."""

import os
import sys
import unittest

# Ensure project root is in sys.path when script is executed directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from rtsp_pipeline.sampler import FrameSampler


class TestFrameSampler(unittest.TestCase):
    """Unit tests for FrameSampler time-interval throttling logic."""

    def test_target_fps_interval_throttling(self) -> None:
        """Verify sampling at 5 FPS accepts frames roughly every 0.2s."""
        sampler = FrameSampler(target_fps=5.0)  # interval = 0.2s
        start_t = 1000.0

        # Frame 0 at t=1000.0 (Accepted - first frame)
        self.assertTrue(sampler.should_sample(start_t))

        # Frame 1 at t=1000.05 (Discarded - elapsed 0.05s < 0.2s)
        self.assertFalse(sampler.should_sample(start_t + 0.05))

        # Frame 2 at t=1000.15 (Discarded - elapsed 0.15s < 0.2s)
        self.assertFalse(sampler.should_sample(start_t + 0.15))

        # Frame 3 at t=1000.21 (Accepted - elapsed 0.21s >= 0.2s)
        self.assertTrue(sampler.should_sample(start_t + 0.21))

        # Frame 4 at t=1000.30 (Discarded - elapsed 0.09s from last accepted)
        self.assertFalse(sampler.should_sample(start_t + 0.30))

        # Frame 5 at t=1000.41 (Accepted - elapsed 0.20s from last accepted)
        self.assertTrue(sampler.should_sample(start_t + 0.41))

        self.assertEqual(sampler.total_evaluated, 6)
        self.assertEqual(sampler.total_accepted, 3)

    def test_invalid_target_fps(self) -> None:
        """Verify ValueError raised for zero or negative target FPS."""
        with self.assertRaises(ValueError):
            FrameSampler(target_fps=0.0)
        with self.assertRaises(ValueError):
            FrameSampler(target_fps=-5.0)

    def test_process_frame(self) -> None:
        """Verify process_frame returns numpy array iff accepted."""
        sampler = FrameSampler(target_fps=10.0)  # interval = 0.1s
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        accepted, frame, ts = sampler.process_frame(dummy_frame, frame_timestamp=100.0)
        self.assertTrue(accepted)
        self.assertIsNotNone(frame)

        accepted, frame, ts = sampler.process_frame(dummy_frame, frame_timestamp=100.05)
        self.assertFalse(accepted)
        self.assertIsNone(frame)


if __name__ == "__main__":
    unittest.main()
