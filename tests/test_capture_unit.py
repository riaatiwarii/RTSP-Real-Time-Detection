"""Unit test for Stage 1 RTSPCapture class using a generated local test video."""

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


class TestRTSPCapture(unittest.TestCase):
    """Unit tests for RTSPCapture."""

    def setUp(self) -> None:
        """Create a temporary synthetic MP4 video file."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.video_path = os.path.join(self.temp_dir.name, "test_stream.mp4")

        # Create synthetic video (30 frames, 640x480)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.video_path, fourcc, 10.0, (640, 480))
        for i in range(30):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:, :] = (i * 5, 100, 200)
            writer.write(frame)
        writer.release()

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_capture_reads_latest_frame(self) -> None:
        """Verify threaded reader connects and updates latest frame."""
        capture = RTSPCapture(self.video_path)
        connected = capture.start()
        self.assertTrue(connected)
        self.assertTrue(capture.is_connected)

        time.sleep(0.2)  # Give reader thread time to grab frame
        has_frame, frame, ts = capture.read_latest()
        self.assertTrue(has_frame)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertGreater(ts, 0.0)

        capture.stop()
        self.assertFalse(capture.is_connected)

    def test_invalid_stream_reconnect_backoff(self) -> None:
        """Verify reconnect attempts when stream URL is invalid."""
        capture = RTSPCapture(
            rtsp_url="rtsp://invalid_host_1234:8554/live",
            initial_reconnect_delay=0.1,
            max_reconnect_delay=0.4,
            max_retries=2,
        )
        connected = capture.start()
        self.assertFalse(connected)
        time.sleep(0.5)
        has_frame, frame, ts = capture.read_latest()
        self.assertFalse(has_frame)
        self.assertIsNone(frame)
        capture.stop()


if __name__ == "__main__":
    unittest.main()
