"""Unit test for Stage 1 RTSPCapture class using a generated local test video."""

import os
import sys
import tempfile
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

        # Create synthetic video (20 frames, 640x480)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.video_path, fourcc, 10.0, (640, 480))
        for i in range(20):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:, :] = (i * 10, 100, 200)
            writer.write(frame)
        writer.release()

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_capture_reads_frames(self) -> None:
        """Verify reading frames from a video source."""
        capture = RTSPCapture(self.video_path)
        self.assertTrue(capture.connect())
        self.assertTrue(capture.is_connected)

        frames_read = 0
        for _ in range(15):
            ret, frame = capture.read_frame()
            if ret and frame is not None:
                frames_read += 1
                self.assertEqual(frame.shape, (480, 640, 3))

        self.assertGreater(frames_read, 0)
        capture.release()
        self.assertFalse(capture.is_connected)

    def test_invalid_stream_reconnect_backoff(self) -> None:
        """Verify reconnect attempts when stream URL is invalid."""
        capture = RTSPCapture(
            rtsp_url="rtsp://invalid_host_1234:8554/live",
            initial_reconnect_delay=0.1,
            max_reconnect_interval=0.4,
            max_retries=2,
        )
        self.assertFalse(capture.connect())
        # Attempt frame read which should trigger backoff and eventually fail
        ret, frame = capture.read_frame()
        self.assertFalse(ret)
        self.assertIsNone(frame)


if __name__ == "__main__":
    unittest.main()
