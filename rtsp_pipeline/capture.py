"""RTSP Video Stream Capture Module.

Provides robust OpenCV VideoCapture wrapper with exponential backoff reconnect logic
for handling unstable network streams and RTSP dropouts.
"""

import logging
import time
from typing import Optional, Tuple
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class RTSPCapture:
    """Manages an RTSP video stream with automatic reconnect and exponential backoff."""

    def __init__(
        self,
        rtsp_url: str,
        initial_reconnect_delay: float = 1.0,
        max_reconnect_interval: float = 32.0,
        max_retries: Optional[int] = None,
    ) -> None:
        """Initialize the RTSP capture instance.

        Args:
            rtsp_url: RTSP stream connection URL.
            initial_reconnect_delay: Initial wait time in seconds before first reconnect retry.
            max_reconnect_interval: Maximum cap for exponential backoff delay in seconds.
            max_retries: Maximum number of reconnection attempts before giving up (None for infinite).
        """
        self.rtsp_url = rtsp_url
        self.initial_reconnect_delay = initial_reconnect_delay
        self.max_reconnect_interval = max_reconnect_interval
        self.max_retries = max_retries

        self._cap: Optional[cv2.VideoCapture] = None
        self._is_connected = False
        self._is_running = False

    @property
    def is_connected(self) -> bool:
        """Return True if currently connected to the stream."""
        return self._is_connected and self._cap is not None and self._cap.isOpened()

    def connect(self) -> bool:
        """Attempt initial connection to the RTSP stream.

        Returns:
            True if connection succeeded, False otherwise.
        """
        logger.info("Connecting to RTSP stream: %s", self.rtsp_url)
        if self._cap is not None:
            self._cap.release()

        self._cap = cv2.VideoCapture(self.rtsp_url)
        if self._cap.isOpened():
            self._is_connected = True
            self._is_running = True
            logger.info("Successfully connected to RTSP stream.")
            return True

        self._is_connected = False
        logger.error("Failed to open RTSP stream: %s", self.rtsp_url)
        return False

    def _reconnect(self) -> bool:
        """Attempt stream reconnection using exponential backoff.

        Returns:
            True if reconnection succeeded, False if retries exhausted.
        """
        self._is_connected = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None

        logger.warning("Stream disconnected. Initiating reconnection backoff loop...")
        delay = self.initial_reconnect_delay
        attempts = 0

        while self._is_running:
            attempts += 1
            if self.max_retries is not None and attempts > self.max_retries:
                logger.error("Exceeded maximum reconnection attempts (%d). Giving up.", self.max_retries)
                return False

            logger.info("Reconnection attempt #%d (waiting %.1fs)...", attempts, delay)
            time.sleep(delay)

            try:
                self._cap = cv2.VideoCapture(self.rtsp_url)
                if self._cap.isOpened():
                    self._is_connected = True
                    logger.info("Reconnected to RTSP stream successfully on attempt #%d.", attempts)
                    return True
            except Exception as exc:
                logger.warning("Exception during reconnect attempt #%d: %s", attempts, exc)

            # Exponential backoff capped at max_reconnect_interval
            delay = min(delay * 2.0, self.max_reconnect_interval)

        return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the next raw frame from the RTSP stream.

        If a frame read fails due to stream drop, automatically triggers reconnection.

        Returns:
            Tuple of (success_flag, frame_bgr_numpy_array).
        """
        if not self._is_running:
            return False, None

        if not self.is_connected:
            if not self._reconnect():
                return False, None

        try:
            ret, frame = self._cap.read()
            if ret and frame is not None and frame.size > 0:
                return True, frame

            logger.warning("cv2.VideoCapture read returned ret=False or empty frame.")
        except Exception as exc:
            logger.error("Error reading frame from RTSP stream: %s", exc)

        # Trigger reconnection if frame reading failed
        if self._reconnect():
            return self.read_frame()

        return False, None

    def release(self) -> None:
        """Release the VideoCapture resource and close the stream connection."""
        self._is_running = False
        self._is_connected = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("RTSP capture released.")

    def __enter__(self) -> "RTSPCapture":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
