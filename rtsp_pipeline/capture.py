"""RTSP Video Stream Capture Module with Lightweight Background Grabber.

Background thread continuously calls `cap.grab()` (fast buffer flushing without decoding).
`cap.retrieve()` (the actual frame decode step) is executed on demand inside `read_latest()`.
Thread safety around OpenCV VideoCapture operations is enforced via `threading.Lock()`.
"""

import logging
import threading
import time
from typing import Optional, Tuple
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class RTSPCapture:
    """Manages RTSP stream with background buffer grabber and exponential backoff reconnect."""

    def __init__(
        self,
        rtsp_url: str,
        initial_reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 30.0,
        max_retries: Optional[int] = None,
    ) -> None:
        """Initialize RTSPCapture.

        Args:
            rtsp_url: RTSP connection URL.
            initial_reconnect_delay: Initial reconnect delay in seconds (default: 1.0).
            max_reconnect_delay: Maximum delay cap for exponential backoff in seconds (default: 30.0).
            max_retries: Maximum reconnection attempts before stopping (None for infinite).
        """
        self.rtsp_url = rtsp_url
        self.initial_reconnect_delay = initial_reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay
        self.max_retries = max_retries

        self._cap: Optional[cv2.VideoCapture] = None
        self._cap_lock = threading.Lock()

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_timestamp: float = 0.0
        self._has_new_grab = False
        self._last_grab_timestamp: float = 0.0

        self._is_running = False
        self._is_connected = False
        self._thread: Optional[threading.Thread] = None

    @property
    def is_connected(self) -> bool:
        """Return True if stream is currently connected."""
        return self._is_connected

    @property
    def is_running(self) -> bool:
        """Return True if background grabber thread is active."""
        return self._is_running

    def start(self) -> bool:
        """Connect to stream and start the background grabber thread.

        Returns:
            True if initial connection succeeded, False otherwise.
        """
        if self._is_running:
            logger.warning("RTSPCapture is already running.")
            return self._is_connected

        self._is_running = True
        connected = self._connect_stream()

        # Launch background thread to continuously call cap.grab()
        self._thread = threading.Thread(target=self._grabber_loop, daemon=True, name="RTSPGrabberThread")
        self._thread.start()
        logger.info("Background grabber thread started.")
        return connected

    def _connect_stream(self) -> bool:
        """Open VideoCapture object to the RTSP stream under lock."""
        import os
        # Force OpenCV FFmpeg to use RTSP over TCP (prevents UDP packet drop & 30s timeout drops)
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

        logger.info("Connecting to RTSP stream (TCP mode): %s", self.rtsp_url)
        with self._cap_lock:
            if self._cap is not None:
                self._cap.release()

            try:
                self._cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                if self._cap.isOpened():
                    self._is_connected = True
                    logger.info("Successfully connected to RTSP stream.")
                    return True
            except Exception as exc:
                logger.error("Exception opening RTSP stream: %s", exc)

            self._is_connected = False
            logger.error("Failed to open RTSP stream: %s", self.rtsp_url)
            return False

    def _reconnect(self) -> bool:
        """Attempt stream reconnection using exponential backoff."""
        if not self._is_running:
            return False

        logger.warning("Stream connection dropped. Starting exponential backoff reconnect...")
        self._is_connected = False
        with self._cap_lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None

        delay = self.initial_reconnect_delay
        attempts = 0

        while self._is_running:
            attempts += 1
            if self.max_retries is not None and attempts > self.max_retries:
                logger.error("Exceeded maximum reconnection attempts (%d). Stopping grabber.", self.max_retries)
                return False

            logger.info("Reconnect attempt #%d (waiting %.1fs delay)...", attempts, delay)
            time.sleep(delay)

            if not self._is_running:
                return False

            if self._connect_stream():
                logger.info(
                    "Reconnection successful on attempt #%d. Resetting reconnect delay to %.1fs.",
                    attempts,
                    self.initial_reconnect_delay,
                )
                return True

            delay = min(delay * 2.0, self.max_reconnect_delay)

        return False

    def _grabber_loop(self) -> None:
        """Continuous background loop calling cap.grab() to flush internal buffers."""
        while self._is_running:
            if not self._is_connected:
                if not self._reconnect():
                    break
                continue

            with self._cap_lock:
                if self._cap is None or not self._cap.isOpened():
                    grabbed = False
                else:
                    try:
                        # cap.grab() is lightweight; flushes OpenCV/FFmpeg buffer without decoding
                        grabbed = self._cap.grab()
                    except Exception as exc:
                        logger.error("Exception during cap.grab(): %s", exc)
                        grabbed = False

            if grabbed:
                now = time.time()
                self._last_grab_timestamp = now
                self._has_new_grab = True
                # Small sleep to prevent 100% CPU tight spin when grab() succeeds instantly
                time.sleep(0.001)
            else:
                logger.warning("cap.grab() failed (stream drop or EOF). Triggering reconnect...")
                self._reconnect()

        logger.info("Background grabber thread exited.")

    def read_latest(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Decode and fetch the latest frame on demand.

        Only calls cap.retrieve() when downstream requests a frame.

        Returns:
            Tuple of (has_frame_flag, frame_bgr_numpy_array, frame_timestamp).
        """
        with self._cap_lock:
            if self._is_connected and self._cap is not None and self._has_new_grab:
                try:
                    ret, frame = self._cap.retrieve()
                    if ret and frame is not None and frame.size > 0:
                        self._latest_frame = frame
                        self._latest_timestamp = self._last_grab_timestamp
                        self._has_new_grab = False
                    else:
                        logger.warning("cap.retrieve() failed despite successful grab().")
                except Exception as exc:
                    logger.error("Exception during cap.retrieve(): %s", exc)

            if self._latest_frame is not None:
                return True, self._latest_frame.copy(), self._latest_timestamp

        return False, None, 0.0

    def stop(self) -> None:
        """Stop background thread and release RTSP capture resources."""
        logger.info("Stopping RTSP capture and background thread...")
        self._is_running = False
        self._is_connected = False

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)
            self._thread = None

        with self._cap_lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
        logger.info("RTSP capture stopped and resources released.")

    def release(self) -> None:
        """Alias for stop()."""
        self.stop()

    def __enter__(self) -> "RTSPCapture":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
