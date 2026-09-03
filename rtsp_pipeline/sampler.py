"""Frame Sampler Module.

Provides time-interval based frame rate throttling pulling latest frames from RTSPCapture.
Ensures zero latency drift over extended pipeline runs.
"""

import logging
import time
from typing import Optional, Tuple
import numpy as np
from rtsp_pipeline.capture import RTSPCapture

logger = logging.getLogger(__name__)


class FrameSampler:
    """Throttles RTSPCapture frame consumption to a target FPS based on time intervals."""

    def __init__(self, capture: RTSPCapture, target_fps: float = 5.0) -> None:
        """Initialize FrameSampler.

        Args:
            capture: An instance of RTSPCapture.
            target_fps: Desired sampling rate in FPS (must be > 0).
        """
        if target_fps <= 0:
            raise ValueError(f"target_fps must be greater than 0, got {target_fps}")

        self.capture = capture
        self.target_fps = target_fps
        self.sampling_interval = 1.0 / target_fps
        self.last_accepted_time: Optional[float] = None

        self._total_checked = 0
        self._total_accepted = 0

    def sample_latest(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Check if time interval has elapsed and fetch the latest frame from capture if due.

        Returns:
            Tuple of (accepted_flag, frame_bgr_numpy_array, timestamp).
        """
        now = time.time()
        self._total_checked += 1

        # Check if interval elapsed
        if self.last_accepted_time is not None:
            elapsed = now - self.last_accepted_time
            if elapsed < self.sampling_interval:
                return False, None, now

        # Attempt to pull latest frame from RTSPCapture
        has_frame, frame, ts = self.capture.read_latest()
        if has_frame and frame is not None:
            self.last_accepted_time = now
            self._total_accepted += 1
            logger.debug("Accepted frame #%d at timestamp %.4fs", self._total_accepted, ts)
            return True, frame, ts

        return False, None, now

    def reset(self) -> None:
        """Reset internal sampling state and stats."""
        self.last_accepted_time = None
        self._total_checked = 0
        self._total_accepted = 0

    @property
    def total_checked(self) -> int:
        """Total number of sample checks performed."""
        return self._total_checked

    @property
    def total_accepted(self) -> int:
        """Total number of frames successfully accepted."""
        return self._total_accepted
