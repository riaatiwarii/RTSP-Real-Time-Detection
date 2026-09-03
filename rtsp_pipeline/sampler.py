"""Frame Sampler Module.

Provides time-interval based frame rate throttling (e.g., native ~24 FPS down to 4-5 FPS).
Time-based sampling self-corrects against stream FPS fluctuations unlike static N-frame skipping.
"""

import time
from typing import Tuple, Optional
import numpy as np


class FrameSampler:
    """Throttles incoming video frames to a target FPS based on elapsed time intervals."""

    def __init__(self, target_fps: float = 5.0) -> None:
        """Initialize the FrameSampler.

        Args:
            target_fps: Target frame rate per second to accept. Must be > 0.
        """
        if target_fps <= 0:
            raise ValueError(f"target_fps must be greater than 0, got {target_fps}")

        self.target_fps = target_fps
        self.sampling_interval = 1.0 / target_fps
        self.last_accepted_time: Optional[float] = None
        self._total_evaluated = 0
        self._total_accepted = 0

    def should_sample(self, frame_timestamp: Optional[float] = None) -> bool:
        """Determine whether the current frame should be accepted based on time elapsed.

        Args:
            frame_timestamp: Optional explicit timestamp (seconds). Uses time.time() if None.

        Returns:
            True if the frame is accepted (interval elapsed), False if discarded.
        """
        now = frame_timestamp if frame_timestamp is not None else time.time()
        self._total_evaluated += 1

        if self.last_accepted_time is None:
            self.last_accepted_time = now
            self._total_accepted += 1
            return True

        elapsed = now - self.last_accepted_time
        if elapsed >= self.sampling_interval:
            self.last_accepted_time = now
            self._total_accepted += 1
            return True

        return False

    def process_frame(
        self, frame: np.ndarray, frame_timestamp: Optional[float] = None
    ) -> Tuple[bool, Optional[np.ndarray], float]:
        """Evaluate a frame array and return sampling status.

        Args:
            frame: Raw image frame numpy array.
            frame_timestamp: Timestamp of frame arrival.

        Returns:
            Tuple of (accepted_flag, frame_if_accepted_else_None, current_timestamp).
        """
        now = frame_timestamp if frame_timestamp is not None else time.time()
        accepted = self.should_sample(now)
        if accepted:
            return True, frame, now
        return False, None, now

    def reset(self) -> None:
        """Reset internal sampling state timers and statistics."""
        self.last_accepted_time = None
        self._total_evaluated = 0
        self._total_accepted = 0

    @property
    def total_evaluated(self) -> int:
        """Total number of frames evaluated so far."""
        return self._total_evaluated

    @property
    def total_accepted(self) -> int:
        """Total number of frames accepted so far."""
        return self._total_accepted
