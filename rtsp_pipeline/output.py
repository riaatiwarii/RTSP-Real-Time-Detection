"""Output Layer Module.

Enforces conditional persistence: saves frame image and writes metadata JSONL record
ONLY if at least one detection occurred (len(detections) > 0). If zero detections, discards frame.
"""

from datetime import datetime
import json
import logging
import os
import threading
import uuid
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class OutputWriter:
    """Handles conditional frame image persistence and metadata JSONL logging."""

    def __init__(
        self,
        frames_dir: str = "output/frames",
        logs_dir: str = "output/logs",
        metadata_filename: str = "detections.jsonl",
    ) -> None:
        """Initialize OutputWriter and create target directories.

        Args:
            frames_dir: Directory path to save frame images.
            logs_dir: Directory path to save metadata logs.
            metadata_filename: JSONL log file name (default: 'detections.jsonl').
        """
        self.frames_dir = frames_dir
        self.logs_dir = logs_dir
        self.metadata_filename = metadata_filename
        self.metadata_file_path = os.path.join(self.logs_dir, self.metadata_filename)
        self._write_lock = threading.Lock()

        # Ensure directories exist
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        logger.info(
            "OutputWriter initialized | Frames Dir: %s | Logs File: %s",
            self.frames_dir,
            self.metadata_file_path,
        )

    def save_detection_result(
        self,
        frame_bgr: np.ndarray,
        detections: List[Dict[str, Any]],
        frame_timestamp: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Save frame image and append metadata record iff len(detections) > 0.

        Args:
            frame_bgr: Raw/sampled image numpy array (BGR).
            detections: List of Detection dicts: [{"label": str, "confidence": float, "colour": str | None}]
            frame_timestamp: Timestamp of frame arrival (seconds).

        Returns:
            Metadata record dict if saved, None if discarded (len(detections) == 0).
        """
        # STORAGE OPTIMIZATION SAVE RULE: Discard immediately if zero detections
        if not detections:
            logger.debug("Zero detections for frame. Discarding frame without I/O.")
            return None

        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
            logger.warning("Empty frame passed to OutputWriter. Skipping save.")
            return None

        now_dt = datetime.now()
        timestamp_str = now_dt.strftime("%Y%m%d_%H%M%S_%f")
        unique_id = uuid.uuid4().hex[:6]

        image_filename = f"frame_{timestamp_str}_{unique_id}.jpg"
        image_path = os.path.join(self.frames_dir, image_filename)

        # 1. Save frame image to disk
        try:
            saved = cv2.imwrite(image_path, frame_bgr)
            if not saved:
                logger.error("cv2.imwrite failed to save image to %s", image_path)
                return None
        except Exception as exc:
            logger.error("Exception saving frame image (%s): %s", image_path, exc)
            return None

        # 2. Build metadata record
        record = {
            "timestamp": now_dt.isoformat(),
            "frame_timestamp": frame_timestamp,
            "image_file": image_filename,
            "image_path": os.path.abspath(image_path),
            "detection_count": len(detections),
            "detections": detections,
        }

        # 3. Append to JSONL log file under lock
        try:
            json_line = json.dumps(record) + "\n"
            with self._write_lock:
                with open(self.metadata_file_path, "a", encoding="utf-8") as f:
                    f.write(json_line)
            logger.info(
                "Saved frame '%s' with %d detections to %s",
                image_filename,
                len(detections),
                self.metadata_file_path,
            )
            return record
        except Exception as exc:
            logger.error("Failed to write metadata JSONL record to %s: %s", self.metadata_file_path, exc)
            return record
