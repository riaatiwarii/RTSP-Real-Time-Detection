import json
import logging
import os
import sqlite3
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class OutputWriter:
    """Handles conditional frame image persistence, metadata JSONL logging, and SQLite database storage."""

    def __init__(
        self,
        frames_dir: str = "output/frames",
        logs_dir: str = "output/logs",
        metadata_filename: str = "detections.jsonl",
        db_filename: str = "detections.db",
        enable_sqlite: bool = True,
    ) -> None:
        """Initialize OutputWriter and create target directories and SQLite database.

        Args:
            frames_dir: Directory path to save frame images.
            logs_dir: Directory path to save metadata logs.
            metadata_filename: JSONL log file name (default: 'detections.jsonl').
            db_filename: SQLite database file name (default: 'detections.db').
            enable_sqlite: Flag to enable/disable SQLite database logging.
        """
        self.frames_dir = frames_dir
        self.logs_dir = logs_dir
        self.metadata_filename = metadata_filename
        self.db_filename = db_filename
        self.enable_sqlite = enable_sqlite

        self.metadata_file_path = os.path.join(self.logs_dir, self.metadata_filename)
        self.db_file_path = os.path.join(self.logs_dir, self.db_filename)
        self._write_lock = threading.Lock()

        # Ensure directories exist
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        if self.enable_sqlite:
            self._init_sqlite_db()

        logger.info(
            "OutputWriter initialized | Frames Dir: %s | JSONL File: %s | SQLite DB: %s",
            self.frames_dir,
            self.metadata_file_path,
            self.db_file_path if self.enable_sqlite else "Disabled",
        )

    def _init_sqlite_db(self) -> None:
        """Initialize SQLite tables for frame metadata and individual detection events."""
        try:
            with sqlite3.connect(self.db_file_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS detection_frames (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        frame_id TEXT UNIQUE,
                        timestamp TEXT NOT NULL,
                        frame_timestamp REAL,
                        image_file TEXT NOT NULL,
                        image_path TEXT NOT NULL,
                        detection_count INTEGER NOT NULL
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS detection_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        frame_id TEXT NOT NULL,
                        label TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        colour TEXT,
                        bbox_json TEXT,
                        track_id INTEGER,
                        FOREIGN KEY (frame_id) REFERENCES detection_frames (frame_id)
                    )
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_frames_ts ON detection_frames(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_label ON detection_events(label)")
                conn.commit()
            logger.info("SQLite database tables initialized at %s", self.db_file_path)
        except Exception as exc:
            logger.error("Failed to initialize SQLite database (%s): %s", self.db_file_path, exc)

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
        frame_id = f"{timestamp_str}_{unique_id}"

        image_filename = f"frame_{frame_id}.jpg"
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
            "frame_id": frame_id,
            "timestamp": now_dt.isoformat(),
            "frame_timestamp": frame_timestamp,
            "image_file": image_filename,
            "image_path": os.path.abspath(image_path),
            "detection_count": len(detections),
            "detections": detections,
        }

        # 3. Append to JSONL log file and SQLite database under lock
        with self._write_lock:
            # Save to JSONL
            try:
                json_line = json.dumps(record) + "\n"
                with open(self.metadata_file_path, "a", encoding="utf-8") as f:
                    f.write(json_line)
            except Exception as exc:
                logger.error("Failed to write metadata JSONL record to %s: %s", self.metadata_file_path, exc)

            # Save to SQLite DB
            if self.enable_sqlite:
                try:
                    with sqlite3.connect(self.db_file_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute(
                            """
                            INSERT INTO detection_frames 
                            (frame_id, timestamp, frame_timestamp, image_file, image_path, detection_count)
                            VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (
                                frame_id,
                                record["timestamp"],
                                frame_timestamp,
                                image_filename,
                                os.path.abspath(image_path),
                                len(detections),
                            ),
                        )
                        for det in detections:
                            cursor.execute(
                                """
                                INSERT INTO detection_events
                                (frame_id, label, confidence, colour, bbox_json, track_id)
                                VALUES (?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    frame_id,
                                    str(det.get("label", "unknown")),
                                    float(det.get("confidence", 0.0)),
                                    det.get("colour"),
                                    json.dumps(det.get("bbox")) if det.get("bbox") is not None else None,
                                    det.get("track_id"),
                                ),
                            )
                        conn.commit()
                except Exception as exc:
                    logger.error("Failed to insert record into SQLite DB: %s", exc)

        logger.info(
            "Saved frame '%s' with %d detections to disk, JSONL & SQLite",
            image_filename,
            len(detections),
        )
        return record

