"""Analyzer Registry Module.

Central dispatcher that executes all enabled analyzers (Object, Face, Colour)
and merges detection results into a unified flat List[Detection].
"""

import concurrent.futures
import logging
from typing import Any, Dict, List, Optional
import numpy as np
from rtsp_pipeline.base_analyzer import BaseAnalyzer
from rtsp_pipeline.colour_analyzer import ColourAnalyzer
from rtsp_pipeline.face_analyzer import FaceAnalyzer
from rtsp_pipeline.object_analyzer import ObjectAnalyzer

logger = logging.getLogger(__name__)


class AnalyzerRegistry:
    """Central dispatcher managing model analyzers and merging outputs via ThreadPoolExecutor."""

    def __init__(
        self,
        object_analyzer: Optional[ObjectAnalyzer] = None,
        face_analyzer: Optional[FaceAnalyzer] = None,
        colour_analyzer: Optional[ColourAnalyzer] = None,
        enable_object: bool = True,
        enable_face: bool = True,
        enable_colour: bool = False,
        max_workers: int = 4,
    ) -> None:
        """Initialize AnalyzerRegistry.

        Args:
            object_analyzer: Instance of ObjectAnalyzer (Stage 4).
            face_analyzer: Instance of FaceAnalyzer (Stage 5).
            colour_analyzer: Instance of ColourAnalyzer (Stage 6).
            enable_object: Flag to enable/disable object detection.
            enable_face: Flag to enable/disable face detection.
            enable_colour: Flag to enable/disable bounding box colour extraction.
            max_workers: Maximum threads for ThreadPoolExecutor.
        """
        self.object_analyzer = object_analyzer
        self.face_analyzer = face_analyzer
        self.colour_analyzer = colour_analyzer

        self.enable_object = enable_object
        self.enable_face = enable_face
        self.enable_colour = enable_colour

        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="ModelWorker"
        )

        logger.info(
            "AnalyzerRegistry initialized with ThreadPoolExecutor (%d workers) | Object: %s | Face: %s | Colour: %s",
            max_workers,
            self.enable_object,
            self.enable_face,
            self.enable_colour,
        )

    def _run_object_analysis(self, raw_frame: np.ndarray) -> List[Dict[str, Any]]:
        """Worker task for ObjectAnalyzer."""
        if not self.enable_object or self.object_analyzer is None:
            return []
        try:
            yolo_frame = BaseAnalyzer.preprocess_yolo(raw_frame)
            obj_dets, _ = self.object_analyzer.analyze(yolo_frame)
            return obj_dets
        except Exception as exc:
            logger.error("Error executing ObjectAnalyzer worker thread: %s", exc)
            return []

    def _run_face_analysis(self, raw_frame: np.ndarray) -> List[Dict[str, Any]]:
        """Worker task for FaceAnalyzer."""
        if not self.enable_face or self.face_analyzer is None:
            return []
        try:
            rgb_frame, meta = BaseAnalyzer.preprocess_insightface(raw_frame, target_size=(640, 640))
            face_dets = self.face_analyzer.analyze(rgb_frame, meta=meta)
            return face_dets
        except Exception as exc:
            logger.error("Error executing FaceAnalyzer worker thread: %s", exc)
            return []

    def analyze_frame(self, raw_frame: np.ndarray) -> List[Dict[str, Any]]:
        """Process a raw video frame concurrently through enabled analyzers and merge results.

        Args:
            raw_frame: Raw BGR frame numpy array.

        Returns:
            Flat list of Detection dictionaries:
            [{"label": str, "confidence": float, "colour": str | None, ...}, ...]
        """
        if raw_frame is None or not isinstance(raw_frame, np.ndarray) or raw_frame.size == 0:
            return []

        merged_detections: List[Dict[str, Any]] = []
        futures = []

        # 1 & 2. Dispatch Object & Face Analyzers in parallel to ThreadPoolExecutor
        if self.enable_object and self.object_analyzer is not None:
            futures.append(self.executor.submit(self._run_object_analysis, raw_frame))

        if self.enable_face and self.face_analyzer is not None:
            futures.append(self.executor.submit(self._run_face_analysis, raw_frame))

        # Wait for parallel model executions to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                res = future.result()
                if res:
                    merged_detections.extend(res)
            except Exception as exc:
                logger.error("Analyzer task execution failed: %s", exc)

        # 3. Run Colour Analyzer sequentially on merged bounding boxes if enabled
        if self.enable_colour and self.colour_analyzer is not None and merged_detections:
            try:
                merged_detections = self.colour_analyzer.enrich_detections(raw_frame, merged_detections)
            except Exception as exc:
                logger.error("Error executing ColourAnalyzer in registry: %s", exc)

        # 4. Clean up internal bounding box field for final output schema conformity
        final_detections: List[Dict[str, Any]] = []
        for det in merged_detections:
            clean_det = {
                "label": str(det["label"]),
                "confidence": float(det["confidence"]),
                "colour": det.get("colour"),
            }
            if "bbox" in det and det["bbox"] is not None:
                clean_det["bbox"] = det["bbox"]
            if "track_id" in det and det["track_id"] is not None:
                clean_det["track_id"] = det["track_id"]
            final_detections.append(clean_det)

        return final_detections

    def shutdown(self) -> None:
        """Shutdown the worker thread pool executor."""
        logger.info("Shutting down AnalyzerRegistry ThreadPoolExecutor...")
        self.executor.shutdown(wait=True)

