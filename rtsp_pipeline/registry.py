"""Analyzer Registry Module.

Central dispatcher that executes all enabled analyzers (Object, Face, Colour)
and merges detection results into a unified flat List[Detection].
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
from rtsp_pipeline.base_analyzer import BaseAnalyzer
from rtsp_pipeline.colour_analyzer import ColourAnalyzer
from rtsp_pipeline.face_analyzer import FaceAnalyzer
from rtsp_pipeline.object_analyzer import ObjectAnalyzer

logger = logging.getLogger(__name__)


class AnalyzerRegistry:
    """Central dispatcher managing model analyzers and merging outputs."""

    def __init__(
        self,
        object_analyzer: Optional[ObjectAnalyzer] = None,
        face_analyzer: Optional[FaceAnalyzer] = None,
        colour_analyzer: Optional[ColourAnalyzer] = None,
        enable_object: bool = True,
        enable_face: bool = True,
        enable_colour: bool = False,
    ) -> None:
        """Initialize AnalyzerRegistry.

        Args:
            object_analyzer: Instance of ObjectAnalyzer (Stage 4).
            face_analyzer: Instance of FaceAnalyzer (Stage 5).
            colour_analyzer: Instance of ColourAnalyzer (Stage 6).
            enable_object: Flag to enable/disable object detection.
            enable_face: Flag to enable/disable face detection.
            enable_colour: Flag to enable/disable bounding box colour extraction.
        """
        self.object_analyzer = object_analyzer
        self.face_analyzer = face_analyzer
        self.colour_analyzer = colour_analyzer

        self.enable_object = enable_object
        self.enable_face = enable_face
        self.enable_colour = enable_colour

        logger.info(
            "AnalyzerRegistry initialized | Object: %s | Face: %s | Colour: %s",
            self.enable_object,
            self.enable_face,
            self.enable_colour,
        )

    def analyze_frame(self, raw_frame: np.ndarray) -> List[Dict[str, Any]]:
        """Process a raw video frame through all enabled analyzers and merge results.

        Args:
            raw_frame: Raw BGR frame numpy array.

        Returns:
            Flat list of Detection dictionaries:
            [{"label": str, "confidence": float, "colour": str | None}, ...]
        """
        if raw_frame is None or not isinstance(raw_frame, np.ndarray) or raw_frame.size == 0:
            return []

        merged_detections: List[Dict[str, Any]] = []

        # 1. Run Object & Crowd Analyzer (YOLO path: passthrough preprocessing)
        if self.enable_object and self.object_analyzer is not None:
            try:
                yolo_frame = BaseAnalyzer.preprocess_yolo(raw_frame)
                obj_dets, _ = self.object_analyzer.analyze(yolo_frame)
                merged_detections.extend(obj_dets)
            except Exception as exc:
                logger.error("Error executing ObjectAnalyzer in registry: %s", exc)

        # 2. Run Face Analyzer (InsightFace path: letterbox + RGB preprocessing)
        if self.enable_face and self.face_analyzer is not None:
            try:
                rgb_frame, meta = BaseAnalyzer.preprocess_insightface(raw_frame, target_size=(640, 640))
                face_dets = self.face_analyzer.analyze(rgb_frame, meta=meta)
                merged_detections.extend(face_dets)
            except Exception as exc:
                logger.error("Error executing FaceAnalyzer in registry: %s", exc)

        # 3. Run Colour Analyzer (if enabled, enriches object bboxes)
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
            final_detections.append(clean_det)

        return final_detections
