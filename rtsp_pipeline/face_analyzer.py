"""Face Analyzer Module using InsightFace (buffalo_l model).

Performs face detection and outputs standard Detection dictionary objects.
Conforms to the uniform Detection schema.
Default execution targets GPU (CUDAExecutionProvider) with CPU fallback.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import insightface
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)


class FaceAnalyzer:
    """InsightFace Face Analyzer for detecting faces."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        confidence_threshold: float = 0.5,
        input_size: Tuple[int, int] = (640, 640),
        providers: Optional[List[str]] = None,
    ) -> None:
        """Initialize FaceAnalyzer and load InsightFace model once.

        Args:
            model_name: Name of InsightFace model package (default: "buffalo_l").
            confidence_threshold: Minimum detection confidence threshold [0.0 - 1.0].
            input_size: Input resolution tuple (width, height) for detection model (default: (640, 640)).
            providers: ONNX Runtime Execution Providers list (default: ['CUDAExecutionProvider', 'CPUExecutionProvider']).
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        self.providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]

        logger.info("Initializing InsightFace FaceAnalysis (%s) with providers %s...", self.model_name, self.providers)
        try:
            self.app = FaceAnalysis(name=self.model_name, providers=self.providers)
            self.app.prepare(ctx_id=0, det_size=self.input_size)
            logger.info("Successfully loaded InsightFace (%s) model.", self.model_name)
        except Exception as exc:
            logger.error("Failed to initialize InsightFace model (%s): %s", self.model_name, exc)
            raise

    def analyze(
        self,
        frame_rgb: np.ndarray,
        meta: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Run face detection on a preprocessed RGB frame.

        Args:
            frame_rgb: Preprocessed RGB frame numpy array.
            meta: Optional letterbox metadata from BaseAnalyzer to un-pad and un-scale coordinates.

        Returns:
            List of standard Detection dictionaries: {"label": "face", "confidence": float, "colour": None, "bbox": (x1, y1, x2, y2)}.
        """
        if frame_rgb is None or not isinstance(frame_rgb, np.ndarray) or frame_rgb.size == 0:
            raise ValueError("Invalid frame input: must be a non-empty numpy array.")

        try:
            faces = self.app.get(frame_rgb)
        except Exception as exc:
            logger.error("Error during InsightFace inference: %s", exc)
            return []

        detections: List[Dict[str, Any]] = []
        if not faces:
            return detections

        for face in faces:
            confidence = float(face.det_score) if hasattr(face, "det_score") else 0.0
            if confidence < self.confidence_threshold:
                continue

            bbox_raw = face.bbox.astype(int)
            x1, y1, x2, y2 = int(bbox_raw[0]), int(bbox_raw[1]), int(bbox_raw[2]), int(bbox_raw[3])

            # Map coordinates back to original frame dimensions if letterbox metadata provided
            if meta and "scale" in meta and meta["scale"] > 0:
                scale = meta["scale"]
                pad_l = meta.get("pad_left", 0)
                pad_t = meta.get("pad_top", 0)

                x1 = max(0, int((x1 - pad_l) / scale))
                y1 = max(0, int((y1 - pad_t) / scale))
                x2 = int((x2 - pad_l) / scale)
                y2 = int((y2 - pad_t) / scale)

            detection = {
                "label": "face",
                "confidence": round(confidence, 4),
                "colour": None,  # Face detections do not have object colour
                "bbox": (x1, y1, x2, y2),
            }
            detections.append(detection)

        return detections
