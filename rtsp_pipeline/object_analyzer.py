"""Object and Crowd Analyzer Module using pretrained YOLOv8.

Performs object detection (COCO classes) and derives person count / crowd metrics.
Outputs standardized Detection dictionary objects.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class ObjectAnalyzer:
    """YOLOv8 Object and Crowd Analyzer."""

    def __init__(
        self,
        model_weights: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        crowd_threshold: int = 3,
        device: str = "auto",
    ) -> None:
        """Initialize ObjectAnalyzer and load pretrained YOLOv8 model weights once.

        Args:
            model_weights: Path or name of YOLOv8 checkpoint (default: "yolov8n.pt").
            confidence_threshold: Minimum confidence score to filter detections [0.0 - 1.0].
            crowd_threshold: Minimum person count required to trigger a "crowd" detection.
            device: Computing device ('cpu', 'cuda', or 'auto').
        """
        self.model_weights = model_weights
        self.confidence_threshold = confidence_threshold
        self.crowd_threshold = crowd_threshold
        self.device = device

        logger.info("Loading YOLOv8 model (%s)...", self.model_weights)
        try:
            self.model = YOLO(self.model_weights)
            logger.info("Successfully loaded YOLOv8 model.")
        except Exception as exc:
            logger.error("Failed to load YOLOv8 model weights (%s): %s", self.model_weights, exc)
            raise

    def analyze(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], int]:
        """Run object detection on a preprocessed raw BGR frame.

        Args:
            frame: Preprocessed input image numpy array (BGR).

        Returns:
            Tuple of (list_of_detections, person_count).
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid frame input: must be a non-empty numpy array.")

        # Run inference (device is handled automatically or explicitly)
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
        person_count = 0

        if not results:
            return detections, person_count

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return detections, person_count

        for box in boxes:
            cls_id = int(box.cls[0].item())
            label = self.model.names[cls_id] if hasattr(self.model, "names") else str(cls_id)
            confidence = float(box.conf[0].item())

            # Bounding box coordinates (x1, y1, x2, y2)
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

            detection = {
                "label": label,
                "confidence": round(confidence, 4),
                "colour": None,  # Populated downstream by Stage 6 Colour Analyzer
                "bbox": (x1, y1, x2, y2),
            }
            detections.append(detection)

            if label == "person":
                person_count += 1

        # Derive crowd metric if person count meets/exceeds crowd threshold
        if person_count >= self.crowd_threshold:
            crowd_detection = {
                "label": "crowd",
                "confidence": float(person_count),
                "colour": None,
                "bbox": None,
            }
            detections.append(crowd_detection)
            logger.debug("Crowd threshold met: %d persons detected.", person_count)

        return detections, person_count
