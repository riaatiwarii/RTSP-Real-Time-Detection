"""Object and Crowd Analyzer Module using pretrained YOLOv8.

Performs object detection (COCO classes) and derives person count / crowd metrics.
Outputs standardized Detection dictionary objects.
Default execution targets GPU (CUDA).
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
        model_weights: str = "yolov8m.pt",
        confidence_threshold: float = 0.35,
        crowd_threshold: int = 3,
        imgsz: int = 1280,
        device: str = "cuda",
        classes: Optional[List[int]] = None,
        person_only: bool = False,
    ) -> None:
        """Initialize ObjectAnalyzer and load pretrained YOLOv8 model weights once.

        Args:
            model_weights: Path or name of YOLOv8 checkpoint (default: "yolov8m.pt").
            confidence_threshold: Minimum confidence score to filter detections [0.0 - 1.0].
            crowd_threshold: Minimum person count required to trigger a "crowd" detection.
            imgsz: Target inference resolution dimension (default: 1280).
            device: Computing device ('cuda', 'cpu', '0', etc. default: 'cuda').
            classes: Optional list of class IDs to detect (e.g. [0] for person).
            person_only: If True, restricts YOLO detection strictly to person objects (class 0).
        """
        self.model_weights = model_weights
        self.confidence_threshold = confidence_threshold
        self.crowd_threshold = crowd_threshold
        self.imgsz = imgsz
        self.device = device
        self.person_only = person_only

        if self.person_only and classes is None:
            self.classes = [0]
        else:
            self.classes = classes

        logger.info("Loading YOLOv8 model (%s) at imgsz=%d on device '%s'...", self.model_weights, self.imgsz, self.device)
        try:
            self.model = YOLO(self.model_weights)
            logger.info("Successfully loaded YOLOv8 model on device '%s'.", self.device)
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

        infer_kwargs: Dict[str, Any] = {
            "conf": self.confidence_threshold,
            "imgsz": self.imgsz,
            "device": self.device,
            "verbose": False,
        }
        if self.classes is not None:
            infer_kwargs["classes"] = self.classes

        # Run inference specifying target device and high-resolution imgsz
        results = self.model(frame, **infer_kwargs)

        detections: List[Dict[str, Any]] = []
        person_count = 0

        if not results:
            return detections, person_count

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return detections, person_count

        for box in boxes:
            raw_cls = box.cls[0]
            raw_conf = box.conf[0]

            cls_id = int(raw_cls.item() if hasattr(raw_cls, "item") else raw_cls)
            label = self.model.names[cls_id] if hasattr(self.model, "names") and cls_id in self.model.names else str(cls_id)
            confidence = float(raw_conf.item() if hasattr(raw_conf, "item") else raw_conf)

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
