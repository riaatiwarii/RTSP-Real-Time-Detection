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
        confidence_threshold: float = 0.60,
        iou_threshold: float = 0.30,
        crowd_threshold: int = 3,
        imgsz: int = 1280,
        device: str = "cuda",
        classes: Optional[List[int]] = None,
        person_only: bool = False,
        enable_tracking: bool = False,
        tracker_type: str = "bytetrack.yaml",
    ) -> None:
        """Initialize ObjectAnalyzer and load pretrained YOLOv8 model weights once.

        Args:
            model_weights: Path or name of YOLOv8 checkpoint (default: "yolov8m.pt").
            confidence_threshold: Minimum confidence score to filter detections [0.0 - 1.0].
            iou_threshold: Minimum IoU threshold for Non-Maximum Suppression (NMS).
            crowd_threshold: Minimum person count required to trigger a "crowd" detection.
            imgsz: Target inference resolution dimension (default: 1280).
            device: Computing device ('cuda', 'cpu', '0', etc. default: 'cuda').
            classes: Optional list of class IDs to detect (e.g. [0] for person).
            person_only: If True, restricts YOLO detection strictly to person objects (class 0).
            enable_tracking: If True, uses YOLO persistent multi-object tracking (model.track).
            tracker_type: Tracking algorithm config file (default: 'bytetrack.yaml').
        """
        self.model_weights = model_weights
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.crowd_threshold = crowd_threshold
        self.imgsz = imgsz
        self.device = device
        self.person_only = person_only
        self.enable_tracking = enable_tracking
        self.tracker_type = tracker_type

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
        """Run object detection / tracking on a preprocessed raw BGR frame.

        Args:
            frame: Preprocessed input image numpy array (BGR).

        Returns:
            Tuple of (list_of_detections, person_count).
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid frame input: must be a non-empty numpy array.")

        infer_kwargs: Dict[str, Any] = {
            "conf": self.confidence_threshold,
            "iou": self.iou_threshold,
            "imgsz": self.imgsz,
            "device": self.device,
            "verbose": False,
        }
        if self.classes is not None:
            infer_kwargs["classes"] = self.classes

        # Run inference using persistent tracking (model.track) or standard detection (model)
        results = None
        if self.enable_tracking:
            try:
                results = self.model.track(
                    frame,
                    persist=True,
                    tracker=self.tracker_type,
                    **infer_kwargs,
                )
            except Exception as exc:
                logger.debug("Tracking call fallback to standard inference: %s", exc)
                results = self.model(frame, **infer_kwargs)
        else:
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

            # Extract track_id if ByteTrack / tracker assigned an ID
            track_id: Optional[int] = None
            if hasattr(box, "id") and box.id is not None:
                try:
                    raw_id = box.id[0]
                    track_id = int(raw_id.item() if hasattr(raw_id, "item") else raw_id)
                except Exception:
                    track_id = None

            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

            detection = {
                "label": label,
                "confidence": round(confidence, 4),
                "track_id": track_id,
                "colour": None,  # Populated downstream by Stage 6 Colour Analyzer
                "bbox": (x1, y1, x2, y2),
            }
            detections.append(detection)

        # Strict post-processing Non-Maximum Suppression (NMS) to guarantee zero duplicates
        sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
        filtered_dets: List[Dict[str, Any]] = []

        for det in sorted_dets:
            bbox = det.get("bbox")
            if bbox is None:
                filtered_dets.append(det)
                continue

            is_duplicate = False
            for existing in filtered_dets:
                ex_bbox = existing.get("bbox")
                if ex_bbox is not None and existing.get("label") == det.get("label"):
                    # Calculate IoU between det and existing box
                    x1 = max(bbox[0], ex_bbox[0])
                    y1 = max(bbox[1], ex_bbox[1])
                    x2 = min(bbox[2], ex_bbox[2])
                    y2 = min(bbox[3], ex_bbox[3])
                    inter = max(0, x2 - x1) * max(0, y2 - y1)
                    if inter > 0:
                        area1 = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        area2 = (ex_bbox[2] - ex_bbox[0]) * (ex_bbox[3] - ex_bbox[1])
                        union = area1 + area2 - inter
                        iou = inter / union if union > 0 else 0.0
                        if iou > self.iou_threshold:
                            is_duplicate = True
                            break

            if not is_duplicate:
                filtered_dets.append(det)

        person_count = sum(1 for d in filtered_dets if d.get("label") == "person")

        # Derive crowd metric if person count meets/exceeds crowd threshold
        if person_count >= self.crowd_threshold:
            crowd_detection = {
                "label": "crowd",
                "confidence": float(person_count),
                "colour": None,
                "bbox": None,
            }
            filtered_dets.append(crowd_detection)
            logger.debug("Crowd threshold met: %d persons detected.", person_count)

        return filtered_dets, person_count
