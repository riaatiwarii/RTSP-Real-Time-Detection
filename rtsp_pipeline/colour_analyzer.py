"""Colour Analyzer Module.

Extracts bounding box crop regions from object detections, converts crops to HSV,
computes dominant color via histogram/pixel analysis, and enriches existing Detection objects.
"""

import logging
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ColourAnalyzer:
    """Analyzes dominant color for bounding box crops of object detections."""

    # Standard HSV color boundaries (H: 0-180, S: 0-255, V: 0-255 in OpenCV)
    COLOR_RANGES = [
        ("red", np.array([0, 50, 50], dtype=np.uint8), np.array([10, 255, 255], dtype=np.uint8)),
        ("red", np.array([170, 50, 50], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8)),
        ("orange", np.array([11, 50, 50], dtype=np.uint8), np.array([25, 255, 255], dtype=np.uint8)),
        ("yellow", np.array([26, 50, 50], dtype=np.uint8), np.array([35, 255, 255], dtype=np.uint8)),
        ("green", np.array([36, 50, 50], dtype=np.uint8), np.array([85, 255, 255], dtype=np.uint8)),
        ("blue", np.array([86, 50, 50], dtype=np.uint8), np.array([125, 255, 255], dtype=np.uint8)),
        ("purple", np.array([126, 50, 50], dtype=np.uint8), np.array([145, 255, 255], dtype=np.uint8)),
        ("pink", np.array([146, 50, 50], dtype=np.uint8), np.array([169, 255, 255], dtype=np.uint8)),
    ]

    @classmethod
    def get_dominant_colour(cls, crop_bgr: np.ndarray) -> Optional[str]:
        """Compute the dominant color name from a BGR crop numpy array.

        Args:
            crop_bgr: BGR crop image array (H, W, 3).

        Returns:
            Dominant color name string (e.g. 'red', 'blue', 'black', 'white') or None if crop invalid.
        """
        if crop_bgr is None or not isinstance(crop_bgr, np.ndarray) or crop_bgr.size == 0:
            return None

        h, w = crop_bgr.shape[:2]
        if h < 2 or w < 2:
            return None

        # Convert crop from BGR to HSV
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        total_pixels = h * w

        # Check Achromatic Colors (Black, White, Gray) based on S and V channels
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]

        black_mask = v_channel < 45
        white_mask = (s_channel < 35) & (v_channel > 200)
        gray_mask = (s_channel < 40) & (v_channel >= 45) & (v_channel <= 200)

        black_count = int(np.sum(black_mask))
        white_count = int(np.sum(white_mask))
        gray_count = int(np.sum(gray_mask))

        # Check Chromatic Colors
        color_counts: Dict[str, int] = {
            "black": black_count,
            "white": white_count,
            "gray": gray_count,
        }

        for color_name, lower_hsv, upper_hsv in cls.COLOR_RANGES:
            mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
            count = int(np.sum(mask > 0))
            color_counts[color_name] = color_counts.get(color_name, 0) + count

        # Find color with maximum matching pixel count
        dominant_color = max(color_counts, key=color_counts.get)
        max_count = color_counts[dominant_color]

        if max_count == 0 or (max_count / total_pixels) < 0.1:
            return "unknown"

        return dominant_color

    def enrich_detections(
        self,
        frame_bgr: np.ndarray,
        detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Enrich object detections in-place by computing dominant color for each bbox.

        Does not modify 'face' or 'crowd' detections or detections without a valid bbox.

        Args:
            frame_bgr: Original full raw frame array (BGR).
            detections: List of Detection dicts from analyzers.

        Returns:
            Enriched list of Detection dicts.
        """
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
            return detections

        img_h, img_w = frame_bgr.shape[:2]

        for det in detections:
            label = det.get("label")
            bbox = det.get("bbox")

            # Skip face and crowd entries or detections without bbox
            if label in ("face", "crowd") or bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            # Clamp bounding box coordinates to image boundaries
            x1 = max(0, min(int(x1), img_w - 1))
            y1 = max(0, min(int(y1), img_h - 1))
            x2 = max(0, min(int(x2), img_w))
            y2 = max(0, min(int(y2), img_h))

            if (x2 - x1) < 2 or (y2 - y1) < 2:
                det["colour"] = None
                continue

            # Crop region from original BGR frame
            crop = frame_bgr[y1:y2, x1:x2]
            colour_name = self.get_dominant_colour(crop)
            det["colour"] = colour_name

        return detections
