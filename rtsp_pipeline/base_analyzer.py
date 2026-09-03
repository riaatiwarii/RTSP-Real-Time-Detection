"""Base Analyzer / Preprocessor Module.

Provides model-specific frame preprocessing transformations:
- YOLO path: Passthrough (Ultralytics YOLO handles letterboxing, RGB conversion, and tensor formatting internally).
- InsightFace path: Aspect-ratio preserving letterbox resize, BGR->RGB conversion, and metadata tracking.
"""

from typing import Any, Dict, Tuple
import cv2
import numpy as np


class BaseAnalyzer:
    """Handles image preprocessing transformations prior to model inference."""

    @staticmethod
    def preprocess_yolo(frame: np.ndarray) -> np.ndarray:
        """YOLO preprocessing path.

        Ultralytics YOLO manages image resizing, letterboxing, BGR-to-RGB conversion,
        and array normalization internally. This method passes raw BGR frame arrays through
        completely unmodified.

        Args:
            frame: Raw BGR input frame numpy array.

        Returns:
            Unmodified input numpy array.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid frame input: must be a non-empty numpy array.")
        return frame

    @staticmethod
    def letterbox(
        frame: np.ndarray,
        target_size: Tuple[int, int] = (640, 640),
        color: Tuple[int, int, int] = (114, 114, 114),
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform aspect-ratio preserving letterbox resize with padding.

        Args:
            frame: Input frame numpy array (H, W, C).
            target_size: Target (width, height) tuple (default: 640x640).
            color: Fill color tuple for padding borders (default: gray 114).

        Returns:
            Tuple of (letterboxed_image_array, letterbox_metadata_dict).
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid frame input: must be a non-empty numpy array.")

        h, w = frame.shape[:2]
        target_w, target_h = target_size

        # Compute scale factor preserving aspect ratio
        scale = min(target_w / w, target_h / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        # Resize image
        if (w, h) != (new_w, new_h):
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = frame.copy()

        # Compute symmetric padding
        dw = target_w - new_w
        dh = target_h - new_h

        top = dh // 2
        bottom = dh - top
        left = dw // 2
        right = dw - left

        letterboxed = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=color,
        )

        meta = {
            "original_shape": (h, w),
            "target_size": target_size,
            "scale": scale,
            "pad_top": top,
            "pad_bottom": bottom,
            "pad_left": left,
            "pad_right": right,
        }

        return letterboxed, meta

    @staticmethod
    def preprocess_insightface(
        frame: np.ndarray,
        target_size: Tuple[int, int] = (640, 640),
        normalize: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """InsightFace preprocessing path.

        Performs:
        1. Aspect-ratio preserving letterbox resize.
        2. BGR to RGB color space conversion.
        3. Optional normalization (uint8 [0, 255] to float32 [0.0, 1.0]).

        Args:
            frame: Raw BGR frame numpy array (uint8).
            target_size: Desired model input dimensions (width, height).
            normalize: If True, normalizes output to float32 range [0.0, 1.0].

        Returns:
            Tuple of (preprocessed_rgb_frame, letterbox_metadata_dict).
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid frame input: must be a non-empty numpy array.")

        # 1. Aspect-ratio preserving letterbox resize
        letterboxed, meta = BaseAnalyzer.letterbox(frame, target_size=target_size)

        # 2. BGR -> RGB conversion
        rgb_frame = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)

        # 3. Optional normalization
        if normalize:
            normalized = rgb_frame.astype(np.float32) / 255.0
            return normalized, meta

        return rgb_frame, meta
