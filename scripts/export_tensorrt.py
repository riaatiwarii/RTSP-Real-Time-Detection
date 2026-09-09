"""Utility Script to Export PyTorch YOLO Checkpoints to NVIDIA TensorRT FP16 Engines.

Usage:
    python scripts/export_tensorrt.py --model yolov8m.pt --imgsz 1280
    python scripts/export_tensorrt.py --model yolov8n.pt --half --device 0
"""

import argparse
import logging
import sys
import torch

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics package is required to run export_tensorrt.py. Install via `pip install ultralytics`.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ExportTensorRT")


def export_engine(model_path: str, imgsz: int = 1280, half: bool = True, device: str = "0") -> str:
    """Export YOLO model checkpoint to NVIDIA TensorRT FP16 engine file format.

    Args:
        model_path: Path to PyTorch model (.pt).
        imgsz: Inference input resolution dimension (default: 1280).
        half: Enable FP16 half precision quantization (default: True).
        device: CUDA GPU device ID (default: "0").

    Returns:
        Path to exported .engine model file.
    """
    if not torch.cuda.is_available():
        logger.error("CUDA GPU is not available on this environment. TensorRT export requires NVIDIA GPU & CUDA.")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("STARTING NVIDIA TENSORRT ENGINE EXPORT")
    logger.info("Source Checkpoint : %s", model_path)
    logger.info("Inference Imgsz   : %d", imgsz)
    logger.info("FP16 Half Precision: %s", half)
    logger.info("Target Device     : %s", device)
    logger.info("=" * 60)

    try:
        model = YOLO(model_path)
        engine_path = model.export(
            format="engine",
            imgsz=imgsz,
            half=half,
            device=device,
            verbose=True,
        )
        logger.info("TensorRT export completed successfully! Engine file: %s", engine_path)
        return str(engine_path)
    except Exception as exc:
        logger.error("TensorRT engine export failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch YOLO models to NVIDIA TensorRT FP16 engine format.")
    parser.add_argument("--model", type=str, default="yolov8m.pt", help="Path to PyTorch .pt model file")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference resolution dimension (default: 1280)")
    parser.add_argument("--half", action="store_true", default=True, help="Enable FP16 half precision (default: True)")
    parser.add_argument("--device", type=str, default="0", help="CUDA GPU device ID (default: '0')")

    args = parser.parse_args()
    export_engine(model_path=args.model, imgsz=args.imgsz, half=args.half, device=args.device)
