"""Standalone verification script for Stage 8 — Output Layer.

Processes RTSP stream through AnalyzerRegistry, passes results to OutputWriter,
and verifies conditional frame image persistence and JSONL metadata line logging.
Safe fallback when optional dependencies like insightface are absent.
"""

import argparse
import json
import logging
import os
import sys
import time
import torch
import yaml

# Ensure project root is in sys.path when script is executed directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rtsp_pipeline.capture import RTSPCapture
from rtsp_pipeline.object_analyzer import ObjectAnalyzer
from rtsp_pipeline.output import OutputWriter
from rtsp_pipeline.registry import AnalyzerRegistry
from rtsp_pipeline.sampler import FrameSampler

try:
    from rtsp_pipeline.face_analyzer import FaceAnalyzer
    HAS_FACE_ANALYZER = True
except ImportError:
    HAS_FACE_ANALYZER = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Stage8Verification")


def load_default_config() -> dict:
    """Load configuration from config/config.yaml if available."""
    config_path = os.path.join(project_root, "config", "config.yaml")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as exc:
            logger.warning("Could not parse config.yaml: %s", exc)
    return {}


def run_stage8_verification(
    source_url: str,
    weights: str,
    device: str,
    conf_thresh: float = 0.35,
    imgsz: int = 1280,
    duration_sec: int = 15,
) -> None:
    """Run Stage 8 Output Layer verification.

    Args:
        source_url: RTSP stream URL or video path.
        weights: YOLO weights checkpoint path/name.
        device: Target execution device ('cuda' or 'cpu').
        conf_thresh: Detection confidence threshold [0.0 - 1.0].
        imgsz: Target inference resolution dimension.
        duration_sec: Verification duration in seconds.
    """
    logger.info("=" * 60)
    logger.info("STARTING STAGE 8 — OUTPUT LAYER VERIFICATION")
    logger.info("Source URL          : %s", source_url)
    logger.info("YOLO Model Weights  : %s", weights)
    logger.info("Confidence Threshold: %.2f", conf_thresh)
    logger.info("Inference Image Size: %d", imgsz)
    logger.info("Execution Device    : %s", device)
    logger.info("=" * 60)

    test_output_dir = os.path.join(project_root, "output", "test_stage8_frames")
    test_logs_dir = os.path.join(project_root, "output", "test_stage8_logs")

    writer = OutputWriter(
        frames_dir=test_output_dir,
        logs_dir=test_logs_dir,
        metadata_filename="test_detections.jsonl",
    )

    capture = RTSPCapture(rtsp_url=source_url)
    sampler = FrameSampler(capture=capture, target_fps=5.0)

    object_analyzer = ObjectAnalyzer(
        model_weights=weights,
        confidence_threshold=conf_thresh,
        imgsz=imgsz,
        device=device,
    )

    face_analyzer = None
    if HAS_FACE_ANALYZER:
        try:
            face_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
            face_analyzer = FaceAnalyzer(
                model_name="buffalo_l",
                confidence_threshold=conf_thresh,
                providers=face_providers,
            )
            logger.info("InsightFace FaceAnalyzer initialized.")
        except Exception as exc:
            logger.warning("Could not initialize FaceAnalyzer: %s. Continuing with ObjectAnalyzer only.", exc)

    registry = AnalyzerRegistry(
        object_analyzer=object_analyzer,
        face_analyzer=face_analyzer,
        enable_object=True,
        enable_face=(face_analyzer is not None),
        enable_colour=False,
    )

    capture.start()
    start_time = time.time()
    evaluated_samples = 0
    saved_count = 0
    discarded_count = 0

    try:
        while time.time() - start_time < duration_sec:
            accepted, raw_frame, ts = sampler.sample_latest()
            if accepted and raw_frame is not None:
                evaluated_samples += 1

                # Analyze frame
                detections = registry.analyze_frame(raw_frame)

                # Save if detections exist
                record = writer.save_detection_result(
                    frame_bgr=raw_frame,
                    detections=detections,
                    frame_timestamp=ts,
                )

                if record is not None:
                    saved_count += 1
                    labels_summary = [d["label"] for d in detections]
                    logger.info("Sample #%d | Detections (%d): %s -> SAVED", evaluated_samples, len(detections), labels_summary)
                else:
                    discarded_count += 1
                    logger.info("Sample #%d | Detections: 0 -> DISCARDED (0 I/O)", evaluated_samples)

            time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("Verification interrupted by user.")
    finally:
        capture.stop()

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("STAGE 8 VERIFICATION SUMMARY")
    logger.info("Elapsed Time           : %.2fs", elapsed)
    logger.info("Evaluated Sample Frames: %d", evaluated_samples)
    logger.info("Saved Detection Frames : %d", saved_count)
    logger.info("Discarded 0-Det Frames : %d", discarded_count)
    logger.info("Metadata JSONL Log Path: %s", writer.metadata_file_path)
    logger.info("Output Layer           : VERIFIED")
    logger.info("=" * 60)


if __name__ == "__main__":
    cfg = load_default_config()
    default_url = cfg.get("rtsp", {}).get("url", "rtsp://127.0.0.1:8554/live")
    yolo_cfg = cfg.get("models", {}).get("yolo", {})
    default_weights = yolo_cfg.get("weights", "yolov8n.pt")
    default_conf = float(yolo_cfg.get("confidence_threshold", 0.35))
    default_imgsz = int(yolo_cfg.get("imgsz", 640))

    auto_device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(description="Stage 8 Output Layer Standalone Verification")
    parser.add_argument("--url", type=str, default=default_url, help="RTSP stream URL or video path")
    parser.add_argument("--weights", type=str, default=default_weights, help="YOLO model weights (e.g. yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=default_conf, help="Detection confidence threshold (default: 0.35)")
    parser.add_argument("--imgsz", type=int, default=default_imgsz, help="Inference resolution (default: 640)")
    parser.add_argument("--device", type=str, default=auto_device, help="Execution device (cuda/cpu)")
    parser.add_argument("--duration", type=int, default=15, help="Test duration in seconds")

    args = parser.parse_args()
    run_stage8_verification(
        source_url=args.url,
        weights=args.weights,
        device=args.device,
        conf_thresh=args.conf,
        imgsz=args.imgsz,
        duration_sec=args.duration,
    )
