"""Standalone verification script for Stage 7 — Analyzer Registry.

Fetches sampled frames from RTSP stream, dispatches through AnalyzerRegistry (combining
ObjectAnalyzer and FaceAnalyzer), and prints/logs flat merged Detection outputs.
"""

import argparse
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
from rtsp_pipeline.face_analyzer import FaceAnalyzer
from rtsp_pipeline.object_analyzer import ObjectAnalyzer
from rtsp_pipeline.registry import AnalyzerRegistry
from rtsp_pipeline.sampler import FrameSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Stage7Verification")


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


def run_stage7_verification(
    source_url: str,
    device: str,
    duration_sec: int = 15,
) -> None:
    """Run Stage 7 Analyzer Registry verification.

    Args:
        source_url: RTSP stream URL or video path.
        device: Target execution device ('cuda' or 'cpu').
        duration_sec: Verification duration in seconds.
    """
    logger.info("=" * 60)
    logger.info("STARTING STAGE 7 — ANALYZER REGISTRY VERIFICATION")
    logger.info("Source URL       : %s", source_url)
    logger.info("Execution Device : %s", device)
    logger.info("=" * 60)

    capture = RTSPCapture(rtsp_url=source_url)
    sampler = FrameSampler(capture=capture, target_fps=5.0)

    # Initialize Stage 4 & Stage 5 Analyzers
    object_analyzer = ObjectAnalyzer(
        model_weights="yolov8n.pt",
        confidence_threshold=0.5,
        crowd_threshold=3,
        device=device,
    )

    face_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    face_analyzer = FaceAnalyzer(
        model_name="buffalo_l",
        confidence_threshold=0.5,
        providers=face_providers,
    )

    # Initialize Central Registry
    registry = AnalyzerRegistry(
        object_analyzer=object_analyzer,
        face_analyzer=face_analyzer,
        enable_object=True,
        enable_face=True,
        enable_colour=False,  # Color extraction left disabled per specification
    )

    capture.start()
    start_time = time.time()
    evaluated_samples = 0
    total_detections_merged = 0

    try:
        while time.time() - start_time < duration_sec:
            accepted, raw_frame, ts = sampler.sample_latest()
            if accepted and raw_frame is not None:
                evaluated_samples += 1

                # Dispatch frame through central registry
                start_dispatch = time.time()
                detections = registry.analyze_frame(raw_frame)
                dispatch_ms = (time.time() - start_dispatch) * 1000.0

                total_detections_merged += len(detections)

                logger.info(
                    "Sample #%d | Registry Dispatch Time: %.1fms | Total Detections: %d",
                    evaluated_samples,
                    dispatch_ms,
                    len(detections),
                )

                for idx, det in enumerate(detections, 1):
                    # Verify strict schema: {"label": str, "confidence": float, "colour": str | None}
                    assert "label" in det and "confidence" in det and "colour" in det
                    assert "bbox" not in det, "Internal bbox field should be stripped from final detection record"
                    logger.info("  └─ [%d] Detection Record: %s", idx, det)

            time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("Verification manually interrupted by user.")
    finally:
        capture.stop()

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("STAGE 7 VERIFICATION SUMMARY")
    logger.info("Elapsed Time           : %.2fs", elapsed)
    logger.info("Evaluated Sample Frames: %d", evaluated_samples)
    logger.info("Total Merged Detections: %d", total_detections_merged)
    logger.info("Registry Dispatch      : VERIFIED")
    logger.info("=" * 60)


if __name__ == "__main__":
    cfg = load_default_config()
    default_url = cfg.get("rtsp", {}).get("url", "rtsp://127.0.0.1:8554/live")
    auto_device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(description="Stage 7 Analyzer Registry Standalone Verification")
    parser.add_argument("--url", type=str, default=default_url, help="RTSP stream URL or video path")
    parser.add_argument("--device", type=str, default=auto_device, help="Execution device (cuda/cpu)")
    parser.add_argument("--duration", type=int, default=15, help="Test duration in seconds")

    args = parser.parse_args()
    run_stage7_verification(
        source_url=args.url,
        device=args.device,
        duration_sec=args.duration,
    )
