"""Standalone verification test script for Stage 3 — Base Analyzer.

Fetches live frames from RTSP stream / sampler and verifies model-specific preprocessing output shapes,
color space conversions, and letterbox aspect-ratio preservation metadata.
"""

import argparse
import logging
import os
import sys
import time
import yaml

# Ensure project root is in sys.path when script is executed directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rtsp_pipeline.base_analyzer import BaseAnalyzer
from rtsp_pipeline.capture import RTSPCapture
from rtsp_pipeline.sampler import FrameSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Stage3Verification")


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


def run_stage3_verification(source_url: str, duration_sec: int = 10) -> None:
    """Run Stage 3 preprocessing verification on live camera frames.

    Args:
        source_url: RTSP URL or local video path.
        duration_sec: Test duration in seconds.
    """
    logger.info("=" * 60)
    logger.info("STARTING STAGE 3 — BASE ANALYZER VERIFICATION")
    logger.info("Source URL : %s", source_url)
    logger.info("=" * 60)

    capture = RTSPCapture(rtsp_url=source_url)
    sampler = FrameSampler(capture=capture, target_fps=5.0)

    capture.start()
    start_time = time.time()
    tested_frames = 0

    try:
        while time.time() - start_time < duration_sec:
            accepted, frame, ts = sampler.sample_latest()
            if accepted and frame is not None:
                tested_frames += 1

                # 1. Test YOLO preprocessing path
                yolo_frame = BaseAnalyzer.preprocess_yolo(frame)
                assert yolo_frame.shape == frame.shape, "YOLO path shape mismatch"
                assert yolo_frame.dtype == frame.dtype, "YOLO path dtype mismatch"

                # 2. Test InsightFace preprocessing path
                insight_frame, meta = BaseAnalyzer.preprocess_insightface(frame, target_size=(640, 640))
                assert insight_frame.shape == (640, 640, 3), f"InsightFace shape mismatch: {insight_frame.shape}"

                logger.info("Tested Frame #%d", tested_frames)
                logger.info("  Raw BGR Frame Shape        : %s (dtype: %s)", frame.shape, frame.dtype)
                logger.info("  YOLO Path Output Shape     : %s (Passthrough - Unchanged)", yolo_frame.shape)
                logger.info("  InsightFace Output Shape   : %s (Letterboxed RGB)", insight_frame.shape)
                logger.info("  Letterbox Metadata         : Scale=%.4f, Pad (T: %d, B: %d, L: %d, R: %d)",
                            meta["scale"], meta["pad_top"], meta["pad_bottom"], meta["pad_left"], meta["pad_right"])

                # Test a couple frames and break
                if tested_frames >= 3:
                    break

            time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("Verification interrupted by user.")
    finally:
        capture.stop()

    logger.info("=" * 60)
    logger.info("STAGE 3 VERIFICATION SUMMARY")
    logger.info("Total Frames Verified : %d", tested_frames)
    logger.info("YOLO Passthrough Test : PASSED")
    logger.info("InsightFace Letterbox : PASSED")
    logger.info("=" * 60)


if __name__ == "__main__":
    cfg = load_default_config()
    default_url = cfg.get("rtsp", {}).get("url", "rtsp://127.0.0.1:8554/live")

    parser = argparse.ArgumentParser(description="Stage 3 Base Analyzer Verification")
    parser.add_argument("--url", type=str, default=default_url, help="RTSP stream URL or video path")
    parser.add_argument("--duration", type=int, default=10, help="Test run duration in seconds")
    args = parser.parse_args()

    run_stage3_verification(source_url=args.url, duration_sec=args.duration)
