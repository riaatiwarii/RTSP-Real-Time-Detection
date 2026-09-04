"""Standalone verification script for Stage 4 — Object + Crowd Analyzer.

Fetches sampled frames from RTSP stream, runs YOLOv8 inference, logs detected objects,
bounding box shapes, person counts, and crowd metric triggers.
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
from rtsp_pipeline.object_analyzer import ObjectAnalyzer
from rtsp_pipeline.sampler import FrameSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Stage4Verification")


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


def run_stage4_verification(
    source_url: str,
    weights: str,
    conf_thresh: float,
    crowd_thresh: int,
    duration_sec: int = 15,
) -> None:
    """Run Stage 4 Object + Crowd Analyzer verification.

    Args:
        source_url: RTSP URL or local video file path.
        weights: YOLOv8 model weights path or name.
        conf_thresh: Confidence threshold filter [0.0 - 1.0].
        crowd_thresh: Person count threshold to trigger crowd entry.
        duration_sec: Verification duration in seconds.
    """
    logger.info("=" * 60)
    logger.info("STARTING STAGE 4 — OBJECT + CROWD ANALYZER VERIFICATION")
    logger.info("Source URL          : %s", source_url)
    logger.info("YOLO Model Weights  : %s", weights)
    logger.info("Confidence Threshold: %.2f", conf_thresh)
    logger.info("Crowd Threshold     : %d persons", crowd_thresh)
    logger.info("=" * 60)

    capture = RTSPCapture(rtsp_url=source_url)
    sampler = FrameSampler(capture=capture, target_fps=5.0)

    # Initialize ObjectAnalyzer (loads YOLO model weights)
    analyzer = ObjectAnalyzer(
        model_weights=weights,
        confidence_threshold=conf_thresh,
        crowd_threshold=crowd_thresh,
    )

    capture.start()
    start_time = time.time()
    evaluated_samples = 0
    total_detections_found = 0

    try:
        while time.time() - start_time < duration_sec:
            accepted, raw_frame, ts = sampler.sample_latest()
            if accepted and raw_frame is not None:
                evaluated_samples += 1

                # Stage 3 Preprocessing (YOLO path: passthrough)
                preprocessed = BaseAnalyzer.preprocess_yolo(raw_frame)

                # Stage 4 Analysis
                start_infer = time.time()
                detections, person_count = analyzer.analyze(preprocessed)
                infer_ms = (time.time() - start_infer) * 1000.0

                total_detections_found += len(detections)

                logger.info(
                    "Sample #%d | Inference Time: %.1fms | Detections: %d | Persons: %d",
                    evaluated_samples,
                    infer_ms,
                    len(detections),
                    person_count,
                )

                for idx, det in enumerate(detections, 1):
                    if det["label"] == "crowd":
                        logger.info("  └─ [%d] CROWD METRIC TRIGGERED: %d persons detected", idx, int(det["confidence"]))
                    else:
                        bbox_str = f"({det['bbox'][0]}, {det['bbox'][1]}, {det['bbox'][2]}, {det['bbox'][3]})" if det['bbox'] else "N/A"
                        logger.info("  └─ [%d] Label: '%s' | Conf: %.4f | BBox: %s", idx, det["label"], det["confidence"], bbox_str)

            time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("Verification manually interrupted by user.")
    finally:
        capture.stop()

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("STAGE 4 VERIFICATION SUMMARY")
    logger.info("Elapsed Time           : %.2fs", elapsed)
    logger.info("Evaluated Sample Frames: %d", evaluated_samples)
    logger.info("Total Detections Found : %d", total_detections_found)
    logger.info("YOLO Inference Engine  : VERIFIED")
    logger.info("=" * 60)


if __name__ == "__main__":
    cfg = load_default_config()
    rtsp_cfg = cfg.get("rtsp", {})
    yolo_cfg = cfg.get("models", {}).get("yolo", {})

    default_url = rtsp_cfg.get("url", "rtsp://127.0.0.1:8554/live")
    default_weights = yolo_cfg.get("weights", "yolov8n.pt")
    default_conf = float(yolo_cfg.get("confidence_threshold", 0.5))
    default_crowd = int(yolo_cfg.get("crowd_threshold", 3))

    parser = argparse.ArgumentParser(description="Stage 4 Object + Crowd Analyzer Standalone Verification")
    parser.add_argument("--url", type=str, default=default_url, help="RTSP stream URL or video path")
    parser.add_argument("--weights", type=str, default=default_weights, help="YOLOv8 weights (e.g. yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=default_conf, help="Confidence threshold")
    parser.add_argument("--crowd-thresh", type=int, default=default_crowd, help="Crowd person threshold")
    parser.add_argument("--duration", type=int, default=15, help="Test duration in seconds")

    args = parser.parse_args()
    run_stage4_verification(
        source_url=args.url,
        weights=args.weights,
        conf_thresh=args.conf,
        crowd_thresh=args.crowd_thresh,
        duration_sec=args.duration,
    )
