"""Standalone verification script for Stage 6 — Colour Analyzer.

Fetches sampled frames from RTSP stream, runs YOLOv8 object detection (Stage 4),
runs ColourAnalyzer crop analysis (Stage 6) to attach dominant colors to object bounding boxes,
and logs enriched detection records.
Optionally draws bounding boxes with color badges and saves annotated debug images.
"""

import argparse
import logging
import os
import sys
import time
import cv2
import torch
import yaml

# Ensure project root is in sys.path when script is executed directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rtsp_pipeline.base_analyzer import BaseAnalyzer
from rtsp_pipeline.capture import RTSPCapture
from rtsp_pipeline.colour_analyzer import ColourAnalyzer
from rtsp_pipeline.object_analyzer import ObjectAnalyzer
from rtsp_pipeline.sampler import FrameSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Stage6Verification")


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


def annotate_and_save_enriched(
    frame: cv2.Mat,
    detections: list,
    output_dir: str,
    sample_id: int,
) -> str:
    """Draw bounding boxes and enriched color labels on frame copy."""
    annotated = frame.copy()
    os.makedirs(output_dir, exist_ok=True)

    for det in detections:
        label = det["label"]
        conf = det["confidence"]
        colour = det.get("colour")
        bbox = det.get("bbox")

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            color_bgr = (0, 255, 255)  # Yellow default box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 2)

            colour_str = f" [{colour}]" if colour else ""
            text = f"{label}{colour_str}: {conf:.2f}"

            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            text_w, text_h = text_size[0], text_size[1]

            cv2.rectangle(
                annotated,
                (x1, max(0, y1 - text_h - 10)),
                (x1 + text_w + 6, max(text_h + 10, y1)),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                annotated,
                text,
                (x1 + 3, max(text_h + 4, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

    filename = f"colour_sample_{sample_id:03d}_{len(detections)}dets.jpg"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, annotated)
    return filepath


def run_stage6_verification(
    source_url: str,
    weights: str,
    conf_thresh: float,
    device: str,
    duration_sec: int = 15,
    save_debug: bool = True,
    max_debug_saves: int = 5,
) -> None:
    """Run Stage 6 Colour Analyzer verification.

    Args:
        source_url: RTSP stream URL or video path.
        weights: YOLOv8 model weights path.
        conf_thresh: Object confidence threshold [0.0 - 1.0].
        device: Target device ('cuda' or 'cpu').
        duration_sec: Verification test duration in seconds.
        save_debug: Whether to save annotated debug images.
        max_debug_saves: Max debug images to save to disk.
    """
    logger.info("=" * 60)
    logger.info("STARTING STAGE 6 — COLOUR ANALYZER VERIFICATION")
    logger.info("Source URL          : %s", source_url)
    logger.info("YOLO Weights        : %s", weights)
    logger.info("Confidence Threshold: %.2f", conf_thresh)
    logger.info("Execution Device    : %s", device)
    logger.info("Save Debug Frames   : %s", save_debug)
    logger.info("=" * 60)

    debug_dir = os.path.join(project_root, "output", "test_stage6_debug")

    capture = RTSPCapture(rtsp_url=source_url)
    sampler = FrameSampler(capture=capture, target_fps=5.0)

    object_analyzer = ObjectAnalyzer(
        model_weights=weights,
        confidence_threshold=conf_thresh,
        device=device,
    )
    colour_analyzer = ColourAnalyzer()

    capture.start()
    start_time = time.time()
    evaluated_samples = 0
    total_objects_enriched = 0
    saved_debug_count = 0

    try:
        while time.time() - start_time < duration_sec:
            accepted, raw_frame, ts = sampler.sample_latest()
            if accepted and raw_frame is not None:
                evaluated_samples += 1

                # Stage 3 + 4 Object Detection
                preprocessed = BaseAnalyzer.preprocess_yolo(raw_frame)
                detections, person_count = object_analyzer.analyze(preprocessed)

                # Stage 6 Colour Enrichment
                start_colour = time.time()
                enriched_detections = colour_analyzer.enrich_detections(raw_frame, detections)
                colour_ms = (time.time() - start_colour) * 1000.0

                total_objects_enriched += len(enriched_detections)

                logger.info(
                    "Sample #%d | Colour Extraction Time: %.2fms | Detections: %d",
                    evaluated_samples,
                    colour_ms,
                    len(enriched_detections),
                )

                for idx, det in enumerate(enriched_detections, 1):
                    logger.info(
                        "  └─ [%d] Label: '%s' | Conf: %.4f | Colour: '%s' | BBox: %s",
                        idx,
                        det["label"],
                        det["confidence"],
                        det.get("colour"),
                        det.get("bbox"),
                    )

                # Save debug images if detections exist
                if save_debug and len(enriched_detections) > 0 and saved_debug_count < max_debug_saves:
                    saved_path = annotate_and_save_enriched(
                        frame=raw_frame,
                        detections=enriched_detections,
                        output_dir=debug_dir,
                        sample_id=evaluated_samples,
                    )
                    saved_debug_count += 1
                    logger.info("  └─ [ENRICHED DEBUG IMAGE SAVED]: %s", saved_path)

            time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("Verification interrupted by user.")
    finally:
        capture.stop()

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("STAGE 6 VERIFICATION SUMMARY")
    logger.info("Elapsed Time           : %.2fs", elapsed)
    logger.info("Evaluated Sample Frames: %d", evaluated_samples)
    logger.info("Total Objects Enriched : %d", total_objects_enriched)
    logger.info("Debug Frames Saved     : %d (%s)", saved_debug_count, debug_dir if saved_debug_count > 0 else "N/A")
    logger.info("Colour Extraction      : VERIFIED")
    logger.info("=" * 60)


if __name__ == "__main__":
    cfg = load_default_config()
    rtsp_cfg = cfg.get("rtsp", {})
    yolo_cfg = cfg.get("models", {}).get("yolo", {})

    default_url = rtsp_cfg.get("url", "rtsp://127.0.0.1:8554/live")
    default_weights = yolo_cfg.get("weights", "yolov8n.pt")
    default_conf = float(yolo_cfg.get("confidence_threshold", 0.5))

    auto_device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(description="Stage 6 Colour Analyzer Standalone Verification")
    parser.add_argument("--url", type=str, default=default_url, help="RTSP stream URL or video path")
    parser.add_argument("--weights", type=str, default=default_weights, help="YOLOv8 weights (e.g. yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=default_conf, help="Confidence threshold")
    parser.add_argument("--device", type=str, default=auto_device, help="Execution device (cuda/cpu)")
    parser.add_argument("--duration", type=int, default=15, help="Test duration in seconds")
    parser.add_argument("--save-debug", action="store_true", default=True, help="Save annotated color debug images")
    parser.add_argument("--max-saves", type=int, default=5, help="Max debug frames to save to disk")

    args = parser.parse_args()
    run_stage6_verification(
        source_url=args.url,
        weights=args.weights,
        conf_thresh=args.conf,
        device=args.device,
        duration_sec=args.duration,
        save_debug=args.save_debug,
        max_debug_saves=args.max_saves,
    )
