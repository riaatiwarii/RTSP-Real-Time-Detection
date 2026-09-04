"""Standalone verification script for Stage 5 — Face Analyzer.

Fetches sampled frames from RTSP stream, applies InsightFace letterboxing/RGB preprocessing,
runs face detection, and logs detected face confidences and coordinates.
Optionally draws bounding boxes and saves annotated test frames to output/test_stage5_debug/.
Defaults to CUDA provider if GPU is present, otherwise falls back to CPU for local testing.
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
from rtsp_pipeline.face_analyzer import FaceAnalyzer
from rtsp_pipeline.sampler import FrameSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Stage5Verification")


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


def annotate_and_save_faces(
    frame: cv2.Mat,
    face_detections: list,
    output_dir: str,
    sample_id: int,
) -> str:
    """Draw face bounding boxes and confidence badges on frame copy."""
    annotated = frame.copy()
    os.makedirs(output_dir, exist_ok=True)

    for det in face_detections:
        conf = det["confidence"]
        bbox = det["bbox"]
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        color = (255, 0, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        text = f"face: {conf:.2f}"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_w, text_h = text_size[0], text_size[1]

        cv2.rectangle(
            annotated,
            (x1, max(0, y1 - text_h - 10)),
            (x1 + text_w + 6, max(text_h + 10, y1)),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            text,
            (x1 + 3, max(text_h + 4, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    filename = f"face_sample_{sample_id:03d}_{len(face_detections)}faces.jpg"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, annotated)
    return filepath


def run_stage5_verification(
    source_url: str,
    model_name: str,
    conf_thresh: float,
    providers: list,
    duration_sec: int = 15,
    save_debug: bool = True,
    max_debug_saves: int = 5,
) -> None:
    """Run Stage 5 Face Analyzer verification.

    Args:
        source_url: RTSP stream URL or video path.
        model_name: InsightFace model package (default: buffalo_l).
        conf_thresh: Confidence threshold [0.0 - 1.0].
        providers: ONNX Runtime execution providers.
        duration_sec: Verification duration in seconds.
        save_debug: Whether to save annotated face debug images.
        max_debug_saves: Max debug images to save to disk.
    """
    logger.info("=" * 60)
    logger.info("STARTING STAGE 5 — FACE ANALYZER VERIFICATION")
    logger.info("Source URL          : %s", source_url)
    logger.info("InsightFace Model   : %s", model_name)
    logger.info("Confidence Threshold: %.2f", conf_thresh)
    logger.info("Execution Providers : %s", providers)
    logger.info("Save Debug Frames   : %s", save_debug)
    logger.info("=" * 60)

    debug_dir = os.path.join(project_root, "output", "test_stage5_debug")

    capture = RTSPCapture(rtsp_url=source_url)
    sampler = FrameSampler(capture=capture, target_fps=5.0)

    analyzer = FaceAnalyzer(
        model_name=model_name,
        confidence_threshold=conf_thresh,
        providers=providers,
    )

    capture.start()
    start_time = time.time()
    evaluated_samples = 0
    total_faces_found = 0
    saved_debug_count = 0

    try:
        while time.time() - start_time < duration_sec:
            accepted, raw_frame, ts = sampler.sample_latest()
            if accepted and raw_frame is not None:
                evaluated_samples += 1

                # Stage 3 Preprocessing (InsightFace path: letterbox + RGB)
                rgb_frame, meta = BaseAnalyzer.preprocess_insightface(raw_frame, target_size=(640, 640))

                # Stage 5 Analysis
                start_infer = time.time()
                detections = analyzer.analyze(rgb_frame, meta=meta)
                infer_ms = (time.time() - start_infer) * 1000.0

                total_faces_found += len(detections)

                logger.info(
                    "Sample #%d | Inference Time: %.1fms | Faces Detected: %d",
                    evaluated_samples,
                    infer_ms,
                    len(detections),
                )

                for idx, det in enumerate(detections, 1):
                    bbox_str = f"({det['bbox'][0]}, {det['bbox'][1]}, {det['bbox'][2]}, {det['bbox'][3]})" if det['bbox'] else "N/A"
                    logger.info("  └─ [%d] Label: '%s' | Conf: %.4f | BBox: %s", idx, det["label"], det["confidence"], bbox_str)

                # Save debug images if faces detected
                if save_debug and len(detections) > 0 and saved_debug_count < max_debug_saves:
                    saved_path = annotate_and_save_faces(
                        frame=raw_frame,
                        face_detections=detections,
                        output_dir=debug_dir,
                        sample_id=evaluated_samples,
                    )
                    saved_debug_count += 1
                    logger.info("  └─ [FACE DEBUG IMAGE SAVED]: %s", saved_path)

            time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("Verification interrupted by user.")
    finally:
        capture.stop()

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("STAGE 5 VERIFICATION SUMMARY")
    logger.info("Elapsed Time           : %.2fs", elapsed)
    logger.info("Evaluated Sample Frames: %d", evaluated_samples)
    logger.info("Total Faces Found      : %d", total_faces_found)
    logger.info("Debug Frames Saved     : %d (%s)", saved_debug_count, debug_dir if saved_debug_count > 0 else "N/A")
    logger.info("InsightFace Engine     : VERIFIED")
    logger.info("=" * 60)


if __name__ == "__main__":
    cfg = load_default_config()
    rtsp_cfg = cfg.get("rtsp", {})
    face_cfg = cfg.get("models", {}).get("insightface", {})

    default_url = rtsp_cfg.get("url", "rtsp://127.0.0.1:8554/live")
    default_model = face_cfg.get("model_name", "buffalo_l")
    default_conf = float(face_cfg.get("confidence_threshold", 0.5))

    has_cuda = torch.cuda.is_available()
    default_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if has_cuda else ["CPUExecutionProvider"]

    parser = argparse.ArgumentParser(description="Stage 5 Face Analyzer Standalone Verification")
    parser.add_argument("--url", type=str, default=default_url, help="RTSP stream URL or video path")
    parser.add_argument("--model-name", type=str, default=default_model, help="InsightFace model package name")
    parser.add_argument("--conf", type=float, default=default_conf, help="Confidence threshold")
    parser.add_argument("--duration", type=int, default=15, help="Test duration in seconds")
    parser.add_argument("--save-debug", action="store_true", default=True, help="Save annotated face debug images")
    parser.add_argument("--max-saves", type=int, default=5, help="Max debug frames to save to disk")

    args = parser.parse_args()
    run_stage5_verification(
        source_url=args.url,
        model_name=args.model_name,
        conf_thresh=args.conf,
        providers=default_providers,
        duration_sec=args.duration,
        save_debug=args.save_debug,
        max_debug_saves=args.max_saves,
    )
