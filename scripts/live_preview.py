"""Standalone Live Window Display Script for RTSP Real-Time Object Tracking.

Runs object detection using RTSPCapture, FrameSampler, and ObjectAnalyzer,
rendering clean live bounding boxes, object labels, confidence scores, person counters,
and FPS overlay in an interactive OpenCV GUI window (`cv2.imshow`).

This script is completely decoupled from the core pipeline logic (`rtsp_pipeline/`).

Usage:
    # Live preview from RTSP stream (default high-precision configuration)
    python scripts/live_preview.py --url "rtsp://your-stream-url"

    # Live preview with custom confidence threshold
    python scripts/live_preview.py --url "rtsp://your-stream-url" --conf 0.55

Controls inside GUI Window:
    - 'p' : Toggle Person-Only vs All Objects
    - 't' : Toggle Persistent Tracking (ByteTrack) ON / OFF
    - '+' : Increase confidence threshold (+0.05) to eliminate loose/overlapping boxes
    - '-' : Decrease confidence threshold (-0.05) to pick up distant people
    - 'q' or ESC : Quit preview
"""

import argparse
import logging
import os
import sys
import time
import cv2
import numpy as np
import torch
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rtsp_pipeline.base_analyzer import BaseAnalyzer
from rtsp_pipeline.capture import RTSPCapture
from rtsp_pipeline.object_analyzer import ObjectAnalyzer
from rtsp_pipeline.sampler import FrameSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("LivePreview")


def load_config() -> dict:
    """Load configuration settings from config/config.yaml if available."""
    config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(PROJECT_ROOT, "config", "config.example.yaml")

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as exc:
            logger.warning("Failed to parse config file: %s", exc)
    return {}


def draw_hud_overlay(
    image: np.ndarray,
    detections: list,
    person_count: int,
    crowd_threshold: int,
    fps: float,
    sample_id: int,
    conf_thresh: float,
    person_only: bool = False,
    tracking_enabled: bool = False,
) -> np.ndarray:
    """Draw clean bounding boxes, compact labels, crowd banners, and HUD metrics onto frame copy."""
    canvas = image.copy()
    h, w = canvas.shape[:2]

    COLOR_PERSON = (0, 255, 0)      # Bright Green (2px thin border)
    COLOR_OTHER = (255, 200, 0)     # Cyan/Yellow

    is_crowd = person_count >= crowd_threshold

    # 1. Draw Clean Bounding Boxes
    for det in detections:
        label = det.get("label")
        conf = det.get("confidence", 0.0)
        bbox = det.get("bbox")

        if label == "crowd" or bbox is None:
            continue

        if person_only and label != "person":
            continue

        x1, y1, x2, y2 = bbox
        color = COLOR_PERSON if label == "person" else COLOR_OTHER

        # Draw Clean 2px Bounding Box Outline
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        # Compact Label Badge (e.g., "#1 85%" or "PERSON 85%")
        track_id = det.get("track_id")
        if tracking_enabled and track_id is not None:
            badge_text = f"#{track_id} {int(conf * 100)}%"
        else:
            badge_text = f"{label.upper()} {int(conf * 100)}%"

        (text_w, text_h), baseline = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

        pill_top = max(0, y1 - text_h - 6)
        pill_bottom = max(text_h + 6, y1)

        # Dark semi-transparent pill background for maximum text clarity
        cv2.rectangle(
            canvas,
            (x1, pill_top),
            (x1 + text_w + 6, pill_bottom),
            (10, 10, 10),
            cv2.FILLED,
        )
        # 1px border around pill badge
        cv2.rectangle(
            canvas,
            (x1, pill_top),
            (x1 + text_w + 6, pill_bottom),
            color,
            1,
        )
        cv2.putText(
            canvas,
            badge_text,
            (x1 + 3, pill_bottom - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # 2. Top HUD Header Bar (Semi-transparent black overlay)
    hud_bg = canvas.copy()
    cv2.rectangle(hud_bg, (0, 0), (w, 45), (15, 15, 15), cv2.FILLED)
    cv2.addWeighted(hud_bg, 0.75, canvas, 0.25, 0, canvas)

    # HUD Status Text
    mode_str = "PERSONS ONLY [P]" if person_only else "ALL OBJECTS [P]"
    track_str = "TRACKING: ON [T]" if tracking_enabled else "TRACKING: OFF [T]"
    hud_text = (
        f"FPS: {fps:.1f}  |  Mode: {mode_str}  |  {track_str}  |  "
        f"Conf Thresh: {conf_thresh:.2f} [+/-]  |  Persons Captured: {person_count}"
    )
    cv2.putText(
        canvas,
        hud_text,
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # 3. Crowd Warning Banner if triggered
    if is_crowd:
        banner_text = f"ALERT: CROWD DETECTED ({person_count} Persons)"
        (bw, bh), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)

        banner_bg = canvas.copy()
        cv2.rectangle(banner_bg, (10, h - 50), (10 + bw + 20, h - 10), (0, 0, 200), cv2.FILLED)
        cv2.addWeighted(banner_bg, 0.85, canvas, 0.15, 0, canvas)

        cv2.putText(
            canvas,
            banner_text,
            (20, h - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return canvas


def run_live_preview(
    source_url: str,
    weights: str,
    conf_thresh: float,
    iou_thresh: float,
    crowd_thresh: int,
    target_fps: float,
    imgsz: int,
    device: str,
    person_only: bool = True,
    enable_tracking: bool = False,
    window_title: str = "RTSP Real-Time Detection Preview",
) -> None:
    """Run real-time RTSP capture with live OpenCV window visualization."""
    logger.info("=" * 60)
    logger.info("STARTING LIVE WINDOW DISPLAY PREVIEW")
    logger.info("Stream URL / File  : %s", source_url)
    logger.info("YOLO Model Weights : %s", weights)
    logger.info("Confidence Thresh  : %.2f", conf_thresh)
    logger.info("IoU NMS Thresh     : %.2f", iou_thresh)
    logger.info("Inference Imgsz    : %d", imgsz)
    logger.info("ByteTrack Tracking : %s", enable_tracking)
    logger.info("Target Sampler FPS : %.1f", target_fps)
    logger.info("Execution Device   : %s", device)
    logger.info("Person Only Mode   : %s", person_only)
    logger.info("Window Title       : %s", window_title)
    logger.info("Controls: [P] Toggle Person-Only | [T] Toggle Tracking | [+/-] Adjust Conf Thresh | [Q/ESC] Exit")
    logger.info("=" * 60)

    current_conf = conf_thresh
    current_person_only = person_only
    current_tracking = enable_tracking

    # Initialize RTSP Capture, Sampler, and Object Analyzer
    capture = RTSPCapture(rtsp_url=source_url)
    sampler = FrameSampler(capture=capture, target_fps=target_fps)
    analyzer = ObjectAnalyzer(
        model_weights=weights,
        confidence_threshold=current_conf,
        iou_threshold=iou_thresh,
        crowd_threshold=crowd_thresh,
        imgsz=imgsz,
        device=device,
        person_only=current_person_only,
        enable_tracking=current_tracking,
    )

    if not capture.start():
        logger.error("Failed to initialize stream source: %s", source_url)
        return

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    sample_counter = 0
    fps_calc_counter = 0
    fps_calc_start = time.time()
    current_fps = 0.0

    try:
        while True:
            accepted, raw_frame, ts = sampler.sample_latest()

            if accepted and raw_frame is not None:
                sample_counter += 1
                fps_calc_counter += 1

                # Calculate real-time rendering FPS
                elapsed_fps_time = time.time() - fps_calc_start
                if elapsed_fps_time >= 1.0:
                    current_fps = fps_calc_counter / elapsed_fps_time
                    fps_calc_counter = 0
                    fps_calc_start = time.time()

                # Stage 3: Preprocessing
                preprocessed = BaseAnalyzer.preprocess_yolo(raw_frame)

                # Stage 4: Object Analysis & Detection
                detections, person_count = analyzer.analyze(preprocessed)

                # Draw Visual HUD & Bounding Boxes
                display_frame = draw_hud_overlay(
                    image=raw_frame,
                    detections=detections,
                    person_count=person_count,
                    crowd_threshold=crowd_thresh,
                    fps=current_fps,
                    sample_id=sample_counter,
                    conf_thresh=current_conf,
                    person_only=current_person_only,
                    tracking_enabled=current_tracking,
                )

                # Render inside OpenCV Window
                cv2.imshow(window_title, display_frame)

            # Check key presses
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                logger.info("Quit signal received ('q'/ESC). Closing live window.")
                break
            elif key in (ord('p'), ord('P')):
                current_person_only = not current_person_only
                analyzer.classes = [0] if current_person_only else None
                logger.info("Toggled Person-Only filter mode: %s", current_person_only)
            elif key in (ord('t'), ord('T')):
                current_tracking = not current_tracking
                analyzer.enable_tracking = current_tracking
                logger.info("Toggled ByteTrack tracking mode: %s", current_tracking)
            elif key in (ord('+'), ord('=')):
                current_conf = min(0.95, round(current_conf + 0.05, 2))
                analyzer.confidence_threshold = current_conf
                logger.info("Increased Confidence Threshold to: %.2f", current_conf)
            elif key in (ord('-'), ord('_')):
                current_conf = max(0.10, round(current_conf - 0.05, 2))
                analyzer.confidence_threshold = current_conf
                logger.info("Decreased Confidence Threshold to: %.2f", current_conf)

            # Check if OpenCV window was closed manually via window 'X' button
            try:
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                    logger.info("Preview window closed by user.")
                    break
            except Exception:
                pass

            time.sleep(0.005)

    except KeyboardInterrupt:
        logger.info("Live preview interrupted by user (Ctrl+C).")
    finally:
        logger.info("Cleaning up capture resources and destroying windows...")
        capture.stop()
        cv2.destroyAllWindows()
        logger.info("Live preview session ended cleanly.")


if __name__ == "__main__":
    cfg = load_config()
    rtsp_cfg = cfg.get("rtsp", {})
    models_cfg = cfg.get("models", {}).get("yolo", {})
    sampling_cfg = cfg.get("sampling", {})

    default_url = rtsp_cfg.get("url", "rtsp://127.0.0.1:8554/live")
    default_weights = models_cfg.get("weights", "yolov8n.pt")
    default_conf = float(models_cfg.get("confidence_threshold", 0.60))
    default_crowd = int(models_cfg.get("crowd_threshold", 3))
    default_fps = float(sampling_cfg.get("target_fps", 5.0))
    auto_device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(description="Standalone RTSP Live Window Display & Object Tracking Preview")
    parser.add_argument("--url", type=str, default=default_url, help="RTSP stream URL or video file path")
    parser.add_argument("--weights", type=str, default=default_weights, help="YOLOv8 weights file (e.g. yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=default_conf, help="Confidence threshold (default: 0.60)")
    parser.add_argument("--iou", type=float, default=0.30, help="NMS IoU threshold (default: 0.30)")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference resolution dimension (default: 1280)")
    parser.add_argument("--crowd-thresh", type=int, default=default_crowd, help="Crowd detection person threshold")
    parser.add_argument("--fps", type=float, default=default_fps, help="Target sampling FPS (e.g. 5.0)")
    parser.add_argument("--device", type=str, default=auto_device, help="Execution device ('cuda' or 'cpu')")
    parser.add_argument("--person-only", action="store_true", default=True, help="Capture ONLY person detections (default: True)")
    parser.add_argument("--all-objects", action="store_false", dest="person_only", help="Detect all COCO object classes")
    parser.add_argument("--enable-tracking", action="store_true", default=False, help="Enable ByteTrack persistent tracking")

    args = parser.parse_args()

    run_live_preview(
        source_url=args.url,
        weights=args.weights,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        crowd_thresh=args.crowd_thresh,
        target_fps=args.fps,
        imgsz=args.imgsz,
        device=args.device,
        person_only=args.person_only,
        enable_tracking=args.enable_tracking,
    )
