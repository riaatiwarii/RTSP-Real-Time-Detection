"""Master Production Pipeline Launcher for Single RTSP Stream Video Analytics.

Combines:
- Stage 1: RTSPCapture (Daemon Thread background frame grabbing)
- Stage 2: FrameSampler (Zero-Latency rate limiter)
- Stage 3: AnalyzerRegistry (Multi-Threaded Parallel Execution with ThreadPoolExecutor)
- Stage 4: Post-Processing & Conditional Output Persistence (Disk + JSONL + SQLite)
- Stage 5: Live WebSocket Video Overlay Server + Prometheus Metrics Exporter (FastAPI + Uvicorn)

Usage:
    python scripts/run_pipeline.py --url "rtsp://127.0.0.1:8554/live"
    python scripts/run_pipeline.py --url test_video.mp4 --weights yolov8n.pt --port 8000
"""

import argparse
import asyncio
import logging
import os
import sys
import threading
import time
import cv2
import numpy as np
import torch
import uvicorn

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rtsp_pipeline.capture import RTSPCapture
from rtsp_pipeline.sampler import FrameSampler
from rtsp_pipeline.object_analyzer import ObjectAnalyzer
from rtsp_pipeline.face_analyzer import FaceAnalyzer
from rtsp_pipeline.colour_analyzer import ColourAnalyzer
from rtsp_pipeline.registry import AnalyzerRegistry
from rtsp_pipeline.output import OutputWriter
from rtsp_pipeline.metrics import PipelineMetrics
from rtsp_pipeline.server import create_app, broadcaster

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("RTSPProductionPipeline")


def draw_hud_overlay(
    image: np.ndarray,
    detections: list,
    fps: float,
    sample_id: int,
    conf_thresh: float,
) -> np.ndarray:
    """Draw bounding boxes, badges, and top HUD bar onto frame copy."""
    canvas = image.copy()
    h, w = canvas.shape[:2]

    COLOR_PERSON = (0, 255, 0)      # Bright Green
    COLOR_OTHER = (255, 200, 0)     # Yellow/Cyan

    person_count = sum(1 for d in detections if d.get("label") == "person")

    # 1. Draw Bounding Boxes
    for det in detections:
        label = str(det.get("label", ""))
        conf = float(det.get("confidence", 0.0))
        bbox = det.get("bbox")

        if label == "crowd" or bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        color = COLOR_PERSON if label == "person" else COLOR_OTHER

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        track_id = det.get("track_id")
        if track_id is not None:
            badge_text = f"#{track_id} {label.upper()} {int(conf * 100)}%"
        else:
            badge_text = f"{label.upper()} {int(conf * 100)}%"

        (text_w, text_h), baseline = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        pill_top = max(0, y1 - text_h - 6)
        pill_bottom = max(text_h + 6, y1)

        cv2.rectangle(canvas, (x1, pill_top), (x1 + text_w + 6, pill_bottom), (10, 10, 10), cv2.FILLED)
        cv2.rectangle(canvas, (x1, pill_top), (x1 + text_w + 6, pill_bottom), color, 1)
        cv2.putText(canvas, badge_text, (x1 + 3, pill_bottom - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # 2. Draw Top Status HUD Bar
    hud_bg = canvas.copy()
    cv2.rectangle(hud_bg, (0, 0), (w, 40), (15, 15, 15), cv2.FILLED)
    cv2.addWeighted(hud_bg, 0.75, canvas, 0.25, 0, canvas)

    hud_text = f"LIVE STREAM | FPS: {fps:.1f} | Persons: {person_count} | Detections: {len(detections)} | Conf: {conf_thresh:.2f}"
    cv2.putText(canvas, hud_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    return canvas


def run_server_in_thread(app: any, host: str, port: int, event_loop: asyncio.AbstractEventLoop) -> threading.Thread:
    """Run FastAPI Uvicorn web server in a background thread."""
    config = uvicorn.Config(app=app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    
    def target():
        asyncio.set_event_loop(event_loop)
        event_loop.run_until_complete(server.serve())

    thread = threading.Thread(target=target, daemon=True, name="UvicornServerThread")
    thread.start()
    logger.info("FastAPI Web API & WebSocket server launched at http://%s:%d", host, port)
    logger.info("  - Live WebSocket Video Feed: ws://%s:%d/ws/stream", host, port)
    logger.info("  - Prometheus Metrics Endpoint: http://%s:%d/metrics", host, port)
    logger.info("  - REST Detections Query Endpoint: http://%s:%d/api/detections", host, port)
    return thread


def main() -> None:
    parser = argparse.ArgumentParser(description="Master Single RTSP Stream Production Analytics Pipeline")
    parser.add_argument("--url", type=str, default="rtsp://127.0.0.1:8554/live", help="RTSP stream URL or video file path")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="YOLOv8 checkpoint or TensorRT .engine file")
    parser.add_argument("--conf", type=float, default=0.50, help="Confidence threshold (default: 0.50)")
    parser.add_argument("--fps", type=float, default=5.0, help="Target sampling FPS (default: 5.0)")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference resolution dimension (default: 1280)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Execution device ('cuda' or 'cpu')")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Web API bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Web API bind port (default: 8000)")
    parser.add_argument("--enable-face", action="store_true", default=False, help="Enable InsightFace analyzer")
    parser.add_argument("--enable-colour", action="store_true", default=False, help="Enable HSV bounding box colour extractor")
    parser.add_argument("--enable-gui", action="store_true", default=False, help="Open local desktop cv2.imshow GUI window")

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("LAUNCHING PRODUCTION SINGLE RTSP STREAM ANALYTICS PIPELINE")
    logger.info("Stream URL / File  : %s", args.url)
    logger.info("Model Weights      : %s", args.weights)
    logger.info("Confidence Thresh  : %.2f", args.conf)
    logger.info("Target Sampler FPS : %.1f", args.fps)
    logger.info("Execution Device   : %s", args.device)
    logger.info("=" * 70)

    # 1. Initialize Stage 1 Capture & Stage 2 Sampler
    capture = RTSPCapture(rtsp_url=args.url)
    sampler = FrameSampler(capture=capture, target_fps=args.fps)

    # 2. Initialize Stage 3 Parallel Model Registry
    obj_analyzer = ObjectAnalyzer(
        model_weights=args.weights,
        confidence_threshold=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        enable_tracking=True,
    )
    face_analyzer = FaceAnalyzer() if args.enable_face else None
    colour_analyzer = ColourAnalyzer() if args.enable_colour else None

    registry = AnalyzerRegistry(
        object_analyzer=obj_analyzer,
        face_analyzer=face_analyzer,
        colour_analyzer=colour_analyzer,
        enable_object=True,
        enable_face=args.enable_face,
        enable_colour=args.enable_colour,
        max_workers=4,
    )

    # 3. Initialize Stage 5 Output Writer (SQLite + JSONL)
    writer = OutputWriter(enable_sqlite=True)

    # 4. Initialize & Launch FastAPI & WebSockets Server in Background Event Loop
    server_loop = asyncio.new_event_loop()
    app = create_app(db_path=writer.db_file_path)
    run_server_in_thread(app=app, host=args.host, port=args.port, event_loop=server_loop)

    if not capture.start():
        logger.error("Failed to connect to video stream source: %s", args.url)
        return

    sample_counter = 0
    fps_calc_counter = 0
    fps_calc_start = time.time()
    current_fps = 0.0

    if args.enable_gui:
        cv2.namedWindow("Production RTSP Analytics Feed", cv2.WINDOW_NORMAL)

    try:
        while True:
            t0 = time.time()
            accepted, raw_frame, frame_ts = sampler.sample_latest()

            if accepted and raw_frame is not None:
                sample_counter += 1
                fps_calc_counter += 1

                # Calculate pipeline rendering FPS
                elapsed_fps_time = time.time() - fps_calc_start
                if elapsed_fps_time >= 1.0:
                    current_fps = fps_calc_counter / elapsed_fps_time
                    fps_calc_counter = 0
                    fps_calc_start = time.time()
                    PipelineMetrics.set_fps(current_fps)

                # Stage 3: Multi-Threaded Parallel Inference
                detections = registry.analyze_frame(raw_frame)

                # Draw Visual HUD Bounding Box Overlay
                hud_frame = draw_hud_overlay(
                    image=raw_frame,
                    detections=detections,
                    fps=current_fps,
                    sample_id=sample_counter,
                    conf_thresh=args.conf,
                )

                # Broadcast live JPEG over WebSockets to web dashboard
                try:
                    success, jpeg_buf = cv2.imencode(".jpg", hud_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if success:
                        asyncio.run_coroutine_threadsafe(
                            broadcaster.broadcast_jpeg(jpeg_buf.tobytes()),
                            server_loop
                        )
                except Exception as exc:
                    logger.debug("Error broadcasting WebSocket frame: %s", exc)

                # Stage 4 & 5: Conditional Storage Persistence (saves ONLY if len(detections) > 0)
                saved_record = writer.save_detection_result(
                    frame_bgr=raw_frame,
                    detections=detections,
                    frame_timestamp=frame_ts,
                )

                # Record Prometheus Telemetry
                latency = time.time() - t0
                PipelineMetrics.record_latency(latency)
                PipelineMetrics.record_detections(detections)
                PipelineMetrics.record_frame_processed(saved=(saved_record is not None))

                if args.enable_gui:
                    cv2.imshow("Production RTSP Analytics Feed", hud_frame)

            if args.enable_gui:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    logger.info("User requested shutdown via desktop window.")
                    break

            time.sleep(0.005)

    except KeyboardInterrupt:
        logger.info("Pipeline stopped by user (Ctrl+C).")
    finally:
        logger.info("Cleaning up pipeline resources...")
        capture.stop()
        registry.shutdown()
        if args.enable_gui:
            cv2.destroyAllWindows()
        logger.info("Pipeline shutdown complete.")


if __name__ == "__main__":
    main()
