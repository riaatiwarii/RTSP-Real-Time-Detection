"""Standalone verification script for Stage 1 — RTSP Capture.

Tests lightweight background cap.grab(), on-demand cap.retrieve(), exponential backoff,
and correct unique frame counting.
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

from rtsp_pipeline.capture import RTSPCapture

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Stage1Verification")


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


def run_stage1_verification(
    source_url: str,
    initial_delay: float,
    max_delay: float,
    duration_sec: int,
) -> None:
    """Execute Stage 1 verification loop.

    Args:
        source_url: RTSP stream URL or video file path.
        initial_delay: Initial reconnect delay in seconds.
        max_delay: Maximum reconnect delay in seconds.
        duration_sec: Verification duration in seconds.
    """
    logger.info("=" * 60)
    logger.info("STARTING STAGE 1 — RTSP CAPTURE VERIFICATION")
    logger.info("Source URL            : %s", source_url)
    logger.info("Initial Reconnect Delay: %.1fs", initial_delay)
    logger.info("Max Reconnect Delay    : %.1fs", max_delay)
    logger.info("Test Duration         : %d seconds", duration_sec)
    logger.info("=" * 60)

    capture = RTSPCapture(
        rtsp_url=source_url,
        initial_reconnect_delay=initial_delay,
        max_reconnect_delay=max_delay,
    )

    capture.start()
    start_time = time.time()
    last_log_time = time.time()

    unique_frames_fetched = 0
    last_frame_ts = 0.0

    try:
        while time.time() - start_time < duration_sec:
            has_frame, frame, ts = capture.read_latest()
            if has_frame and frame is not None and ts > last_frame_ts:
                unique_frames_fetched += 1
                last_frame_ts = ts
                curr_time = time.time()
                if curr_time - last_log_time >= 2.0:
                    h, w, c = frame.shape
                    logger.info(
                        "New Stream Frame #%d | Resolution: %dx%d | Channels: %d | Timestamp: %.4fs",
                        unique_frames_fetched,
                        w,
                        h,
                        c,
                        ts,
                    )
                    last_log_time = curr_time

            # Small poll interval to check for new frames from the background grabber
            time.sleep(0.005)

    except KeyboardInterrupt:
        logger.info("Verification interrupted by user.")

    elapsed = time.time() - start_time
    connected_state = capture.is_connected

    # Clean shutdown
    capture.stop()

    logger.info("=" * 60)
    logger.info("STAGE 1 VERIFICATION SUMMARY")
    logger.info("Elapsed Time               : %.2fs", elapsed)
    logger.info("Unique Stream Frames Fetched: %d", unique_frames_fetched)
    logger.info("Stream Decoded FPS         : %.2f", unique_frames_fetched / elapsed if elapsed > 0 else 0.0)
    logger.info("Stream Connected State     : %s", connected_state)
    logger.info("=" * 60)


if __name__ == "__main__":
    cfg = load_default_config()
    rtsp_cfg = cfg.get("rtsp", {})

    default_url = rtsp_cfg.get("url", "rtsp://127.0.0.1:8554/live")
    default_initial = float(rtsp_cfg.get("initial_reconnect_delay", 1.0))
    default_max = float(rtsp_cfg.get("max_reconnect_delay", 30.0))

    parser = argparse.ArgumentParser(description="Stage 1 RTSP Capture Standalone Verification")
    parser.add_argument("--url", type=str, default=default_url, help="RTSP stream URL or video path")
    parser.add_argument("--initial-delay", type=float, default=default_initial, help="Initial reconnect delay (sec)")
    parser.add_argument("--max-delay", type=float, default=default_max, help="Max reconnect delay cap (sec)")
    parser.add_argument("--duration", type=int, default=15, help="Test run duration in seconds")

    args = parser.parse_args()
    run_stage1_verification(
        source_url=args.url,
        initial_delay=args.initial_delay,
        max_delay=args.max_delay,
        duration_sec=args.duration,
    )
