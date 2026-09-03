"""Standalone verification test script for Stage 1 — RTSP Capture.

Can be run against a real RTSP URL, a local MP4 file path, or an invalid stream to verify
reconnect backoff logic.
"""

import argparse
import logging
import os
import sys
import time

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


def run_verification(source_url: str, duration_sec: int = 15) -> None:
    """Run frame reading loop for a set duration to verify capture stability and reconnects.

    Args:
        source_url: Video stream URL or file path.
        duration_sec: How long to run the verification loop in seconds.
    """
    logger.info("Starting Stage 1 RTSP Capture Verification on source: %s", source_url)
    capture = RTSPCapture(
        rtsp_url=source_url,
        initial_reconnect_delay=1.0,
        max_reconnect_interval=8.0,
        max_retries=5,
    )

    if not capture.connect():
        logger.error("Initial connection failed. Entering loop to test reconnect backoff...")

    start_time = time.time()
    frame_count = 0
    last_log_time = time.time()

    try:
        while time.time() - start_time < duration_sec:
            ret, frame = capture.read_frame()
            if ret and frame is not None:
                frame_count += 1
                curr_time = time.time()
                if curr_time - last_log_time >= 2.0:
                    h, w, c = frame.shape
                    logger.info(
                        "Grabbed frame #%d | Resolution: %dx%d | Channels: %d",
                        frame_count,
                        w,
                        h,
                        c,
                    )
                    last_log_time = curr_time
            else:
                logger.warning("No frame returned. Short sleep before next read attempt...")
                time.sleep(0.5)

    except KeyboardInterrupt:
        logger.info("Verification manually interrupted by user.")
    finally:
        capture.release()

    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0.0
    logger.info(
        "Verification complete. Total frames: %d | Elapsed: %.2fs | Avg Grab FPS: %.2f",
        frame_count,
        elapsed,
        fps,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 RTSP Capture Verification")
    parser.add_argument(
        "--url",
        type=str,
        default="rtsp://admin:123456@192.168.0.166:554/ch01.264?dev=1",
        help="RTSP URL or video file path for testing",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=15,
        help="Test run duration in seconds",
    )
    args = parser.parse_args()
    run_verification(source_url=args.url, duration_sec=args.duration)
