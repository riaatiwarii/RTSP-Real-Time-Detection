"""Standalone verification test script for Stage 2 — Frame Sampling.

In-depth verification measuring timestamp deltas between accepted frames to prove
time-interval throttling consistency.
"""

import argparse
import logging
import os
import sys
import time
from typing import List

# Ensure project root is in sys.path when script is executed directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from rtsp_pipeline.capture import RTSPCapture
from rtsp_pipeline.sampler import FrameSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Stage2Verification")


def run_sampler_verification(source_url: str, target_fps: float = 5.0, duration_sec: int = 15) -> None:
    """Run capture + sampler loop and log timestamp deltas of accepted frames.

    Args:
        source_url: RTSP URL or local video file path.
        target_fps: Desired sampling rate in frames per second.
        duration_sec: Verification duration in seconds.
    """
    logger.info("Starting Stage 2 Frame Sampler Verification")
    logger.info("Target Sampling FPS: %.2f (Expected interval: %.4fs)", target_fps, 1.0 / target_fps)
    logger.info("Source stream: %s | Test duration: %d seconds", source_url, duration_sec)

    capture = RTSPCapture(source_url)
    sampler = FrameSampler(target_fps=target_fps)

    if not capture.connect():
        logger.error("Failed to connect to stream source.")
        return

    start_time = time.time()
    accepted_timestamps: List[float] = []
    accepted_deltas: List[float] = []

    try:
        while time.time() - start_time < duration_sec:
            ret, frame = capture.read_frame()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            now = time.time()
            accepted, _, ts = sampler.process_frame(frame, frame_timestamp=now)

            if accepted:
                if accepted_timestamps:
                    delta = ts - accepted_timestamps[-1]
                    accepted_deltas.append(delta)
                    logger.info(
                        "Accepted frame #%d | Timestamp Delta: %.4fs (target: %.4fs)",
                        sampler.total_accepted,
                        delta,
                        sampler.sampling_interval,
                    )
                else:
                    logger.info("Accepted initial frame #1 at t=%.4fs", ts)

                accepted_timestamps.append(ts)

    except KeyboardInterrupt:
        logger.info("Verification interrupted by user.")
    finally:
        capture.release()

    elapsed = time.time() - start_time
    total_raw = sampler.total_evaluated
    total_acc = sampler.total_accepted
    achieved_fps = total_acc / elapsed if elapsed > 0 else 0.0

    logger.info("=" * 60)
    logger.info("STAGE 2 VERIFICATION RESULTS")
    logger.info("=" * 60)
    logger.info("Elapsed Time               : %.2fs", elapsed)
    logger.info("Total Raw Stream Frames    : %d", total_raw)
    logger.info("Total Accepted Frames      : %d", total_acc)
    logger.info("Raw Stream FPS             : %.2f", total_raw / elapsed if elapsed > 0 else 0.0)
    logger.info("Target Sampling FPS        : %.2f", target_fps)
    logger.info("Achieved Sampling FPS      : %.2f", achieved_fps)

    if accepted_deltas:
        mean_delta = float(np.mean(accepted_deltas))
        min_delta = float(np.min(accepted_deltas))
        max_delta = float(np.max(accepted_deltas))
        std_delta = float(np.std(accepted_deltas))

        logger.info("Expected Time Delta        : %.4fs", 1.0 / target_fps)
        logger.info("Actual Mean Time Delta     : %.4fs", mean_delta)
        logger.info("Min Time Delta             : %.4fs", min_delta)
        logger.info("Max Time Delta             : %.4fs", max_delta)
        logger.info("Std Dev of Time Deltas     : %.4fs", std_delta)
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 Frame Sampler Verification")
    parser.add_argument(
        "--url",
        type=str,
        default="rtsp://127.0.0.1:8554/live",
        help="RTSP stream URL or video file path",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Target sampling FPS (default: 5.0)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=15,
        help="Verification duration in seconds (default: 15)",
    )
    args = parser.parse_args()
    run_sampler_verification(source_url=args.url, target_fps=args.fps, duration_sec=args.duration)
