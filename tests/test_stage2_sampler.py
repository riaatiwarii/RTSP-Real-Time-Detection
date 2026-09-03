"""Standalone verification script for Stage 2 — Frame Sampling.

Verifies time-interval throttling over an extended run to confirm zero latency drift
and consistent sampling deltas (~1 / target_fps).
"""

import argparse
import logging
import os
import sys
import time
from typing import List
import yaml

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


def run_stage2_verification(
    source_url: str,
    target_fps: float,
    initial_delay: float,
    max_delay: float,
    duration_sec: int,
) -> None:
    """Run Stage 2 standalone verification loop.

    Args:
        source_url: RTSP URL or video path.
        target_fps: Target sampling rate in FPS.
        initial_delay: Initial reconnect delay in seconds.
        max_delay: Maximum reconnect delay in seconds.
        duration_sec: Verification test duration in seconds.
    """
    logger.info("=" * 60)
    logger.info("STARTING STAGE 2 — FRAME SAMPLER VERIFICATION")
    logger.info("Source URL            : %s", source_url)
    logger.info("Target Sampling FPS   : %.2f (Interval: %.4fs)", target_fps, 1.0 / target_fps)
    logger.info("Test Duration         : %d seconds", duration_sec)
    logger.info("=" * 60)

    capture = RTSPCapture(
        rtsp_url=source_url,
        initial_reconnect_delay=initial_delay,
        max_reconnect_delay=max_delay,
    )
    sampler = FrameSampler(capture=capture, target_fps=target_fps)

    capture.start()
    start_time = time.time()

    accepted_timestamps: List[float] = []
    accepted_deltas: List[float] = []

    try:
        while time.time() - start_time < duration_sec:
            accepted, frame, ts = sampler.sample_latest()
            if accepted and frame is not None:
                if accepted_timestamps:
                    delta = ts - accepted_timestamps[-1]
                    accepted_deltas.append(delta)
                    logger.info(
                        "Accepted Frame #%d | Delta: %.4fs | Target: %.4fs | Resolution: %dx%d",
                        sampler.total_accepted,
                        delta,
                        sampler.sampling_interval,
                        frame.shape[1],
                        frame.shape[0],
                    )
                else:
                    logger.info("Accepted initial frame #1 at timestamp %.4fs", ts)

                accepted_timestamps.append(ts)

            # Polling sleep (small enough to hit target FPS accurately)
            time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("Verification interrupted by user.")
    finally:
        capture.stop()

    elapsed = time.time() - start_time
    total_acc = sampler.total_accepted
    achieved_fps = total_acc / elapsed if elapsed > 0 else 0.0

    logger.info("=" * 60)
    logger.info("STAGE 2 VERIFICATION RESULTS")
    logger.info("=" * 60)
    logger.info("Elapsed Time               : %.2fs", elapsed)
    logger.info("Total Accepted Frames      : %d", total_acc)
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
    cfg = load_default_config()
    rtsp_cfg = cfg.get("rtsp", {})
    sampling_cfg = cfg.get("sampling", {})

    default_url = rtsp_cfg.get("url", "rtsp://127.0.0.1:8554/live")
    default_initial = float(rtsp_cfg.get("initial_reconnect_delay", 1.0))
    default_max = float(rtsp_cfg.get("max_reconnect_delay", 30.0))
    default_fps = float(sampling_cfg.get("target_fps", 5.0))

    parser = argparse.ArgumentParser(description="Stage 2 Frame Sampler Standalone Verification")
    parser.add_argument("--url", type=str, default=default_url, help="RTSP stream URL or video path")
    parser.add_argument("--fps", type=float, default=default_fps, help="Target sampling FPS (default: 5.0)")
    parser.add_argument("--initial-delay", type=float, default=default_initial, help="Initial reconnect delay (sec)")
    parser.add_argument("--max-delay", type=float, default=default_max, help="Max reconnect delay cap (sec)")
    parser.add_argument("--duration", type=int, default=15, help="Test run duration in seconds")

    args = parser.parse_args()
    run_stage2_verification(
        source_url=args.url,
        target_fps=args.fps,
        initial_delay=args.initial_delay,
        max_delay=args.max_delay,
        duration_sec=args.duration,
    )
