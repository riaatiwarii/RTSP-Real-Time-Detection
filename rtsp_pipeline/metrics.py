"""Prometheus Metrics Collector Module for RTSP Real-Time Pipeline.

Exposes key operational and performance metrics for Prometheus scraping & Grafana dashboards:
- Live Pipeline FPS
- RTSP Socket Reconnections
- End-to-End Processing Latency
- Detection Counts by Class Label
- Frame Storage Efficiency (Saved vs Discarded)
"""

import logging
import time
from typing import Dict, Optional

try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


if PROMETHEUS_AVAILABLE:
    FPS_GAUGE = Gauge("rtsp_pipeline_fps", "Real-time processed frames per second")
    RECONNECT_COUNTER = Counter("rtsp_pipeline_reconnects_total", "Total RTSP socket reconnect attempts")
    LATENCY_HISTOGRAM = Histogram(
        "rtsp_pipeline_frame_latency_seconds",
        "Time taken to sample, detect, and persist a single frame in seconds",
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
    )
    DETECTIONS_COUNTER = Counter(
        "rtsp_pipeline_detections_total",
        "Total object/face detections recorded",
        ["label"]
    )
    FRAMES_PROCESSED_COUNTER = Counter("rtsp_pipeline_frames_processed_total", "Total frames sampled and evaluated")
    FRAMES_SAVED_COUNTER = Counter("rtsp_pipeline_frames_saved_total", "Frames saved to disk/DB (>0 detections)")
    FRAMES_DISCARDED_COUNTER = Counter("rtsp_pipeline_frames_discarded_total", "Frames discarded (0 detections)")
else:
    logger.warning("prometheus_client package is missing. Prometheus metrics will be disabled.")


class PipelineMetrics:
    """Helper class to record pipeline telemetry."""

    @staticmethod
    def set_fps(fps: float) -> None:
        """Update current FPS metric."""
        if PROMETHEUS_AVAILABLE:
            FPS_GAUGE.set(fps)

    @staticmethod
    def record_reconnect() -> None:
        """Increment RTSP stream reconnect counter."""
        if PROMETHEUS_AVAILABLE:
            RECONNECT_COUNTER.inc()

    @staticmethod
    def record_latency(seconds: float) -> None:
        """Observe frame processing latency."""
        if PROMETHEUS_AVAILABLE:
            LATENCY_HISTOGRAM.observe(seconds)

    @staticmethod
    def record_detections(detections: list) -> None:
        """Increment detection counters grouped by label."""
        if PROMETHEUS_AVAILABLE:
            for det in detections:
                label = str(det.get("label", "unknown"))
                DETECTIONS_COUNTER.labels(label=label).inc()

    @staticmethod
    def record_frame_processed(saved: bool) -> None:
        """Track frame processing counts."""
        if PROMETHEUS_AVAILABLE:
            FRAMES_PROCESSED_COUNTER.inc()
            if saved:
                FRAMES_SAVED_COUNTER.inc()
            else:
                FRAMES_DISCARDED_COUNTER.inc()

    @staticmethod
    def get_metrics_bytes() -> tuple:
        """Generate Prometheus exporter scrape response bytes and content type."""
        if PROMETHEUS_AVAILABLE:
            return generate_latest(), CONTENT_TYPE_LATEST
        return b"# prometheus_client not installed\n", "text/plain"
