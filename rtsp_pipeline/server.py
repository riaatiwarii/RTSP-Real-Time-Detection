"""FastAPI & WebSockets Production Server for RTSP Video Analytics.

Provides:
- Low-latency live JPEG video HUD streaming over WebSockets (`/ws/stream`)
- Prometheus metric scraping endpoint (`/metrics`)
- REST endpoints for health checks (`/health`) and SQLite detection records (`/api/detections`)
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional, Set
import cv2
import numpy as np

try:
    from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect, Response
    from fastapi.middleware.cors import CORSMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    WebSocket = Any  # type: ignore
    WebSocketDisconnect = Exception  # type: ignore


from rtsp_pipeline.metrics import PipelineMetrics

logger = logging.getLogger(__name__)

start_time = time.time()


class FrameBroadcaster:
    """Manages active WebSocket client connections and broadcasts live video frame JPEGs."""

    def __init__(self) -> None:
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)
        logger.info("WebSocket client connected. Active connections: %d", len(self.active_connections))

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self.active_connections.discard(websocket)
        logger.info("WebSocket client disconnected. Remaining: %d", len(self.active_connections))

    async def broadcast_jpeg(self, jpeg_bytes: bytes) -> None:
        """Broadcast binary JPEG image to all connected WebSocket clients."""
        if not self.active_connections:
            return

        disconnected: List[WebSocket] = []
        async with self._lock:
            connections = list(self.active_connections)

        for ws in connections:
            try:
                await ws.send_bytes(jpeg_bytes)
            except Exception:
                disconnected.append(ws)

        if disconnected:
            async with self._lock:
                for ws in disconnected:
                    self.active_connections.discard(ws)


broadcaster = FrameBroadcaster()


def create_app(db_path: str = "output/logs/detections.db") -> Any:
    """Factory function to build FastAPI application instance."""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI package is required to run the Web API & WebSocket server.")

    app = FastAPI(
        title="RTSP Real-Time Detection API & WebSocket Server",
        description="Single RTSP Stream Multi-Threaded Analytics Pipeline",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health_check() -> Dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "uptime_seconds": round(time.time() - start_time, 2),
            "active_websockets": len(broadcaster.active_connections),
        }

    @app.get("/metrics")
    def metrics() -> Response:
        """Prometheus exporter scrape endpoint."""
        content, content_type = PipelineMetrics.get_metrics_bytes()
        return Response(content=content, media_type=content_type)

    @app.get("/api/detections")
    def get_detections(
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
        label: Optional[str] = Query(None, description="Filter by class label e.g. person"),
    ) -> Dict[str, Any]:
        """Query detection records stored in SQLite database."""
        if not os.path.exists(db_path):
            return {"total": 0, "records": []}

        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                if label:
                    query = """
                        SELECT f.frame_id, f.timestamp, f.image_file, e.label, e.confidence, e.colour, e.bbox_json, e.track_id
                        FROM detection_events e
                        JOIN detection_frames f ON e.frame_id = f.frame_id
                        WHERE e.label = ?
                        ORDER BY f.id DESC LIMIT ? OFFSET ?
                    """
                    cursor.execute(query, (label, limit, offset))
                else:
                    query = """
                        SELECT frame_id, timestamp, image_file, image_path, detection_count
                        FROM detection_frames
                        ORDER BY id DESC LIMIT ? OFFSET ?
                    """
                    cursor.execute(query, (limit, offset))

                rows = [dict(row) for row in cursor.fetchall()]
                return {"limit": limit, "offset": offset, "count": len(rows), "records": rows}
        except Exception as exc:
            logger.error("Error querying SQLite database: %s", exc)
            return {"error": str(exc), "records": []}

    @app.websocket("/ws/stream")
    async def websocket_stream(websocket: WebSocket) -> None:
        """WebSocket endpoint for live video feed broadcasting."""
        await broadcaster.connect(websocket)
        try:
            while True:
                # Keep socket open and receive ping/messages if any
                data = await websocket.receive_text()
        except WebSocketDisconnect:
            await broadcaster.disconnect(websocket)
        except Exception as exc:
            logger.debug("WebSocket error: %s", exc)
            await broadcaster.disconnect(websocket)

    return app
