# RTSP Real-Time Detection & Analytics Pipeline

A production-grade, highly resilient Python computer vision pipeline for real-time RTSP video stream ingestion, frame rate sampling, concurrent multi-model analysis (YOLOv8 objects & crowd density, ByteTrack object tracking, InsightFace facial analysis, and HSV bounding-box color extraction), conditional persistence, and live WebSocket stream broadcast with Prometheus metrics.

---

## 🌟 Key Features

- **Resilient RTSP Ingestion**: Non-blocking background reader thread using OpenCV `cap.grab()` and forced TCP transport to eliminate packet loss and video corruption. Features exponential backoff auto-reconnect on stream drops.
- **Target FPS Rate Limiting**: Precise frame rate sampler (default ~5 FPS) that drops stale frames during processing spikes, keeping inference zero-latency.
- **Concurrent Multi-Model Analytics**:
  - **YOLOv8 Object Detection & Crowd Density**: Detects objects, counts crowds, assesses risk levels, and supports GPU PyTorch & TensorRT engines (`.engine`).
  - **ByteTrack Multi-Object Tracking**: Custom tracker integration for continuous track ID assignment across frames.
  - **InsightFace Face Analysis**: Detects faces, extracts 512-d face embeddings, and maps landmarks using ONNX runtime execution.
  - **HSV Bounding Box Colour Extractor**: Analyzes object/person bounding box crops in HSV color space to tag dominant clothing/object colors.
- **Central Concurrent Dispatcher**: `AnalyzerRegistry` uses thread pool execution (`ThreadPoolExecutor`) for parallel model evaluation and dynamic module toggles (`enable_object`, `enable_face`, `enable_colour`).
- **Conditional Persistence & Storage Optimization**: Saves annotated image frames to disk **only** when detections occur. Logs events to structured JSONL logs and an indexed SQLite database (`detection_frames` and `detection_events`).
- **Real-Time Web & Monitoring Server**: Embedded FastAPI web application providing:
  - **WebSocket Live Video Stream (`ws://localhost:8000/ws/stream`)**: MJPEG stream with real-time bounding boxes, track IDs, and crowd overlay.
  - **REST Query API (`http://localhost:8000/api/detections`)**: Search historical detections from SQLite database.
  - **Prometheus Metrics (`http://localhost:8000/metrics`)**: Export pipeline FPS, latency per stage, queue depth, detection counts, and stream status.
  - **Health Endpoint (`http://localhost:8000/health`)**: Instant health status of stream connectivity and worker threads.
- **Production Containerization**: Dockerfile and Docker Compose setup with NVIDIA GPU / CUDA support for effortless server deployment.

---

## 🏗️ Architecture & Project Structure

```
RTSP-Real-Time-Detection/
├── config/
│   ├── config.example.yaml     # Template configuration file
│   ├── config.yaml             # Main configuration file
│   └── bytetrack_custom.yaml   # ByteTrack tracker configuration
├── rtsp_pipeline/
│   ├── __init__.py
│   ├── capture.py              # Stage 1: Threaded RTSP capture with TCP transport & auto-reconnect
│   ├── sampler.py              # Stage 2: Time-interval frame rate sampler (~5 FPS)
│   ├── base_analyzer.py        # Stage 3: Base preprocessor (YOLO passthrough & InsightFace letterboxing)
│   ├── object_analyzer.py      # Stage 4: YOLOv8 object detector, ByteTrack tracking & crowd analysis
│   ├── face_analyzer.py        # Stage 5: InsightFace facial detection & embedding extractor
│   ├── colour_analyzer.py      # Stage 6: HSV bounding box crop dominant color analyzer
│   ├── registry.py             # Stage 7: Concurrent model dispatcher & result merger
│   ├── output.py               # Stage 8: Conditional frame persistence (JPEG, JSONL & SQLite DB)
│   ├── metrics.py              # Prometheus metrics collector & exporter
│   └── server.py               # FastAPI server for WebSocket streaming, REST API & health checks
├── scripts/
│   ├── run_pipeline.py         # Main CLI production entry point
│   ├── live_preview.py         # Desktop interactive OpenCV preview window
│   └── export_tensorrt.py      # TensorRT model conversion helper script (.pt -> .engine)
├── tests/
│   ├── test_*_unit.py          # Stage-specific unit test suites
│   └── test_stage*.py          # Integration verification scripts (Stages 1 through 8)
├── DEPLOYMENT.md               # GPU server Docker deployment guide
├── Dockerfile                  # CUDA/GPU-enabled container build file
├── docker-compose.yml          # Docker Compose deployment setup
├── requirements.txt            # Python dependencies
└── pyproject.toml              # Package configuration
```

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install dependencies in editable mode:

```bash
git clone https://github.com/riaatiwarii/RTSP-Real-Time-Detection.git
cd RTSP-Real-Time-Detection
pip install -e .
```

### 2. Configuration

Copy the example configuration file and customize stream details and model settings:

```bash
cp config/config.example.yaml config/config.yaml
```

Key configuration options (`config/config.yaml`):
- `rtsp.url`: RTSP stream URL or path to a local `.mp4` test video.
- `models.object.weights`: Path to YOLOv8 weights (e.g. `yolov8m.pt` or `yolov8n.engine`).
- `models.face.enabled`: Toggle InsightFace face detection.
- `server.port`: Port for FastAPI & WebSocket server (default: `8000`).

---

## 💻 Running the Pipeline

### Option A: Main Production Pipeline (CLI & Web Server)

Run the full end-to-end pipeline including concurrent analysis, SQLite logging, WebSocket video stream, and Prometheus metrics:

```bash
python scripts/run_pipeline.py --config config/config.yaml
```

### Option B: Desktop Live Preview

Launch an interactive local OpenCV GUI preview with overlay annotations and live performance statistics:

```bash
python scripts/live_preview.py --config config/config.yaml
```

### Option C: Export Model to TensorRT

Optimize YOLOv8 PyTorch model to TensorRT engine format for maximum inference speed on NVIDIA GPUs:

```bash
python scripts/export_tensorrt.py --weights yolov8m.pt --imgsz 1280 --device 0
```

---

## 🌐 Endpoints & API Access

When the pipeline is running, the embedded server provides the following endpoints:

| Endpoint | Protocol | Description |
|---|---|---|
| `http://localhost:8000/health` | HTTP GET | Operational health check status |
| `http://localhost:8000/metrics` | HTTP GET | Prometheus operational metrics |
| `http://localhost:8000/api/detections` | HTTP GET | Query recorded detection history from SQLite |
| `ws://localhost:8000/ws/stream` | WebSocket | Real-time annotated MJPEG video stream |

---

## 🧪 Testing & Verification

### Run Automated Unit Test Suite

Execute all 16 automated unit tests covering frame capture, sampling, preprocessors, analyzers, dispatcher registry, and storage persistence:

```bash
python -m unittest discover tests/
```

### Run Stage Integration Verifications

Validate individual stages against an active stream or video file:

```bash
# Stage 1: RTSP Capture
python tests/test_stage1_capture.py --url "rtsp://your-stream-url" --duration 15

# Stage 2: Target FPS Sampler
python tests/test_stage2_sampler.py --url "rtsp://your-stream-url" --fps 5.0

# Stage 4: Object & Crowd Analyzer
python tests/test_stage4_object_analyzer.py --url "rtsp://your-stream-url"

# Stage 5: Face Analyzer
python tests/test_stage5_face_analyzer.py --url "rtsp://your-stream-url"

# Stage 7: Concurrent Registry Dispatcher
python tests/test_stage7_registry.py --url "rtsp://your-stream-url"

# Stage 8: Output Writer & SQLite Persistence
python tests/test_stage8_output.py --url "rtsp://your-stream-url"
```

---

## 🐳 Docker Deployment

To deploy on a remote Linux server with GPU acceleration, use Docker Compose:

```bash
# Build and start container in detached mode
docker compose up --build -d

# Check live container logs
docker compose logs -f rtsp-pipeline
```

Refer to [DEPLOYMENT.md](file:///d:/VA_RTSP/DEPLOYMENT.md) for detailed GPU driver requirements and production setup instructions.

---

## 📄 License

This project is licensed under the MIT License.
