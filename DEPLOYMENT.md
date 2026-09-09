# GPU Server Docker Deployment Guide

This guide explains how to deploy and run the **RTSP Analytics Pipeline** on a remote Linux GPU server using Docker and Docker Compose.

---

## 📋 Prerequisites on Target GPU Server

Ensure the GPU server has:
1. **NVIDIA GPU Drivers** installed (`nvidia-smi` works).
2. **Docker Engine & Docker Compose** installed.
3. **NVIDIA Container Toolkit** installed (allows Docker containers to use GPU):
   ```bash
   # Quick check on GPU server
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

---

## 🚀 Step 1: Transfer Code to GPU Server

Clone or copy your project repository to the GPU server:
```bash
git clone <your-repo-url>
cd RTSP-Real-Time-Detection
```

---

## ⚙️ Step 2: Configure Environment Variables

Create or update a `.env` file on the server (or edit `docker-compose.yml` directly):
```env
RTSP_URL=rtsp://your-camera-ip:554/stream
YOLO_WEIGHTS=yolov8m.pt
CONF_THRESH=0.50
TARGET_FPS=5.0
```

---

## 🐳 Step 3: Launch with Docker Compose

Start the pipeline container in detached mode:
```bash
docker compose up --build -d
```

---

## 🔍 Step 4: Check Logs & Status

```bash
# View real-time container logs
docker compose logs -f rtsp-pipeline

# Check running status
docker compose ps
```

---

## 🌐 Step 5: Access Live Endpoints from Any Machine

- **Live WebSocket Video Feed**: `ws://<SERVER_IP>:8000/ws/stream`
- **Prometheus Metrics**: `http://<SERVER_IP>:8000/metrics`
- **SQLite Detection API**: `http://<SERVER_IP>:8000/api/detections`
- **Health Check**: `http://<SERVER_IP>:8000/health`

---

## 🛑 Step 6: Stop Pipeline

```bash
docker compose down
```
