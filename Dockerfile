# Production Dockerfile for Single RTSP Stream Video Analytics Pipeline
# Base Image: Official PyTorch with CUDA 12.1 and cuDNN 8 runtime
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory inside container
WORKDIR /app

# Install system dependencies required for OpenCV, GStreamer, and video processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    ffmpeg \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-tools \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy python dependencies file
COPY requirements.txt /app/

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source code into container
COPY . /app/

# Create output directories for persistent storage
RUN mkdir -p /app/output/frames /app/output/logs

# Expose port 8000 for FastAPI, WebSockets stream & Prometheus /metrics
EXPOSE 8000

# Entrypoint default command
CMD ["python", "scripts/run_pipeline.py", "--host", "0.0.0.0", "--port", "8000"]
