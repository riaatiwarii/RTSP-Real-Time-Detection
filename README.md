# RTSP Real-Time Detection Pipeline

A production-grade Python inference pipeline for real-time RTSP video stream ingestion, frame rate sampling, multi-model analysis (YOLOv8 objects/crowd, InsightFace faces, HSV crop colour), and conditional persistence.

## Installation

Clone the repository and install the package in editable mode:

```bash
git clone <repository-url>
cd RTSP-Real-Time-Detection
pip install -e .
```

## Quick Start & Verification

### 1. Configuration
Copy the template configuration file:
```bash
cp config/config.example.yaml config/config.yaml
```

### 2. Stage Verification Commands
- **Stage 1 (RTSP Capture)**:
  ```bash
  python tests/test_stage1_capture.py --url "rtsp://your-stream-url" --duration 15
  ```
- **Stage 2 (Frame Sampler)**:
  ```bash
  python tests/test_stage2_sampler.py --url "rtsp://your-stream-url" --fps 5.0 --duration 15
  ```
- **Stage 3 (Base Preprocessor)**:
  ```bash
  python tests/test_stage3_base_analyzer.py --url "rtsp://your-stream-url"
  ```
- **Stage 4 (Object + Crowd Analyzer)**:
  ```bash
  python tests/test_stage4_object_analyzer.py --url "rtsp://your-stream-url"
  ```

### 3. Automated Unit Tests
```bash
python -m unittest discover tests/
```

## Package Architecture
```
rtsp_pipeline/
  ├── capture.py               # Stage 1: Threaded RTSP capture with cap.grab() background reader
  ├── sampler.py               # Stage 2: Time-interval frame sampler (~5 FPS)
  ├── base_analyzer.py         # Stage 3: Model-specific preprocessor (YOLO passthrough, InsightFace letterbox)
  ├── object_analyzer.py       # Stage 4: YOLOv8 object & crowd detector
  ├── face_analyzer.py          # Stage 5: InsightFace face analyzer (upcoming)
  ├── colour_analyzer.py       # Stage 6: HSV crop colour analyzer (upcoming)
  ├── registry.py              # Stage 7: Analyzer dispatcher registry (upcoming)
  ├── output.py                # Stage 8: Conditional output writer (upcoming)
  └── pipeline.py              # Stage 9: Queue-based threaded pipeline (upcoming)
```
