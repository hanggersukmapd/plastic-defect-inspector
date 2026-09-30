# Plastic Defect Inspector

A FastAPI computer-vision demo for inspecting plastic products from an uploaded image or a live webcam. It uses Ultralytics YOLOv8 for object detection and can generate an Indonesian QA report with Gemini when `GEMINI_API_KEY` is configured.

## Features

- Manual image upload and live webcam capture
- YOLO inference with confidence scores and bounding boxes
- Optional Gemini-generated QA investigation report
- JSON API and interactive API documentation

## Run locally

Requires Python 3.10 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` for manual upload, or `http://127.0.0.1:8000/live` for live inspection. Webcam access requires browser permission; localhost is treated as a secure context by modern browsers.

## API

- `POST /inspect`: current JSON inspection API
- `POST /upload`: compatibility response used by the included web pages
- `GET /health`: service and model status
- `GET /api/info`: service metadata
- `GET /docs`: interactive Swagger UI

The default `yolov8n.pt` weights are downloaded by Ultralytics when needed and are excluded from this repository. The stock YOLOv8n model is trained on COCO general-object classes, not plastic manufacturing defects; defect-specific use requires a model trained for the target defect classes. Gemini reporting is optional; set `GEMINI_API_KEY` in the environment to enable it.

## Project layout

- `app/main.py`: FastAPI routes, inference, and optional QA report generation
- `templates/upload.html`: manual upload interface
- `templates/live.html`: live webcam inspection interface
- `templates/hmi-interface.html`: standalone HMI interface prototype
- `uploads/`: runtime image storage, excluded from Git
