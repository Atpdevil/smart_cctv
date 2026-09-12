# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Stage 1: Export YOLO to ONNX (temporary — torch removed after) ───────────
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir ultralytics && \
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=480, simplify=True)" && \
    mv yolov8n.onnx /tmp/yolov8n.onnx && \
    pip uninstall -y torch torchvision ultralytics && \
    pip cache purge

# ── Stage 2: Install lightweight runtime deps only ───────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir onnxruntime

# ── Copy app code + ONNX model ───────────────────────────────────────────────
COPY . .
RUN mv /tmp/yolov8n.onnx ./yolov8n.onnx

# ── Create runtime directories ───────────────────────────────────────────────
RUN mkdir -p snapshots clips thumbs

# ── Railway sets PORT env var automatically ──────────────────────────────────
EXPOSE ${PORT:-5000}

CMD ["python", "app.py"]
