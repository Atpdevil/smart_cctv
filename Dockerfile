# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1: Export YOLOv8n → ONNX (temporary, discarded after build)
# ═══════════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS exporter

WORKDIR /export
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir ultralytics

RUN python -c "\
from ultralytics import YOLO; \
model = YOLO('yolov8n.pt'); \
model.export(format='onnx', imgsz=480, simplify=True); \
print('ONNX export complete')"

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2: Lightweight runtime (NO torch, NO ultralytics)
# ═══════════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim

# System deps for OpenCV headless + ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install only lightweight Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy ONNX model from exporter stage
COPY --from=exporter /export/yolov8n.onnx ./yolov8n.onnx

# Copy application code
COPY . .

# Create runtime directories
RUN mkdir -p snapshots clips thumbs

EXPOSE ${PORT:-5000}

CMD ["python", "app.py"]
