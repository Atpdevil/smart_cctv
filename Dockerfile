# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1: Export YOLOv8n → ONNX
# ═══════════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS exporter

WORKDIR /export

# System libraries required by OpenCV/Ultralytics during export
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install Ultralytics
RUN pip install --no-cache-dir ultralytics

# Replace GUI OpenCV with headless OpenCV
RUN pip uninstall -y opencv-python opencv-python-headless && \
    pip install --no-cache-dir opencv-python-headless

# Export YOLO model to ONNX
RUN python -c "\
from ultralytics import YOLO; \
model = YOLO('yolov8n.pt'); \
model.export(format='onnx', imgsz=480, simplify=True); \
print('ONNX export complete')"

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2: Lightweight runtime
# ═══════════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim

# Runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install application dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy exported ONNX model
COPY --from=exporter /export/yolov8n.onnx ./yolov8n.onnx

# Copy application
COPY . .

# Runtime directories
RUN mkdir -p snapshots clips thumbs

EXPOSE ${PORT:-5000}

CMD ["python", "app.py"]