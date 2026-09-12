# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System dependencies for OpenCV (headless) & video processing ──────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install CPU-only PyTorch first (saves ~1.5 GB vs CUDA version) ────────────
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# ── Install remaining dependencies ───────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ────────────────────────────────────────────────────
COPY . .

# ── Create directories for runtime artifacts ──────────────────────────────────
RUN mkdir -p snapshots clips thumbs

# ── Pre-download YOLO weights at build time ───────────────────────────────────
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# ── Railway sets PORT env var automatically ───────────────────────────────────
EXPOSE ${PORT:-5000}

CMD ["python", "app.py"]
