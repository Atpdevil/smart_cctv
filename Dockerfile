# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System dependencies for OpenCV (headless) & video processing ──────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install CPU-only PyTorch (no CUDA = saves ~1.5 GB) ───────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir gdown && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn

# ── Copy application code ────────────────────────────────────────────────────
COPY . .

# ── Create directories for runtime artifacts ──────────────────────────────────
RUN mkdir -p snapshots clips thumbs

# ── Pre-download YOLO weights at build time (not at runtime) ──────────────────
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# ── Expose port (Render sets PORT env var, default 5000) ──────────────────────
EXPOSE 5000

# ── Run directly with Flask (threaded) ────────────────────────────────────────
#    Gunicorn's worker timeout kills the process during slow model loading.
#    Flask threaded mode is fine here — single process with background threads.
CMD ["sh", "-c", "python app.py"]
