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
    pip install --no-cache-dir -r requirements.txt

# ── Copy application code ────────────────────────────────────────────────────
COPY . .

# ── Create directories for runtime artifacts ──────────────────────────────────
RUN mkdir -p snapshots clips thumbs

# ── Pre-download model weights at build time (not at runtime) ─────────────────
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
RUN python -c "import torchreid; torchreid.utils.FeatureExtractor(model_name='osnet_ain_x1_0', device='cpu')"

# ── Hugging Face Spaces uses port 7860 ───────────────────────────────────────
ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
