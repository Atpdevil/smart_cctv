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

# ── Expose port (Render sets PORT env var, default 5000) ──────────────────────
EXPOSE 5000

# ── Start with gunicorn (production WSGI server) ─────────────────────────────
#    - preload so start_pipelines() runs once before forking
#    - single worker since app uses background threads for camera pipelines
#    - increased timeout for heavy model loading at startup
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --threads 4 --timeout 120 --preload 'app:app'"]
