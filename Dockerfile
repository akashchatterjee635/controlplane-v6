# ControlPlane v6 — Multi-stage Dockerfile
# Stage 1: Install dependencies
# Stage 2: Copy app and run

FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Stage 1: Dependencies
# ---------------------------------------------------------------------------
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: Application
# ---------------------------------------------------------------------------
FROM deps AS app

# Copy application code
COPY app/ app/
COPY data/ data/
COPY eval/ eval/
COPY ui/ ui/
COPY .env.example .env.example

# Create directories for persistent data
RUN mkdir -p /app/chroma_db /app/eval/results

# Default environment variables (override via .env or docker-compose)
ENV CHROMA_PERSIST_PATH=/app/chroma_db \
    SQLITE_CHECKPOINT_PATH=/app/checkpoints.db \
    FASTAPI_PORT=8000

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health').raise_for_status()" || exit 1

# Default: run FastAPI
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
