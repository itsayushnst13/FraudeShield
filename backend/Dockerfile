# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libgomp is required by XGBoost and LightGBM at runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

# Install the CPU-only PyTorch build explicitly. The default PyPI wheel pulls
# several GB of CUDA libraries that a CPU inference container never uses.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3"

# ML package first: its dependency set changes far less often than app code.
COPY ml/pyproject.toml /app/ml/pyproject.toml
COPY ml/src /app/ml/src
RUN pip install /app/ml

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install -r /app/backend/requirements.txt

COPY backend/app /app/backend/app
COPY configs /app/configs

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

ENV MODEL_DIR=/app/models/fraud_detector \
    PYTHONPATH=/app/backend

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--app-dir", "/app/backend", "--host", "0.0.0.0", "--port", "8000"]
