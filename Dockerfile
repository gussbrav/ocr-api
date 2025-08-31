FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TMPDIR=/dev/shm \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata \
    PYTHONPATH=/app \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MALLOC_ARENA_MAX=2

# Paquetes mínimos para OCR headless (sin mesa)
RUN apt-get update && apt-get install -y --no-install-recommends \
      tesseract-ocr tesseract-ocr-eng tesseract-ocr-spa \
      poppler-utils \
      libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
      ca-certificates curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# No uses cache de pip y usa binarios precompilados si existen
RUN python -m pip install --upgrade --no-cache-dir pip \
 && pip install --no-cache-dir --prefer-binary -r requirements.txt

COPY . .
RUN mkdir -p /app/input /app/output /app/temp \
 && adduser --disabled-password --gecos "" appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -fsS http://127.0.0.1:5000/health || exit 1

# IMPORTANTE: CMD en una sola línea (forma JSON)
CMD ["gunicorn","--bind","0.0.0.0:5000","--workers","2","--worker-tmp-dir","/dev/shm","--timeout","600","--graceful-timeout","60","--keep-alive","10","--max-requests","200","--max-requests-jitter","60","--access-logfile","-","--error-logfile","-","--log-level","info","app:app"]
