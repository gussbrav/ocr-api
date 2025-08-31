FROM python:3.11-slim

# ---- Sistema y libs nativas necesarias para OCR/PDF ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-spa \
    poppler-utils \
    libgl1-mesa-dri libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    ca-certificates curl \
  && rm -rf /var/lib/apt/lists/*

# ---- Entorno robusto (logs y menos hilos para evitar saturación) ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2 \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata \
    TMPDIR=/dev/shm \
    PYTHONPATH=/app

# ---- App ----
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
COPY . .

# Dirs de trabajo y usuario no root
RUN mkdir -p /app/input /app/output /app/temp \
 && adduser --disabled-password --gecos "" appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

# Healthcheck para que el proxy no enrute si el contenedor no está sano
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -fsS http://127.0.0.1:5000/health || exit 1

# ---- Gunicorn endurecido para cargas pesadas (PDFs 30–50 MB) ----
# - /dev/shm para temporales: IO en RAM, evita "no space on device"
# - timeout alto + graceful-timeout
# - max-requests + jitter: recicla workers y evita fugas de memoria
# - logs a stdout/stderr
CMD ["gunicorn",
     "--bind", "0.0.0.0:5000",
     "--workers", "2",
     "--worker-tmp-dir", "/dev/shm",
     "--timeout", "600",
     "--graceful-timeout", "60",
     "--keep-alive", "10",
     "--max-requests", "200",
     "--max-requests-jitter", "60",
     "--access-logfile", "-",
     "--error-logfile", "-",
     "--log-level", "info",
     "app:app"]
