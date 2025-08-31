FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TMPDIR=/dev/shm \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata \
    PYTHONPATH=/app \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MALLOC_ARENA_MAX=2

# Paquetes del sistema (ligeros, sin Mesa/DRM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-spa \
    tesseract-ocr-eng \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    ca-certificates \
    curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

COPY . .

RUN mkdir -p input output temp

EXPOSE 5000

# Igual que tu CMD, solo añadimos tmp en RAM y más timeout (opcional)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--worker-tmp-dir", "/dev/shm", "--timeout", "600", "app:app"]
