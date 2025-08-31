FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    # evita caches de pip y chequeos innecesarios
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # mueve los temporales de pip a disco (no tmpfs)
    TMPDIR=/var/tmp/pip \
    PIP_TMPDIR=/var/tmp/pip \
    # tus vars existentes
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata \
    PYTHONPATH=/app

# Paquetes del sistema (sin recomendados) + crear tmp de pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-spa tesseract-ocr-eng \
    poppler-utils \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    ca-certificates curl \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /var/tmp/pip

WORKDIR /app

# --- Instalar deps de Python con mejor caché y sin reusar /tmp ---
COPY requirements.txt .

# (OPCIONAL pero útil si vas justo de espacio: bajar NumPy a un wheel más pequeño)
# Comenta esta línea si necesitas exactamente 2.2.6
RUN sed -i 's/^numpy==.*/numpy==1.26.4/' requirements.txt

# Usa el caché de BuildKit para pip y evita explosión de temporales
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefer-binary -r requirements.txt

# --- Copiar el código después, para aprovechar caché de capas ---
COPY . .

RUN mkdir -p input output temp

EXPOSE 5000
CMD ["gunicorn","--bind","0.0.0.0:5000","--workers","2","--timeout","300","app:app"]
