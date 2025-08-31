FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    TMPDIR=/var/tmp/pip \
    PIP_TMPDIR=/var/tmp/pip \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata \
    PYTHONPATH=/app

# Paquetes del sistema necesarios (incluyo libgl1 por si cv2 lo pide)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-spa tesseract-ocr-eng \
    poppler-utils \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 libgl1 \
    ca-certificates curl \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /var/tmp/pip

WORKDIR /app

COPY requirements.txt .

# ⚠️ SIN sed que cambia NumPy. Debe quedar el NumPy 2.x de tu requirements.
RUN pip install --prefer-binary -r requirements.txt

COPY . .

RUN mkdir -p input output temp

EXPOSE 5000
CMD ["gunicorn","--bind","0.0.0.0:5000","--workers","2","--timeout","1200","app:app"]
