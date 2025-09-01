import io
import os
import re
import gc
import cv2
import time
import math
import json
import uuid
import base64
import logging
import tempfile
import numpy as np
from typing import List, Tuple

from flask import Flask, request, jsonify
from flask_cors import CORS

from PIL import Image
import pytesseract
from pdf2image import convert_from_path, pdfinfo_from_path
from PyPDF2 import PdfReader


# ----------------------------
# Configuración básica
# ----------------------------
app = Flask(__name__)
CORS(app)

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ocr-api")

# DPI alto para texto pequeño. Puedes subir a 600 si lo ves necesario.
PDF_DPI = int(os.environ.get("PDF_DPI", "500"))

# Idiomas por defecto: español + inglés
DEFAULT_LANG = os.environ.get("OCR_LANG", "spa+eng")


# ----------------------------
# Utilidades de imagen
# ----------------------------
def _fix_orientation_inplace(png_path: str):
    """
    Usa OSD de Tesseract para detectar rotación y corrige si es necesario.
    Modifica el archivo en disco.
    """
    try:
        osd = pytesseract.image_to_osd(png_path)
        m = re.search(r"Rotate:\s+(\d+)", osd)
        rot = int(m.group(1)) if m else 0
        if rot % 360:
            img = Image.open(png_path).convert("RGB")
            img = img.rotate(360 - rot, expand=True, fillcolor="white")
            img.save(png_path)
            logger.debug(f"Orientation fixed by {rot} degrees for {png_path}")
    except Exception as e:
        logger.debug(f"OSD orientation check failed: {e}")


def _deskew(binary_img: np.ndarray) -> np.ndarray:
    """Endereza la imagen binaria estimando el ángulo por el rectángulo mínimo."""
    try:
        coords = np.column_stack(np.where(binary_img == 0))
        if coords.size == 0:
            return binary_img
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        (h, w) = binary_img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(
            binary_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
    except Exception:
        return binary_img


def _remove_horizontal_lines(bin_img: np.ndarray) -> np.ndarray:
    """Elimina renglones/guías horizontales que rompen palabras."""
    try:
        inv = 255 - bin_img
        k = max(30, bin_img.shape[1] // 30)
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
        detected = cv2.morphologyEx(inv, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
        bin_img[detected > 0] = 255
        return bin_img
    except Exception:
        return bin_img


def enhance_image_for_ocr(image_path: str) -> np.ndarray:
    """
    Mejora agresiva pensada para PDFs legales con texto pequeño y líneas.
    Devuelve imagen binaria (np.ndarray) lista para OCR.
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # 1) Escalado para rescatar detalle
        scale = 3.0
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # 2) Gris + CLAHE
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # 3) Des-ruido
        gray = cv2.fastNlMeansDenoising(gray, h=15)

        # 4) Umbralización Otsu
        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 5) Quitar líneas horizontales
        thr = _remove_horizontal_lines(thr)

        # 6) Deskew
        thr = _deskew(thr)

        # 7) Cierre morfológico fino
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((1, 1), np.uint8))

        return thr
    except Exception as e:
        logger.error(f"Error enhancing image: {e}")
        return None


# ----------------------------
# Post-procesado de texto
# ----------------------------
COMMON_FIXES = [
    (r"\s+\n", "\n"),
    (r"\n{3,}", "\n\n"),
    (r"[ \t]{2,}", " "),
    (r"([.,;:])\1{1,}", r"\1"),
    (r"—\s*—", "—"),
    (r"’", "'"),
]


def correct_ocr_text(text: str) -> str:
    """Arreglos ligeros y seguros sobre la salida de OCR."""
    t = text
    for pat, repl in COMMON_FIXES:
        t = re.sub(pat, repl, t)
    # Unir palabras cortadas con guion al salto de línea
    t = re.sub(r"(\w)-\n(\w)", r"\1\2\n", t)
    return t.strip()


# ----------------------------
# OCR
# ----------------------------
def perform_advanced_ocr(image_path: str, language: str) -> str:
    """
    Ejecuta varios pases de OCR y elige el mejor por confianza media.
    """
    try:
        base_cfg = "-c preserve_interword_spaces=1 -c user_defined_dpi=500"
        lang = language

        configs = [
            f"--psm 6 --oem 1 {base_cfg}",  # bloque de texto
            f"--psm 4 --oem 1 {base_cfg}",  # una columna
            f"--psm 11 --oem 1 {base_cfg}",  # líneas sueltas
            f"--psm 6 --oem 1 {base_cfg} "
            "-c tessedit_char_whitelist="
            "0123456789ABCDEFGHIJKLMNÑOPQRSTUVWXYZabcdefghijklmnñopqrstuvwxyzÁÉÍÓÚáéíóú°:-./,()",
        ]

        best_text, max_conf = "", -1.0
        for cfg in configs:
            try:
                data = pytesseract.image_to_data(
                    image_path, lang=lang, config=cfg, output_type=pytesseract.Output.DICT
                )
                confs = [int(c) for c in data["conf"] if str(c).isdigit() and int(c) > 0]
                avg = float(np.mean(confs)) if confs else 0.0

                text = pytesseract.image_to_string(image_path, lang=lang, config=cfg)
                text = correct_ocr_text(text)

                if avg >= max_conf and text.strip():
                    best_text, max_conf = text, avg
            except Exception as e:
                logger.debug(f"OCR pass failed: {e}")

        # Fallback
        if not best_text.strip():
            text = pytesseract.image_to_string(image_path, lang=lang)
            best_text = correct_ocr_text(text)

        return best_text
    except Exception as e:
        logger.error(f"Advanced OCR failed: {e}")
        return ""


# ----------------------------
# PDF helpers
# ----------------------------
def _page_has_text(reader: PdfReader, page_index: int) -> Tuple[bool, str]:
    """
    Intenta extraer texto con PyPDF2. Si obtiene cantidad razonable, evitamos OCR.
    """
    try:
        page = reader.pages[page_index]
        raw = page.extract_text() or ""
        # Heurística: si hay suficiente texto "usable", considerar válido.
        useful = len(re.sub(r"\s+", "", raw))
        return (useful >= 40), correct_ocr_text(raw)
    except Exception:
        return (False, "")


def _save_ndarray_as_png(arr: np.ndarray, path: str):
    try:
        cv2.imwrite(path, arr)
    except Exception as e:
        # Fallback PIL si OpenCV fallara en esta plataforma
        Image.fromarray(arr).save(path)


def ocr_pdf_by_pages(pdf_path: str, language: str) -> Tuple[List[str], int]:
    """
    Recorre el PDF página por página, usando texto embebido cuando existe y,
    si no, conviertiendo esa página a PNG -> mejora -> OCR.
    """
    pages_text: List[str] = []
    pages_processed = 0

    with tempfile.TemporaryDirectory() as tmp:
        try:
            info = pdfinfo_from_path(pdf_path)
            total = int(info.get("Pages", 0))
        except Exception:
            # Fallback a PyPDF2 si pdfinfo falla
            reader = PdfReader(pdf_path)
            total = len(reader.pages)

        reader = None
        try:
            reader = PdfReader(pdf_path)
        except Exception:
            pass

        for p in range(total):
            # 1) Intentar texto embebido
            if reader is not None:
                ok, txt = _page_has_text(reader, p)
                if ok:
                    pages_text.append(txt)
                    pages_processed += 1
                    continue

            # 2) Convertir SOLO la página p a imagen (memoria-amigable)
            png_paths = convert_from_path(
                pdf_path,
                dpi=PDF_DPI,
                first_page=p + 1,
                last_page=p + 1,
                fmt="png",
                output_folder=tmp,
                paths_only=True,
                thread_count=1,
            )
            if not png_paths:
                pages_text.append("")
                pages_processed += 1
                continue

            png_path = png_paths[0]

            # 3) Corregir orientación por OSD
            _fix_orientation_inplace(png_path)

            # 4) Mejora agresiva
            enh = enhance_image_for_ocr(png_path)
            if enh is None:
                # Si algo falla, usar la imagen original
                img_for_ocr = png_path
            else:
                proc_path = os.path.join(tmp, f"proc_{uuid.uuid4().hex}.png")
                _save_ndarray_as_png(enh, proc_path)
                img_for_ocr = proc_path

            # 5) OCR
            text = perform_advanced_ocr(img_for_ocr, language=language)
            pages_text.append(text)
            pages_processed += 1

            # Limpieza por página
            try:
                for f in (png_path, img_for_ocr):
                    if f and os.path.exists(f):
                        os.remove(f)
            except Exception:
                pass

            gc.collect()

    return pages_text, pages_processed


# ----------------------------
# Rutas
# ----------------------------
@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.post("/ocr")
def ocr_endpoint():
    """
    multipart/form-data:
      - file: PDF/JPG/PNG
      - language (opcional): por defecto "spa+eng"
    """
    t0 = time.time()

    if "file" not in request.files:
        return jsonify({"error": "file is required (multipart/form-data)"}), 400

    lang = request.form.get("language", DEFAULT_LANG).strip() or DEFAULT_LANG

    f = request.files["file"]
    filename = f.filename or f"upload-{uuid.uuid4().hex}"
    ext = (os.path.splitext(filename)[1] or "").lower()

    with tempfile.TemporaryDirectory() as tmp:
        upload_path = os.path.join(tmp, f"upload{ext if ext else ''}")
        f.save(upload_path)

        pages_text: List[str] = []
        pages_processed = 0

        try:
            if ext == ".pdf" or open(upload_path, "rb").read(4) == b"%PDF":
                pages_text, pages_processed = ocr_pdf_by_pages(upload_path, language=lang)
            else:
                # Imagen suelta
                # Guardar una copia PNG si hace falta
                img_in = upload_path
                if ext not in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"):
                    # Convertir a png con PIL
                    im = Image.open(upload_path).convert("RGB")
                    img_in = os.path.join(tmp, "input.png")
                    im.save(img_in)

                _fix_orientation_inplace(img_in)
                enh = enhance_image_for_ocr(img_in)
                if enh is None:
                    img_for_ocr = img_in
                else:
                    proc_path = os.path.join(tmp, f"proc_{uuid.uuid4().hex}.png")
                    _save_ndarray_as_png(enh, proc_path)
                    img_for_ocr = proc_path

                text = perform_advanced_ocr(img_for_ocr, language=lang)
                pages_text = [text]
                pages_processed = 1

        except Exception as e:
            logger.exception("OCR failed")
            return jsonify({"error": str(e)}), 500

    elapsed = time.time() - t0
    full_text = "\n\n".join(pages_text).strip()

    return jsonify(
        {
            "success": True,
            "language": lang,
            "pages_processed": pages_processed,
            "processing_time": round(elapsed, 2),
            "pages_text": pages_text,
            "text": full_text,
        }
    )


# ----------------------------
# Main (para desarrollo local)
# En producción usar Gunicorn:
#   gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 300 app:app
# ----------------------------
if __name__ == "__main__":
    # Útil para pruebas locales (no usar en producción)
    app.run(host="0.0.0.0", port=5000, debug=False)
