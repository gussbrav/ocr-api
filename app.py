import os
import io
import cv2
import gc
import re
import zipfile
import logging
import tempfile
import numpy as np
from datetime import datetime
from typing import Tuple, List

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

import pytesseract
from PyPDF2 import PdfReader
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image, ImageEnhance

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# App & CORS
# ----------------------------
app = Flask(__name__)
CORS(app)

# ----------------------------
# Config
# ----------------------------
app.config['UPLOAD_FOLDER']   = '/app/input'
app.config['OUTPUT_FOLDER']   = '/app/output'
app.config['TEMP_FOLDER']     = '/app/temp'
app.config['TRAINING_FOLDER'] = '/app/training'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

for p in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER'],
          app.config['TEMP_FOLDER'], app.config['TRAINING_FOLDER']]:
    os.makedirs(p, exist_ok=True)

# Tesseract (en Debian/Ubuntu dentro de Docker suele estar ahí)
pytesseract.pytesseract.tesseract_cmd = os.environ.get('TESSERACT_CMD', '/usr/bin/tesseract')

# OCR knobs via env
OCR_LANG_DEFAULT = os.getenv('OCR_LANG', 'spa+eng')
OCR_DPI = int(os.getenv('OCR_DPI', '450'))                  # DPI para rasterizar PDFs
OCR_SCALE = float(os.getenv('OCR_SCALE', '3.0'))            # Upscale de imagen antes del OCR (3.0 ~ 300%)
OSD_ENABLED = os.getenv('OSD_ENABLED', '1') == '1'          # Corrección de orientación con OSD

# ----------------------------
# Correcciones y patrones
# ----------------------------
CHAR_CORRECTIONS = {
    'N0': 'N°', 'NO': 'N°', 'Np': 'N°', 'N8': 'N°', 'N6': 'N°',
    'Nro': 'Nro', 'NRO': 'Nro', 'nro': 'Nro', 'Nr0': 'Nro', 'Nrp': 'Nro',
    'Num': 'Núm', 'NUM': 'Núm', 'núm': 'Núm', 'numero': 'número',
    'NUMERO': 'NÚMERO', 'JUZGADO': 'JUZGADO', 'CIVIL': 'CIVIL', 'PENAL': 'PENAL',
    'EXPEDIENTE': 'EXPEDIENTE', 'EXP': 'EXP', 'RESOLUCION': 'RESOLUCIÓN',
    'RESOLUCIÖN': 'RESOLUCIÓN', 'DIGITALIZACION': 'DIGITALIZACIÓN',
    'DIGITALIZACIÖN': 'DIGITALIZACIÓN'
}

LEGAL_PATTERNS = [
    (r'N[0O8p6]\s*(\d+)', r'N° \1'),
    (r'Nr[0o]\s*(\d+)', r'Nro \1'),
    (r'[Nn][úu]m\s*(\d+)', r'Núm \1'),
    (r'EXP\s*[\.:]?\s*N[0O8p6]\s*(\d+)', r'EXP N° \1'),
    (r'EXPEDIENTE\s*N[0O8p6]\s*(\d+)', r'EXPEDIENTE N° \1'),
    (r'RESOLUCI[ÖO]N\s*N[0O8p6]\s*(\d+)', r'RESOLUCIÓN N° \1'),
    # formato de expediente habitual ya está OK, lo mantenemos como recordatorio:
    (r'(\d{4})-(\d{4})-(\w{3})-(\w{2})-(\w{2})', r'\1-\2-\3-\4-\5'),
]

# ----------------------------
# Helpers básicos
# ----------------------------
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def contains_bitmap(pdf_path: str) -> bool:
    """
    Heurística: si el texto embebido es muy poco o hay XObjects /Image,
    tratamos como bitmap para hacer OCR.
    """
    try:
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            text_len = 0
            has_images = False

            for page in reader.pages:
                txt = page.extract_text() or ""
                text_len += len(re.sub(r"\s+", "", txt))

                # detección simple de imágenes
                try:
                    s = str(page.get_object())
                    if re.search(r'/Subtype\s*/Image', s, re.IGNORECASE):
                        has_images = True
                except Exception:
                    pass

            # si casi no hay texto o hay imágenes por todos lados -> bitmap
            return text_len < 100 or has_images
    except Exception as e:
        logger.warning(f"Error checking PDF content: {e}")
        # En duda, tratamos como bitmap
        return True

def get_page_separator(page_num: int, total_pages: int) -> str:
    return f"\n\n--- PÁGINA {page_num} DE {total_pages}\n\n"

# ----------------------------
# Imagen: orientación, deskew, limpieza de líneas
# ----------------------------
def _fix_orientation_inplace(png_path: str):
    if not OSD_ENABLED:
        return
    try:
        osd = pytesseract.image_to_osd(png_path)
        m = re.search(r"Rotate:\s+(\d+)", osd)
        rot = int(m.group(1)) if m else 0
        if rot % 360:
            img = Image.open(png_path).convert("RGB")
            img = img.rotate(360 - rot, expand=True, fillcolor="white")
            img.save(png_path)
    except Exception:
        pass

def _deskew(binary_img: np.ndarray) -> np.ndarray:
    try:
        coords = np.column_stack(np.where(binary_img == 0))
        if coords.size == 0:
            return binary_img
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        (h, w) = binary_img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(binary_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return binary_img

def _remove_horizontal_lines(bin_img: np.ndarray) -> np.ndarray:
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
    Pipeline robusto para documentos legales escaneados:
    upscale -> CLAHE -> des-ruido -> Otsu -> limpiar líneas -> deskew.
    Devuelve imagen binaria (np.uint8).
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            return None

        # Escalado
        if OCR_SCALE and OCR_SCALE != 1.0:
            img = cv2.resize(img, None, fx=OCR_SCALE, fy=OCR_SCALE, interpolation=cv2.INTER_CUBIC)

        # PIL para mejorar contraste un poco
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pil = ImageEnhance.Contrast(pil).enhance(1.4)
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Des-ruido ligero
        gray = cv2.fastNlMeansDenoising(gray, h=14)

        # Umbral Otsu
        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Quitar líneas horizontales y deskew
        thr = _remove_horizontal_lines(thr)
        thr = _deskew(thr)

        # Cierre leve
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((1, 1), np.uint8))

        return thr
    except Exception as e:
        logger.error(f"Error enhancing image: {e}")
        return None

# ----------------------------
# Post-procesado de texto
# ----------------------------
def correct_ocr_text(text: str) -> str:
    t = text

    # Diccionario
    for wrong, correct in CHAR_CORRECTIONS.items():
        t = re.sub(r'\b' + re.escape(wrong) + r'\b', correct, t, flags=re.IGNORECASE)

    # Patrones
    for pattern, replacement in LEGAL_PATTERNS:
        t = re.sub(pattern, replacement, t, flags=re.IGNORECASE)

    # Limpiezas suaves
    t = re.sub(r'[ \t]{2,}', ' ', t)
    t = re.sub(r'\s+\n', '\n', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    # Unir palabras cortadas por guion al final de línea
    t = re.sub(r'(\w)-\n(\w)', r'\1\2\n', t)

    return t.strip()

# ----------------------------
# OCR multipase con selección por confianza
# ----------------------------
def _avg_conf_from_data(data_dict) -> float:
    vals = []
    for c in data_dict.get('conf', []):
        try:
            v = int(float(c))
            if v > 0:
                vals.append(v)
        except Exception:
            continue
    return float(np.mean(vals)) if vals else 0.0

def perform_advanced_ocr(image_path: str, language: str = 'spa') -> str:
    base_cfg = f"-c preserve_interword_spaces=1 -c user_defined_dpi={max(300, OCR_DPI)}"
    configs = [
        f"--psm 6  --oem 1 {base_cfg}",
        f"--psm 4  --oem 1 {base_cfg}",
        f"--psm 11 --oem 1 {base_cfg}",
        f"--psm 6  --oem 1 {base_cfg} -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNÑOPQRSTUVWXYZabcdefghijklmnñopqrstuvwxyzÁÉÍÓÚáéíóú°:-./,()",
    ]

    best_text, max_conf = "", -1.0
    for cfg in configs:
        try:
            data = pytesseract.image_to_data(image_path, lang=language, config=cfg, output_type=pytesseract.Output.DICT)
            avg = _avg_conf_from_data(data)
            text = pytesseract.image_to_string(image_path, lang=language, config=cfg)
            text = correct_ocr_text(text)
            if avg >= max_conf and text.strip():
                best_text, max_conf = text, avg
        except Exception as e:
            logger.debug(f"OCR pass failed: {e}")

    if not best_text.strip():
        # Fallback
        text = pytesseract.image_to_string(image_path, lang=language)
        best_text = correct_ocr_text(text)

    return best_text

# ----------------------------
# Procesamiento de PDFs
# ----------------------------
def process_text_pdf(pdf_path: str, output_path: str) -> Tuple[bool, int, str]:
    try:
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            total = len(reader.pages)
            all_text = ""

            for i, page in enumerate(reader.pages, 1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    all_text += get_page_separator(i, total)
                    all_text += correct_ocr_text(page_text)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(all_text)

        return True, total, all_text
    except Exception as e:
        logger.error(f"Error processing text PDF: {e}")
        return False, 0, str(e)

def _save_ndarray_as_png(arr: np.ndarray, path: str):
    try:
        cv2.imwrite(path, arr)
    except Exception:
        Image.fromarray(arr).save(path)

def process_bitmap_pdf(pdf_path: str, output_path: str, language: str = 'spa') -> Tuple[bool, int, str]:
    """
    Procesa **página por página** para ser amigable con memoria en PDFs grandes.
    No carga todas las páginas a la vez.
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                info = pdfinfo_from_path(pdf_path)
                total_pages = int(info.get("Pages", 0))
            except Exception:
                reader = PdfReader(pdf_path)
                total_pages = len(reader.pages)

            all_text = ""

            for i in range(1, total_pages + 1):
                try:
                    # Convertir solo esta página a PNG en disco (no PIL en memoria)
                    png_paths = convert_from_path(
                        pdf_path,
                        dpi=OCR_DPI,
                        first_page=i,
                        last_page=i,
                        fmt="png",
                        output_folder=temp_dir,
                        paths_only=True,
                        thread_count=1,
                    )
                    if not png_paths:
                        all_text += get_page_separator(i, total_pages)
                        all_text += "(Página vacía o no legible)"
                        continue

                    png_path = png_paths[0]

                    # Orientación
                    _fix_orientation_inplace(png_path)

                    # Mejora agresiva
                    enhanced = enhance_image_for_ocr(png_path)
                    if enhanced is not None:
                        proc_path = os.path.join(temp_dir, f"proc_{i}.png")
                        _save_ndarray_as_png(enhanced, proc_path)
                        img_for_ocr = proc_path
                    else:
                        img_for_ocr = png_path

                    # OCR
                    page_text = perform_advanced_ocr(img_for_ocr, language=language)

                    all_text += get_page_separator(i, total_pages)
                    all_text += page_text

                    # Limpieza por página
                    try:
                        for fdel in [png_path] + ([img_for_ocr] if img_for_ocr != png_path else []):
                            if os.path.exists(fdel):
                                os.remove(fdel)
                    except Exception:
                        pass

                    gc.collect()

                except Exception as e:
                    logger.error(f"Error processing page {i}: {e}")
                    all_text += get_page_separator(i, total_pages)
                    all_text += f"Error procesando página: {str(e)}"

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(all_text)

            return True, total_pages, all_text

    except Exception as e:
        logger.error(f"Error processing bitmap PDF: {e}")
        return False, 0, str(e)

# ----------------------------
# Entrypoints
# ----------------------------
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.1.0',
        'dpi': OCR_DPI,
        'lang_default': OCR_LANG_DEFAULT
    })

@app.route('/ocr/train', methods=['POST'])
def train_corrections():
    try:
        data = request.get_json()
        if not data or 'corrections' not in data:
            return jsonify({'error': 'No corrections data provided'}), 400

        corrections_file = os.path.join(app.config['TRAINING_FOLDER'], 'custom_corrections.txt')

        with open(corrections_file, 'a', encoding='utf-8') as f:
            for wrong, correct in data['corrections'].items():
                f.write(f"{wrong}|{correct}\n")

        load_custom_corrections()

        return jsonify({
            'success': True,
            'message': f'Added {len(data["corrections"])} new corrections',
            'total_corrections': len(CHAR_CORRECTIONS)
        })
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        return jsonify({'error': f'Training failed: {str(e)}'}), 500

def load_custom_corrections():
    corrections_file = os.path.join(app.config['TRAINING_FOLDER'], 'custom_corrections.txt')
    if os.path.exists(corrections_file):
        try:
            with open(corrections_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if '|' in line:
                        wrong, correct = line.strip().split('|', 1)
                        CHAR_CORRECTIONS[wrong] = correct
        except Exception as e:
            logger.warning(f"Error loading custom corrections: {e}")

@app.route('/ocr/process', methods=['POST'])
def process_pdf():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Only PDF files are allowed'}), 400

        # Parámetros
        language = request.form.get('language', OCR_LANG_DEFAULT)
        output_format = request.form.get('format', 'txt')  # txt | json

        # Guardar archivo
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        file.save(file_path)

        # Salida
        output_filename = f"{timestamp}_{os.path.splitext(filename)[0]}.txt"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        logger.info(f"Processing file: {temp_filename} | lang={language} | dpi={OCR_DPI}")

        load_custom_corrections()
        start_time = datetime.now()

        # Elección de estrategia
        if contains_bitmap(file_path):
            logger.info("Mode: enhanced OCR (bitmap/scan)")
            success, pages_processed, text = process_bitmap_pdf(file_path, output_path, language)
            method = 'enhanced_ocr'
        else:
            logger.info("Mode: text extraction with corrections")
            success, pages_processed, text = process_text_pdf(file_path, output_path)
            method = 'text_extraction_corrected'

        processing_time = (datetime.now() - start_time).total_seconds()

        if success:
            if output_format == 'json':
                return jsonify({
                    'success': True,
                    'filename': filename,
                    'pages_processed': pages_processed,
                    'processing_time': round(processing_time, 2),
                    'method': method,
                    'language': language,
                    'corrections_applied': len(CHAR_CORRECTIONS),
                    'text': text,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return send_file(output_path, as_attachment=True,
                                 download_name=f"{os.path.splitext(filename)[0]}.txt")
        else:
            return jsonify({
                'success': False,
                'error': text,
                'processing_time': round(processing_time, 2)
            }), 500

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500
    finally:
        try:
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.warning(f"Could not remove temp file: {e}")

@app.route('/ocr/batch', methods=['POST'])
def process_batch():
    try:
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No files provided'}), 400

        language = request.form.get('language', OCR_LANG_DEFAULT)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_folder = os.path.join(app.config['OUTPUT_FOLDER'], f'batch_{timestamp}')
        os.makedirs(batch_folder, exist_ok=True)

        load_custom_corrections()
        results, temp_files = [], []

        for file in files:
            if file.filename and allowed_file(file.filename):
                try:
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(app.config['TEMP_FOLDER'], f"{timestamp}_{filename}")
                    file.save(file_path)
                    temp_files.append(file_path)

                    output_filename = f"{os.path.splitext(filename)[0]}.txt"
                    output_path = os.path.join(batch_folder, output_filename)

                    if contains_bitmap(file_path):
                        success, pages, text = process_bitmap_pdf(file_path, output_path, language)
                        method = 'enhanced_ocr'
                    else:
                        success, pages, text = process_text_pdf(file_path, output_path)
                        method = 'text_extraction_corrected'

                    results.append({
                        'filename': filename,
                        'success': success,
                        'pages': pages,
                        'method': method,
                        'corrections_applied': len(CHAR_CORRECTIONS),
                        'error': None if success else text
                    })
                except Exception as e:
                    results.append({
                        'filename': file.filename,
                        'success': False,
                        'error': str(e)
                    })

        # Limpieza temp
        for tmp in temp_files:
            try:
                os.remove(tmp)
            except Exception:
                pass

        # Crear ZIP con resultados + resumen
        zip_filename = f'ocr_batch_enhanced_{timestamp}.zip'
        zip_path = os.path.join(app.config['OUTPUT_FOLDER'], zip_filename)

        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for root, _, fs in os.walk(batch_folder):
                for f in fs:
                    zipf.write(os.path.join(root, f), f)
            summary = {
                'batch_id': timestamp,
                'processed_files': len(results),
                'successful': len([r for r in results if r['success']]),
                'failed': len([r for r in results if not r['success']]),
                'total_corrections_available': len(CHAR_CORRECTIONS),
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
            import json as _json
            zipf.writestr('batch_summary.json', _json.dumps(summary, indent=2, ensure_ascii=False))

        return send_file(zip_path, as_attachment=True, download_name=zip_filename)

    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}")
        return jsonify({'error': f'Batch processing failed: {str(e)}'}), 500

@app.route('/output/<filename>')
def get_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/', methods=['GET'])
def home():
    return send_from_directory(app.root_path, 'index.html')

# Cargar correcciones personalizadas al iniciar
def _bootstrap():
    try:
        load_custom_corrections()
    except Exception:
        pass

_bootstrap()

if __name__ == '__main__':
    # En producción usa Gunicorn:
    # gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 600 app:app
    app.run(host='0.0.0.0', port=5000, debug=False)
