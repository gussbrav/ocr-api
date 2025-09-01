import os
import io
import cv2
import gc
import re
import json
import zipfile
import logging
import tempfile
import numpy as np
from datetime import datetime
from typing import Tuple, List, Dict, Any

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
CHAR_CORRECTIONS: Dict[str, str] = {
    'N0': 'N°', 'NO': 'N°', 'Np': 'N°', 'N8': 'N°', 'N6': 'N°',
    'Nro': 'Nro', 'NRO': 'Nro', 'nro': 'Nro', 'Nr0': 'Nro', 'Nrp': 'Nro',
    'Num': 'Núm', 'NUM': 'Núm', 'núm': 'Núm', 'numero': 'número',
    'NUMERO': 'NÚMERO', 'JUZGADO': 'JUZGADO', 'CIVIL': 'CIVIL', 'PENAL': 'PENAL',
    'EXPEDIENTE': 'EXPEDIENTE', 'EXP': 'EXP', 'RESOLUCION': 'RESOLUCIÓN',
    'RESOLUCIÖN': 'RESOLUCIÓN', 'DIGITALIZACION': 'DIGITALIZACIÓN',
    'DIGITALIZACIÖN': 'DIGITALIZACIÓN',
    # Otras frecuentes en policial/fiscalía
    'FISCALIA': 'FISCALÍA', 'INVESTIGACION': 'INVESTIGACIÓN',
    'APROPIACION': 'APROPIACIÓN', 'DELITO': 'DELITO',
}

LEGAL_PATTERNS: List[Tuple[str, str]] = [
    (r'N[0O8p6]\s*(\d+)', r'N° \1'),
    (r'Nr[0o]\s*(\d+)', r'Nro \1'),
    (r'[Nn][úu]m\s*(\d+)', r'Núm \1'),
    (r'EXP\s*[\.:]?\s*N[0O8p6]\s*(\d+)', r'EXP N° \1'),
    (r'EXPEDIENTE\s*N[0O8p6]\s*(\d+)', r'EXPEDIENTE N° \1'),
    (r'RESOLUCI[ÖO]N\s*N[0O8p6]\s*(\d+)', r'RESOLUCIÓN N° \1'),
    # Formatos de expediente
    (r'(\d{4})-(\d{4})-(\w{2,4})-(\w{2})-(\w{2})', r'\1-\2-\3-\4-\5'),
    # Placas tipo ABC-123 o similares
    (r'\b([A-Z]{3})[-\s]?(\d{3})\b', r'\1-\2'),
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

                try:
                    s = str(page.get_object())
                    if re.search(r'/Subtype\s*/Image', s, re.IGNORECASE):
                        has_images = True
                except Exception:
                    pass

            return text_len < 200 or has_images
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

def _remove_lines(bin_img: np.ndarray) -> np.ndarray:
    """
    Elimina líneas horizontales y verticales para facilitar OCR de tablas/firmas.
    """
    try:
        inv = 255 - bin_img
        # horizontales
        hk = max(30, bin_img.shape[1] // 30)
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
        h_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
        # verticales
        vk = max(25, bin_img.shape[0] // 35)
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))
        v_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, vert_kernel, iterations=1)

        cleaned = bin_img.copy()
        cleaned[h_lines > 0] = 255
        cleaned[v_lines > 0] = 255
        return cleaned
    except Exception:
        return bin_img

def enhance_image_for_ocr(image_path: str) -> np.ndarray:
    """
    Pipeline robusto para documentos legales escaneados:
    upscale -> CLAHE -> des-ruido -> Otsu -> quitar líneas -> deskew -> cierre.
    Devuelve imagen binaria (np.uint8).
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            return None

        if OCR_SCALE and OCR_SCALE != 1.0:
            img = cv2.resize(img, None, fx=OCR_SCALE, fy=OCR_SCALE, interpolation=cv2.INTER_CUBIC)

        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pil = ImageEnhance.Contrast(pil).enhance(1.35)
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.fastNlMeansDenoising(gray, h=12)

        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        thr = _remove_lines(thr)
        thr = _deskew(thr)
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

    for wrong, correct in CHAR_CORRECTIONS.items():
        t = re.sub(r'\b' + re.escape(wrong) + r'\b', correct, t, flags=re.IGNORECASE)

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

def _ocr_text_from_data(image_path: str, lang: str, cfg: str) -> Tuple[str, float]:
    data = pytesseract.image_to_data(image_path, lang=lang, config=cfg, output_type=pytesseract.Output.DICT)
    avg = _avg_conf_from_data(data)
    text = pytesseract.image_to_string(image_path, lang=lang, config=cfg)
    return correct_ocr_text(text), avg

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
            text, avg = _ocr_text_from_data(image_path, language, cfg)
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
# Reconstrucción de líneas & tablas
# ----------------------------
def _reconstruct_lines(image_path: str, lang: str) -> List[str]:
    """
    Reconstruye líneas con image_to_data (mejor lectura que string plano).
    """
    cfg = f"--psm 6 --oem 1 -c preserve_interword_spaces=1 -c user_defined_dpi={max(300, OCR_DPI)}"
    d = pytesseract.image_to_data(image_path, lang=lang, config=cfg, output_type=pytesseract.Output.DICT)
    n = len(d['text'])
    lines: Dict[int, List[Tuple[int, str]]] = {}
    for i in range(n):
        if int(d['conf'][i]) < 0:
            continue
        text = d['text'][i].strip()
        if not text:
            continue
        ln = d.get('line_num', [0]*n)[i]
        left = d['left'][i]
        lines.setdefault(ln, []).append((left, text))

    out_lines = []
    for ln in sorted(lines.keys()):
        items = sorted(lines[ln], key=lambda x: x[0])
        s = ' '.join([w for _, w in items])
        out_lines.append(correct_ocr_text(s))
    return out_lines

def _detect_table_grid(bin_img: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    Estima posiciones de líneas horizontales y verticales (coordenadas y/x).
    Devuelve listas de y_lines y x_lines. Puede devolver listas vacías si no hay grid claro.
    """
    try:
        inv = 255 - bin_img
        # horizontales
        hk = max(20, bin_img.shape[1] // 40)
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
        h_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
        # verticales
        vk = max(20, bin_img.shape[0] // 35)
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))
        v_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, vert_kernel, iterations=1)

        # Proyecciones
        h_proj = np.sum(h_lines > 0, axis=1)
        v_proj = np.sum(v_lines > 0, axis=0)

        y_lines = [int(y) for y in np.where(h_proj > (0.5 * np.max(h_proj) if np.max(h_proj)>0 else 9999))[0]]
        x_lines = [int(x) for x in np.where(v_proj > (0.5 * np.max(v_proj) if np.max(v_proj)>0 else 9999))[0]]

        # Reducir líneas cercanas (merge)
        def _merge_lines(arr: List[int], gap: int = 8) -> List[int]:
            if not arr:
                return []
            arr = sorted(arr)
            merged = [arr[0]]
            for v in arr[1:]:
                if v - merged[-1] > gap:
                    merged.append(v)
            return merged

        return _merge_lines(y_lines), _merge_lines(x_lines)
    except Exception:
        return [], []

def _ocr_cell(img: np.ndarray, lang: str) -> str:
    try:
        cfg = f"--psm 7 --oem 1 -c user_defined_dpi={max(300, OCR_DPI)}"
        text = pytesseract.image_to_string(img, lang=lang, config=cfg)
        return correct_ocr_text(text).replace('\n', ' ').strip()
    except Exception:
        return ""

def extract_tables_from_binary(binary_img: np.ndarray, full_img_path: str, lang: str) -> List[Dict[str, Any]]:
    """
    Extrae tablas detectando grid. Si falla, intenta aproximación por líneas reconstruidas.
    Devuelve lista de tablas: {headers: [...], rows: [[...]] , markdown: "..."}
    """
    tables: List[Dict[str, Any]] = []
    try:
        y_lines, x_lines = _detect_table_grid(binary_img)
        if len(y_lines) >= 3 and len(x_lines) >= 2:
            # Generar celdas por pares de líneas
            y_pairs = list(zip(y_lines, y_lines[1:]))
            x_pairs = list(zip(x_lines, x_lines[1:]))

            # recortar márgenes
            pad = 2
            rows_text: List[List[str]] = []
            for (y1, y2) in y_pairs:
                row = []
                for (x1, x2) in x_pairs:
                    if y2 - y1 < 10 or x2 - x1 < 10:
                        row.append("")
                        continue
                    cell = binary_img[max(y1+pad,0):max(y2-pad,0), max(x1+pad,0):max(x2-pad,0)]
                    if cell.size == 0:
                        row.append("")
                        continue
                    # invertir a negro sobre blanco si hace falta
                    cell_for_ocr = 255 - cell if np.mean(cell) < 127 else cell
                    text = _ocr_cell(cell_for_ocr, lang)
                    row.append(text)
                rows_text.append(row)

            # headers heurístico: primera fila si tiene texto
            headers = rows_text[0] if rows_text and any(t.strip() for t in rows_text[0]) else []
            body = rows_text[1:] if headers else rows_text

            # Markdown
            md = []
            if headers:
                md.append('| ' + ' | '.join(h or '' for h in headers) + ' |')
                md.append('| ' + ' | '.join(['---']*len(headers)) + ' |')
            for r in body:
                md.append('| ' + ' | '.join(r) + ' |')

            tables.append({
                'headers': headers,
                'rows': body,
                'markdown': '\n'.join(md) if md else ''
            })
            return tables

        # Fallback: reconstruir líneas y buscar patrones de tabla simples
        lines = _reconstruct_lines(full_img_path, lang)
        # detectar tablas por pipes o puntos alineados
        block: List[str] = []
        for ln in lines:
            if '|' in ln or re.search(r'\s{6,}', ln):
                block.append(ln)
        if block:
            # convertir a markdown simple separando por pipes o gaps
            md_lines = []
            for ln in block:
                if '|' in ln:
                    parts = [p.strip() for p in ln.split('|') if p is not None]
                else:
                    parts = [p.strip() for p in re.split(r'\s{6,}', ln)]
                if parts:
                    md_lines.append('| ' + ' | '.join(parts) + ' |')
            if md_lines:
                tables.append({'headers': [], 'rows': [], 'markdown': '\n'.join(md_lines)})
    except Exception as e:
        logger.debug(f"table extraction failed: {e}")

    return tables

# ----------------------------
# Extracción de campos clave (estilo “llama index”)
# ----------------------------
KEY_PATTERNS: Dict[str, List[re.Pattern]] = {
    'expediente': [re.compile(r'\bEXPEDIENTE[:\s]+([A-Z0-9\-]+)', re.I)],
    'juzgado': [re.compile(r'\bJUZGADO[:\s]+(.+)', re.I)],
    'juez': [re.compile(r'\bJUEZ[:\s]+(.+)', re.I)],
    'especialista': [re.compile(r'\bESPECIALISTA(?: LEGAL)?:\s+(.+)', re.I)],
    'imputado': [re.compile(r'\bIMPUTADO[:\s]+(.+)', re.I)],
    'agraviado': [re.compile(r'\bAGRAVIADO[:\s]+(.+)', re.I)],
    'destinatario': [re.compile(r'\bDESTINATARIO[:\s]+(.+)', re.I)],
    'direccion': [re.compile(r'\bDIRECCI[ÓO]N[:\s]+(.+)', re.I)],
    'placa': [re.compile(r'\bPLACA(?: NRO\.)?[:\s]+([A-Z0-9\-]+)', re.I)],
    'marca': [re.compile(r'\bMARCA[:\s]+([A-Z0-9\-\s]+)', re.I)],
    'color': [re.compile(r'\bCOLOR[:\s]+([A-ZÁÉÍÓÚÑ\s]+)', re.I)],
    'anio':  [re.compile(r'\bAÑ?O[:\s]+(\d{4})', re.I)],
    'fecha': [re.compile(r'\bFECHA[:\s]+([\w\-/\.]+)', re.I)],
    'hora':  [re.compile(r'\bHORA[:\s]+([\w:\.]+)', re.I)],
    'motivo':[re.compile(r'\bMOTIVO[:\s]+(.+)', re.I)],
}

def extract_key_values(text: str) -> Dict[str, str]:
    kv: Dict[str, str] = {}
    for key, patterns in KEY_PATTERNS.items():
        for pat in patterns:
            m = pat.search(text)
            if m:
                val = m.group(1).strip()
                # Limpiar colas ...
                val = re.sub(r'\s{2,}', ' ', val)
                val = re.sub(r'[\|#]+$', '', val).strip()
                kv[key] = val
                break
    return kv

# ----------------------------
# Procesamiento de PDFs
# ----------------------------
def process_text_pdf(pdf_path: str, output_path: str) -> Tuple[bool, int, str, Dict[str, Any]]:
    """
    PDFs con texto embebido (no-escaneados).
    """
    try:
        all_text = ""
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            total = len(reader.pages)

            for i, page in enumerate(reader.pages, 1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    all_text += get_page_separator(i, total)
                    all_text += correct_ocr_text(page_text)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(all_text)

        meta = {'key_values': extract_key_values(all_text), 'tables': []}
        return True, total, all_text, meta
    except Exception as e:
        logger.error(f"Error processing text PDF: {e}")
        return False, 0, str(e), {}

def _save_ndarray_as_png(arr: np.ndarray, path: str):
    try:
        cv2.imwrite(path, arr)
    except Exception:
        Image.fromarray(arr).save(path)

def process_bitmap_pdf(pdf_path: str, output_path: str, language: str = 'spa', want_tables: bool = False
                      ) -> Tuple[bool, int, str, Dict[str, Any]]:
    """
    Procesa **página por página** para ser amigable con memoria en PDFs grandes.
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
            all_tables: List[Dict[str, Any]] = []

            for i in range(1, total_pages + 1):
                try:
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

                    # OCR texto
                    page_text = perform_advanced_ocr(img_for_ocr, language=language)

                    # Tablas
                    if want_tables and enhanced is not None:
                        try:
                            tbls = extract_tables_from_binary(enhanced, img_for_ocr, language)
                            if tbls:
                                all_tables.extend(tbls)
                                # Añadir versión markdown al texto plano para trazabilidad
                                for t in tbls:
                                    if t.get('markdown'):
                                        page_text += "\n\n" + t['markdown'] + "\n"
                        except Exception as e:
                            logger.debug(f"tables page {i} failed: {e}")

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

            # Escribir
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(all_text)

            meta = {
                'key_values': extract_key_values(all_text),
                'tables': all_tables
            }
            return True, total_pages, all_text, meta

    except Exception as e:
        logger.error(f"Error processing bitmap PDF: {e}")
        return False, 0, str(e), {}

# ----------------------------
# Entrypoints
# ----------------------------
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.3.0',
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
        output_format = request.form.get('format', 'txt')  # txt | json | md
        want_tables = request.form.get('tables', '0') == '1'
        force_bitmap = request.form.get('force_bitmap', '0') == '1'

        # Guardar archivo
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        file.save(file_path)

        # Salida
        output_filename = f"{timestamp}_{os.path.splitext(filename)[0]}.txt"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        logger.info(f"Processing file: {temp_filename} | lang={language} | dpi={OCR_DPI} | tables={want_tables}")

        load_custom_corrections()
        start_time = datetime.now()

        # Elección de estrategia
        is_bitmap = contains_bitmap(file_path) or force_bitmap
        if is_bitmap:
            logger.info("Mode: enhanced OCR (bitmap/scan)")
            success, pages_processed, text, meta = process_bitmap_pdf(file_path, output_path, language, want_tables)
            method = 'enhanced_ocr'
        else:
            logger.info("Mode: text extraction with corrections")
            success, pages_processed, text, meta = process_text_pdf(file_path, output_path)
            method = 'text_extraction_corrected'

        processing_time = (datetime.now() - start_time).total_seconds()

        if not success:
            return jsonify({
                'success': False,
                'error': text,
                'processing_time': round(processing_time, 2)
            }), 500

        # Respuestas por formato
        if output_format == 'json':
            return jsonify({
                'success': True,
                'filename': filename,
                'pages_processed': pages_processed,
                'processing_time': round(processing_time, 2),
                'method': method,
                'language': language,
                'corrections_applied': len(CHAR_CORRECTIONS),
                'key_values': meta.get('key_values', {}),
                'tables': meta.get('tables', []),
                'text': text,
                'timestamp': datetime.now().isoformat()
            })
        elif output_format == 'md':
            # Armar markdown con bloques
            md_parts = [f"# OCR de {filename}", f"_Método: {method} — Páginas: {pages_processed}_", ""]
            kv = meta.get('key_values') or {}
            if kv:
                md_parts.append("## Campos Extraídos")
                for k, v in kv.items():
                    md_parts.append(f"- **{k.title()}**: {v}")
                md_parts.append("")
            if meta.get('tables'):
                md_parts.append("## Tablas")
                for t in meta['tables']:
                    if t.get('markdown'):
                        md_parts.append(t['markdown'])
                        md_parts.append("")
            md_parts.append("## Texto")
            md_parts.append(text)
            md = "\n".join(md_parts)
            md_bytes = md.encode('utf-8')
            md_stream = io.BytesIO(md_bytes)
            md_stream.seek(0)
            return send_file(md_stream, as_attachment=True,
                             download_name=f"{os.path.splitext(filename)[0]}.md",
                             mimetype='text/markdown')
        else:
            # txt
            return send_file(output_path, as_attachment=True,
                             download_name=f"{os.path.splitext(filename)[0]}.txt")

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
        want_tables = request.form.get('tables', '0') == '1'
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
                        success, pages, text, meta = process_bitmap_pdf(file_path, output_path, language, want_tables)
                        method = 'enhanced_ocr'
                    else:
                        success, pages, text, meta = process_text_pdf(file_path, output_path)
                        method = 'text_extraction_corrected'

                    # guardar un .json por archivo con meta
                    meta_json = {
                        'filename': filename,
                        'pages': pages,
                        'method': method,
                        'key_values': meta.get('key_values', {}),
                        'tables': meta.get('tables', []),
                    }
                    with open(os.path.join(batch_folder, f"{os.path.splitext(filename)[0]}.json"), "w", encoding="utf-8") as jf:
                        json.dump(meta_json, jf, ensure_ascii=False, indent=2)

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
            zipf.writestr('batch_summary.json', json.dumps(summary, indent=2, ensure_ascii=False))

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
    # En producción usa Gunicorn (y eleva el timeout para PDFs grandes):
    # gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 900 app:app
    app.run(host='0.0.0.0', port=5000, debug=False)
