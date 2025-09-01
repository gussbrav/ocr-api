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
from typing import Tuple, List, Dict, Any

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

import pytesseract
from PyPDF2 import PdfReader
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image, ImageEnhance

# Opcionales (para extracción de tablas nativas y CSVs)
try:
    import pdfplumber  # type: ignore
except Exception:
    pdfplumber = None
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None

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
    (r'(\d{4})-(\d{4})-(\w{3})-(\w{2})-(\w{2})', r'\1-\2-\3-\4-\5'),
]

# Patrones de dominio (Poder Judicial, PNP, SUNARP, etc.)
FIELD_PATTERNS: Dict[str, List[re.Pattern]] = {
    'expediente': [
        re.compile(r'EXPEDIENTE\s*[:\-]?\s*([A-Z0-9\-\./]+)', re.I),
    ],
    'juzgado': [
        re.compile(r'JUZGADO\s*[:\-]?\s*(.+)', re.I),
        re.compile(r'JUZ\.\s*INVESTIGACI[ÓO]N\s*PREP\.\s*DE\s*(.+)', re.I),
    ],
    'juez': [
        re.compile(r'JUEZ(?:A)?\s*[:\-]?\s*([A-ZÁÉÍÓÚÑ\s\.]+)', re.I),
    ],
    'especialista': [
        re.compile(r'ESPECIALISTA(?:\s+LEGAL)?\s*[:\-]?\s*([A-ZÁÉÍÓÚÑ\s\.]+)', re.I),
    ],
    'imputado': [
        re.compile(r'IMPUTADO\s*[:\-]?\s*(.+)', re.I),
    ],
    'agraviado': [
        re.compile(r'AGRAVIADO\s*[:\-]?\s*(.+)', re.I),
    ],
    'delito': [
        re.compile(r'DELITO\s*[:\-]?\s*(.+)', re.I),
    ],
    'carpeta_fiscal': [
        re.compile(r'CARPETA\s+FISCAL\s*[:\-]?\s*([A-Z0-9\-\./]+)', re.I),
    ],
    'placa': [
        re.compile(r'Placa(?:\s*Nro\.)?\s*[:\-]?\s*([A-Z0-9\-]+)', re.I),
        re.compile(r'PLACA\s+NRO\.\s*[:\-]?\s*([A-Z0-9\-]+)', re.I),
    ],
    'marca': [
        re.compile(r'Marca\s*[:\-]?\s*([A-Z0-9\-\s]+)', re.I),
    ],
    'color': [
        re.compile(r'Color\s*[:\-]?\s*([A-ZÁÉÍÓÚÑ\s\-]+)', re.I),
    ],
    'anio': [
        re.compile(r'Añ[oó]\s*[:\-]?\s*(\d{4})', re.I),
        re.compile(r'Año\s*[:\-]?\s*(\d{4})', re.I),
    ],
}

# Normalización de claves “clave → canonical”
KEY_NORMALIZATION = {
    'expediente': 'expediente',
    'juzgado': 'juzgado',
    'juez': 'juez',
    'especialista': 'especialista',
    'es. legal': 'especialista',
    'imputado': 'imputado',
    'agraviado': 'agraviado',
    'delito': 'delito',
    'carpeta fiscal': 'carpeta_fiscal',
    'placa': 'placa',
    'placa nro.': 'placa',
    'marca': 'marca',
    'color': 'color',
    'año': 'anio', 'ano': 'anio',
    'f.inicio': 'fecha_inicio',
    'fecha': 'fecha',
    'hora': 'hora',
}

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
# Estructurado (KV, campos, tablas, markdown)
# ----------------------------
def _normalize_key(k: str) -> str:
    k0 = re.sub(r'[^A-Za-zÁÉÍÓÚÑáéíóú0-9\. ]+', '', k).strip().lower()
    k0 = k0.replace('  ', ' ')
    return KEY_NORMALIZATION.get(k0, k0)

def extract_kv_pairs(text: str) -> Dict[str, str]:
    """
    Extrae pares clave:valor de líneas con ':' o '|' y normaliza claves.
    """
    kv: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if '|' in line:
            parts = [p.strip(' :.-') for p in line.split('|') if p.strip()]
            if len(parts) >= 2 and len(parts[0]) <= 40:
                key = _normalize_key(parts[0])
                val = ' | '.join(parts[1:]).strip()
                if key and val and key not in kv:
                    kv[key] = val
                continue
        if ':' in line:
            left, right = line.split(':', 1)
            key = _normalize_key(left)
            val = right.strip(' .-')
            if key and val and key not in kv and len(left) <= 40:
                kv[key] = val
    return kv

def extract_field_patterns(text: str) -> Dict[str, str]:
    """
    Aplica regex de dominio para capturar campos importantes (expediente, juzgado, etc.).
    """
    found: Dict[str, str] = {}
    # priorizar primeras páginas (menos ruido)
    head = '\n'.join(text.splitlines()[:500])
    for fname, patterns in FIELD_PATTERNS.items():
        for pat in patterns:
            m = pat.search(head)
            if m:
                val = m.group(1).strip().strip(' .-')
                if val:
                    found[fname] = val
                    break
    return found

def extract_ascii_tables_from_text(text: str) -> List['pd.DataFrame']:
    """
    Detección simple de tablas 'ASCII' (líneas con pipes '|') en texto OCR.
    """
    dfs: List['pd.DataFrame'] = []
    if pd is None:
        return dfs
    blocks = text.split('\n\n')
    for blk in blocks:
        lines = [l for l in blk.splitlines() if l.strip()]
        if len(lines) < 2:
            continue
        if sum(1 for l in lines if '|' in l) >= 2:
            rows = []
            for l in lines:
                if '|' in l:
                    parts = [p.strip() for p in l.split('|')]
                    # quitar separadores típicos
                    parts = [re.sub(r'^-+$', '', p) for p in parts]
                    rows.append(parts)
            # Normalizar ancho
            maxlen = max((len(r) for r in rows), default=0)
            norm = [r + [''] * (maxlen - len(r)) for r in rows]
            if norm:
                df = pd.DataFrame(norm)
                # usar primera fila como cabecera si es razonable
                if df.shape[0] >= 2:
                    df.columns = [c if c else f'col{i+1}' for i, c in enumerate(df.iloc[0].tolist())]
                    df = df.iloc[1:].reset_index(drop=True)
                dfs.append(df)
    return dfs

def extract_tables_pdf_native(pdf_path: str) -> List['pd.DataFrame']:
    """
    Usa pdfplumber para extraer tablas de PDFs con texto nativo.
    """
    dfs: List['pd.DataFrame'] = []
    if pdfplumber is None or pd is None:
        return dfs
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                try:
                    tables = page.extract_tables()
                    for t in tables or []:
                        if not t or len(t) < 2:
                            continue
                        df = pd.DataFrame(t[1:], columns=[c or f'col{i+1}' for i, c in enumerate(t[0])])
                        dfs.append(df)
                except Exception:
                    continue
    except Exception as e:
        logger.debug(f"pdfplumber failed: {e}")
    return dfs

def save_tables_to_csv(dfs: List['pd.DataFrame'], out_dir: str, prefix: str = 'table') -> List[str]:
    paths: List[str] = []
    if pd is None:
        return paths
    os.makedirs(out_dir, exist_ok=True)
    for i, df in enumerate(dfs, 1):
        fn = os.path.join(out_dir, f'{prefix}_{i:02d}.csv')
        try:
            df.to_csv(fn, index=False)
            paths.append(fn)
        except Exception:
            # fallback simple
            with open(fn, 'w', encoding='utf-8') as f:
                for idx, row in df.iterrows():
                    f.write(','.join(str(x) for x in row.tolist()) + '\n')
            paths.append(fn)
    return paths

def build_markdown(meta: Dict[str, Any], fields: Dict[str, str], kv: Dict[str, str],
                   tables_preview: List['pd.DataFrame'], plain_text: str) -> str:
    """
    Genera un Markdown con estilo 'index extract'.
    """
    lines = []
    lines.append(f"# METADATOS")
    for k, v in meta.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    if fields:
        lines.append("# CAMPOS")
        for k, v in fields.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")
    if kv:
        lines.append("# KV DETECTADOS")
        for k, v in kv.items():
            if k in fields:
                continue
            lines.append(f"- **{k}**: {v}")
        lines.append("")
    if tables_preview and pd is not None:
        lines.append("# TABLAS (vista previa)")
        for i, df in enumerate(tables_preview[:5], 1):
            lines.append(f"## Tabla {i}")
            # to markdown
            try:
                md = df.head(10).to_markdown(index=False)
            except Exception:
                md = '\n'.join([' | '.join(map(str, r)) for r in df.head(10).values.tolist()])
            lines.append(md)
            lines.append("")
    lines.append("# TEXTO")
    lines.append(plain_text.strip())
    return '\n'.join(lines)

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
        'version': '2.3.0',
        'dpi': OCR_DPI,
        'lang_default': OCR_LANG_DEFAULT,
        'pdfplumber': bool(pdfplumber is not None),
        'pandas': bool(pd is not None)
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

def _structured_outputs(pdf_path: str, text: str, out_base: str, need_tables: bool) -> Dict[str, Any]:
    """
    Genera campos clave, kv y tablas (si se puede) + Markdown.
    """
    kv = extract_kv_pairs(text)
    fields = extract_field_patterns(text)

    table_dfs: List['pd.DataFrame'] = []
    table_csvs: List[str] = []

    if need_tables:
        # Si el PDF es nativo, usa pdfplumber; de lo contrario, intenta tablas ASCII del OCR.
        if pdfplumber is not None:
            try:
                # Heurística: si el PDF NO es bitmap, pdfplumber tendrá mejores resultados
                if not contains_bitmap(pdf_path):
                    table_dfs = extract_tables_pdf_native(pdf_path)
            except Exception:
                pass
        if not table_dfs:
            table_dfs = extract_ascii_tables_from_text(text)

        if table_dfs and pd is not None:
            tables_dir = os.path.join(app.config['OUTPUT_FOLDER'], f'{out_base}_tables')
            table_csvs = save_tables_to_csv(table_dfs, tables_dir, prefix='table')

    # Markdown
    meta = {
        'archivo': out_base,
        'generado': datetime.now().isoformat(),
    }
    md = build_markdown(meta, fields, kv, table_dfs, text)

    return {
        'fields': fields,
        'kv_pairs': kv,
        'tables_csv': [os.path.basename(p) for p in table_csvs],
        'markdown': md
    }

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
        output_format = request.form.get('format', 'txt')  # txt | json | md | zip
        want_structured = request.form.get('structured', '0') == '1'

        # Guardar archivo
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        file.save(file_path)

        # Salida
        base_name = f"{timestamp}_{os.path.splitext(filename)[0]}"
        txt_filename = f"{base_name}.txt"
        txt_path = os.path.join(app.config['OUTPUT_FOLDER'], txt_filename)

        logger.info(f"Processing file: {temp_filename} | lang={language} | dpi={OCR_DPI}")

        load_custom_corrections()
        start_time = datetime.now()

        # Elección de estrategia
        if contains_bitmap(file_path):
            logger.info("Mode: enhanced OCR (bitmap/scan)")
            success, pages_processed, text = process_bitmap_pdf(file_path, txt_path, language)
            method = 'enhanced_ocr'
        else:
            logger.info("Mode: text extraction with corrections")
            success, pages_processed, text = process_text_pdf(file_path, txt_path)
            method = 'text_extraction_corrected'

        processing_time = (datetime.now() - start_time).total_seconds()

        if not success:
            return jsonify({
                'success': False,
                'error': text,
                'processing_time': round(processing_time, 2)
            }), 500

        # Si el cliente quiere estructurado o formatos ricos, generarlo
        extra: Dict[str, Any] = {}
        md_path = None
        json_doc = None

        if want_structured or output_format in ('json', 'md', 'zip'):
            extra = _structured_outputs(file_path, text, base_name, need_tables=True)

            # Guardar markdown si se va a devolver/zip
            md_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{base_name}.md")
            with open(md_path, 'w', encoding='utf-8') as fmd:
                fmd.write(extra['markdown'])

            # JSON estructurado
            json_doc = {
                'success': True,
                'filename': filename,
                'pages_processed': pages_processed,
                'processing_time': round(processing_time, 2),
                'method': method,
                'language': language,
                'corrections_applied': len(CHAR_CORRECTIONS),
                'timestamp': datetime.now().isoformat(),
                'text': text,
                'fields': extra.get('fields', {}),
                'kv_pairs': extra.get('kv_pairs', {}),
                'tables_csv': extra.get('tables_csv', []),
            }

        # Responder según formato
        if output_format == 'json':
            return jsonify(json_doc if json_doc else {
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

        if output_format == 'md' and md_path:
            return send_file(md_path, as_attachment=True, download_name=f"{os.path.splitext(filename)[0]}.md")

        if output_format == 'zip':
            # Empaquetar: txt + (md) + (json) + tablas
            zip_name = f"{base_name}.zip"
            zip_path = os.path.join(app.config['OUTPUT_FOLDER'], zip_name)
            import json as _json
            with zipfile.ZipFile(zip_path, 'w') as z:
                z.write(txt_path, arcname=os.path.basename(txt_path))
                if md_path and os.path.exists(md_path):
                    z.write(md_path, arcname=os.path.basename(md_path))
                # JSON
                if json_doc:
                    z.writestr(f"{base_name}.json", _json.dumps(json_doc, ensure_ascii=False, indent=2))
                # tablas
                tables_dir = os.path.join(app.config['OUTPUT_FOLDER'], f'{base_name}_tables')
                if os.path.isdir(tables_dir):
                    for root, _, files in os.walk(tables_dir):
                        for f in files:
                            z.write(os.path.join(root, f), arcname=f)
            return send_file(zip_path, as_attachment=True, download_name=os.path.basename(zip_path))

        # Por defecto: txt
        return send_file(txt_path, as_attachment=True, download_name=f"{os.path.splitext(filename)[0]}.txt")

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
        structured = request.form.get('structured', '0') == '1'
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

                    base_name = os.path.splitext(filename)[0]
                    out_txt = os.path.join(batch_folder, f"{base_name}.txt")

                    if contains_bitmap(file_path):
                        success, pages, text = process_bitmap_pdf(file_path, out_txt, language)
                        method = 'enhanced_ocr'
                    else:
                        success, pages, text = process_text_pdf(file_path, out_txt)
                        method = 'text_extraction_corrected'

                    item = {
                        'filename': filename,
                        'success': success,
                        'pages': pages,
                        'method': method,
                        'corrections_applied': len(CHAR_CORRECTIONS),
                        'error': None if success else text
                    }

                    if success and structured:
                        extra = _structured_outputs(file_path, text, base_name, need_tables=True)
                        # Guardar MD
                        md_path = os.path.join(batch_folder, f"{base_name}.md")
                        with open(md_path, 'w', encoding='utf-8') as fmd:
                            fmd.write(extra['markdown'])
                        item['fields'] = extra.get('fields', {})
                        item['kv_pairs'] = extra.get('kv_pairs', {})
                        item['tables_csv'] = extra.get('tables_csv', [])

                    results.append(item)

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
        zip_filename = f'ocr_batch_{timestamp}.zip'
        zip_path = os.path.join(app.config['OUTPUT_FOLDER'], zip_filename)

        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # Agregar archivos
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
    # En producción usa Gunicorn (y sube el timeout si hay PDFs grandes):
    # gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 900 app:app
    app.run(host='0.0.0.0', port=5000, debug=False)
