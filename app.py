import os
import io
import cv2
import gc
import re
import json
import math
import zipfile
import logging
import tempfile
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

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
logger = logging.getLogger("ocr.app")

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

# Tesseract executable
pytesseract.pytesseract.tesseract_cmd = os.environ.get('TESSERACT_CMD', '/usr/bin/tesseract')

# OCR knobs via env
OCR_LANG_DEFAULT = os.getenv('OCR_LANG', 'spa+eng')
OCR_DPI = int(os.getenv('OCR_DPI', '450'))
OCR_SCALE = float(os.getenv('OCR_SCALE', '3.0'))
OSD_ENABLED = os.getenv('OSD_ENABLED', '1') == '1'
OCR_WORKERS = int(os.getenv('OCR_WORKERS', max(1, os.cpu_count() // 2)))

# ----------------------------
# Correcciones y patrones
# ----------------------------
CHAR_CORRECTIONS: Dict[str, str] = {
    'Nro': 'Nro', 'NRO': 'Nro', 'nro': 'Nro', 'Nr0': 'Nro', 'Nrp': 'Nro',
    'Num': 'Núm', 'NUM': 'Núm', 'núm': 'Núm', 'numero': 'número',
    'NUMERO': 'NÚMERO', 'JUZGADO': 'JUZGADO', 'CIVIL': 'CIVIL', 'PENAL': 'PENAL',
    'EXPEDIENTE': 'EXPEDIENTE', 'EXP': 'EXP', 'RESOLUCION': 'RESOLUCIÓN',
    'RESOLUCIÖN': 'RESOLUCIÓN', 'DIGITALIZACION': 'DIGITALIZACIÓN',
    'DIGITALIZACIÖN': 'DIGITALIZACIÓN',
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
    # Placas ABC-123 o similares
    (r'\b([A-Z]{3})[-\s]?(\d{3})\b', r'\1-\2'),
]

# ----------------------------
# Helpers
# ----------------------------
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def contains_bitmap(pdf_path: str) -> bool:
    """
    Heurística: poco texto embebido o presencia de XObjects /Image => tratar como escaneo.
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
        logger.warning(f"Bitmap check fail: {e}")
        return True

def get_page_separator(page_num: int, total_pages: int) -> str:
    return f"\n\n--- PÁGINA {page_num} DE {total_pages}\n\n"

# ================
# Matemática avanzada
# ================

def homomorphic_illumination_correction(gray: np.ndarray) -> np.ndarray:
    """
    Filtrado homomórfico (FFT) para normalizar iluminación.
    """
    gray = gray.astype(np.float32) / 255.0 + 1e-6
    log_img = np.log(gray)
    h, w = gray.shape
    M, N = cv2.getOptimalDFTSize(h), cv2.getOptimalDFTSize(w)
    padded = np.zeros((M, N), dtype=np.float32)
    padded[:h, :w] = log_img

    dft = cv2.dft(padded, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft, axes=[0, 1])

    # Filtro pasa-altas gaussiano
    Y, X = np.ogrid[:M, :N]
    cy, cx = M//2, N//2
    D2 = (Y-cy)**2 + (X-cx)**2
    sigma = 30.0
    H = 1 - np.exp(-(D2/(2*sigma*sigma))).astype(np.float32)
    H = np.repeat(H[:, :, None], 2, axis=2)

    filtered = dft_shift * H
    ishift = np.fft.ifftshift(filtered, axes=[0, 1])
    inv = cv2.idft(ishift)
    inv = cv2.magnitude(inv[:, :, 0], inv[:, :, 1])

    inv = inv[:h, :w]
    exp_img = np.exp(inv)
    exp_img = (exp_img - exp_img.min()) / (exp_img.max() - exp_img.min() + 1e-6)
    out = np.clip(exp_img * 255, 0, 255).astype(np.uint8)
    return out

def estimate_skew_pca(binary: np.ndarray) -> float:
    """
    Estima skew con PCA sobre coordenadas de píxeles negros.
    Devuelve ángulo en grados (positivo = rotar antihorario).
    """
    coords = np.column_stack(np.where(binary == 0))
    if coords.size < 100:
        return 0.0
    # Centrar
    mean = coords.mean(axis=0)
    centered = coords - mean
    # PCA via SVD
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    v = Vt[0]
    angle = math.degrees(math.atan2(v[0], v[1]))
    # Queremos ángulo cercano a 0 (texto horizontal), ajustar rango
    if angle < -45:
        angle += 90
    if angle > 45:
        angle -= 90
    return angle

def estimate_skew_hough(binary: np.ndarray) -> float:
    """
    Estima skew midiendo líneas casi horizontales vía Hough.
    """
    edges = cv2.Canny(255 - binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=max(100, binary.shape[1]//8))
    if lines is None:
        return 0.0
    angles = []
    for rho_theta in lines[:200]:
        rho, theta = rho_theta[0]
        deg = (theta * 180/np.pi) - 90  # horizontales ~ 0
        if -30 <= deg <= 30:
            angles.append(deg)
    if not angles:
        return 0.0
    # Mediana robusta
    return float(np.median(angles))

def robust_skew_estimate(binary: np.ndarray) -> float:
    a1 = estimate_skew_pca(binary)
    a2 = estimate_skew_hough(binary)
    # combinación robusta (promedio ponderado por consistencia)
    if abs(a1 - a2) <= 2.0:
        return (a1 + a2) / 2.0
    # fallback: mediana
    return float(np.median([a1, a2, 0.0]))

def dewarp_perspective(color_img: np.ndarray) -> np.ndarray:
    """
    Busca el contorno más grande con 4 vértices y aplica transformación de perspectiva.
    Si falla, devuelve la original.
    """
    try:
        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        thr = cv2.Canny(blur, 60, 180)
        thr = cv2.dilate(thr, np.ones((3,3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours[:5]:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4,2).astype(np.float32)
                # ordenar esquinas: top-left, top-right, bottom-right, bottom-left
                s = pts.sum(axis=1); diff = np.diff(pts, axis=1)
                rect = np.zeros((4,2), dtype=np.float32)
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]
                (tl, tr, br, bl) = rect
                widthA = np.linalg.norm(br - bl)
                widthB = np.linalg.norm(tr - tl)
                heightA = np.linalg.norm(tr - br)
                heightB = np.linalg.norm(tl - bl)
                maxW = int(max(widthA, widthB))
                maxH = int(max(heightA, heightB))
                dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype=np.float32)
                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(color_img, M, (maxW, maxH), flags=cv2.INTER_CUBIC,
                                             borderMode=cv2.BORDER_REPLICATE)
                return warped
        return color_img
    except Exception:
        return color_img

# ----------------------------
# Pre-procesamiento avanzado
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

def _remove_lines(bin_img: np.ndarray) -> np.ndarray:
    inv = 255 - bin_img
    hk = max(30, bin_img.shape[1] // 30)
    vk = max(25, bin_img.shape[0] // 35)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    vert_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))
    h_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    v_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, vert_kernel, iterations=1)
    cleaned = bin_img.copy()
    cleaned[h_lines > 0] = 255
    cleaned[v_lines > 0] = 255
    return cleaned

def enhance_image_for_ocr(image_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Pipeline avanzado:
    - dewarping (perspectiva)
    - upscale + contraste
    - corrección de iluminación homomórfica (FFT)
    - reducción de ruido
    - binarización Otsu
    - eliminación de líneas
    - estimación robusta de skew (PCA+Hough) y deskew
    Devuelve binaria y versión color corregida (para recortes de celdas).
    """
    try:
        color = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if color is None:
            return None, None

        color = dewarp_perspective(color)

        if OCR_SCALE and OCR_SCALE != 1.0:
            color = cv2.resize(color, None, fx=OCR_SCALE, fy=OCR_SCALE, interpolation=cv2.INTER_CUBIC)

        pil = Image.fromarray(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
        pil = ImageEnhance.Contrast(pil).enhance(1.35)
        color = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        gray = homomorphic_illumination_correction(gray)
        # denoise suave
        gray = cv2.fastNlMeansDenoising(gray, h=10)

        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = _remove_lines(thr)

        # deskew robusto
        angle = robust_skew_estimate(thr)
        if abs(angle) > 0.2:
            (h, w) = thr.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            thr = cv2.warpAffine(thr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            color = cv2.warpAffine(color, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # cierre leve
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((1,1), np.uint8))
        return thr, color
    except Exception as e:
        logger.error(f"Enhance error: {e}")
        return None, None

# ----------------------------
# Post-procesado de texto
# ----------------------------
def correct_ocr_text(text: str) -> str:
    t = text
    for wrong, correct in CHAR_CORRECTIONS.items():
        t = re.sub(r'\b' + re.escape(wrong) + r'\b', correct, t, flags=re.IGNORECASE)
    for pattern, replacement in LEGAL_PATTERNS:
        t = re.sub(pattern, replacement, t, flags=re.IGNORECASE)
    # Limpiezas
    t = re.sub(r'[ \t]{2,}', ' ', t)
    t = re.sub(r'\s+\n', '\n', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    t = re.sub(r'(\w)-\n(\w)', r'\1\2\n', t)
    return t.strip()

# ----------------------------
# OCR multipase + consenso
# ----------------------------
def _avg_conf(data_dict) -> float:
    vals = []
    for c in data_dict.get('conf', []):
        try:
            v = float(c)
            if v > 0:
                vals.append(v)
        except Exception:
            pass
    return float(np.mean(vals)) if vals else 0.0

def ocr_pass(image_path: str, language: str, cfg: str) -> Tuple[str, float, List[str], List[float]]:
    d = pytesseract.image_to_data(image_path, lang=language, config=cfg, output_type=pytesseract.Output.DICT)
    avg = _avg_conf(d)
    words, confs = [], []
    for w, c in zip(d['text'], d['conf']):
        w = (w or '').strip()
        try:
            c = float(c)
        except Exception:
            c = -1.0
        if w:
            words.append(w)
            confs.append(c)
    text = pytesseract.image_to_string(image_path, lang=language, config=cfg)
    return correct_ocr_text(text), avg, words, confs

def consensus_by_conf(passes: List[Tuple[str, float, List[str], List[float]]]) -> str:
    """
    Ensamble de palabras por voto de confianza:
    - se toma la secuencia de palabras del mejor pass (mayor avg)
    - para cada posición, si otras pasadas proponen palabra similar (levenshtein simple) y mayor confianza, se sustituye.
    """
    if not passes:
        return ""
    passes_sorted = sorted(passes, key=lambda x: x[1], reverse=True)
    base_text, _, base_words, base_confs = passes_sorted[0]

    def lev(a: str, b: str) -> int:
        # distancia Levenshtein O(len(a)*len(b)) sin deps externas
        la, lb = len(a), len(b)
        dp = list(range(lb+1))
        for i in range(1, la+1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, lb+1):
                cur = dp[j]
                cost = 0 if a[i-1] == b[j-1] else 1
                dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + cost)
                prev = cur
        return dp[lb]

    words = base_words[:]
    confs = base_confs[:]
    for _, _, wlist, clist in passes_sorted[1:]:
        L = min(len(words), len(wlist))
        for i in range(L):
            if wlist[i] and (clist[i] > confs[i]):
                # solo aceptar si parecido para no desordenar
                if min(len(words[i]), len(wlist[i])) == 0:
                    continue
                if lev(words[i].lower(), wlist[i].lower()) <= max(1, min(len(words[i]), len(wlist[i])) // 3):
                    words[i] = wlist[i]
                    confs[i] = clist[i]
    return correct_ocr_text(" ".join(words)) if words else base_text

def perform_advanced_ocr(image_path: str, language: str = 'spa') -> str:
    base_cfg = f"-c preserve_interword_spaces=1 -c user_defined_dpi={max(300, OCR_DPI)}"
    cfgs = [
        f"--psm 6  --oem 1 {base_cfg}",
        f"--psm 4  --oem 1 {base_cfg}",
        f"--psm 11 --oem 1 {base_cfg}",
        f"--psm 6  --oem 1 {base_cfg} -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNÑOPQRSTUVWXYZabcdefghijklmnñopqrstuvwxyzÁÉÍÓÚáéíóú°:-./,()",
    ]
    passes: List[Tuple[str, float, List[str], List[float]]] = []
    for cfg in cfgs:
        try:
            passes.append(ocr_pass(image_path, language, cfg))
        except Exception as e:
            logger.debug(f"OCR pass fail: {e}")

    # también añadir versión invertida por si el binarizado quedó invertido
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            inv = 255 - img
            tmp = image_path + ".inv.png"
            cv2.imwrite(tmp, inv)
            passes.append(ocr_pass(tmp, language, cfgs[0]))
            os.remove(tmp)
    except Exception:
        pass

    if not passes:
        text = pytesseract.image_to_string(image_path, lang=language)
        return correct_ocr_text(text)

    # consenso
    return consensus_by_conf(passes)

# ----------------------------
# Tablas (Hough + clustering)
# ----------------------------
def _cluster_1d(vals: List[int], gap: int) -> List[int]:
    if not vals:
        return []
    vals = sorted(vals)
    clusters = [[vals[0]]]
    for v in vals[1:]:
        if v - clusters[-1][-1] <= gap:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    # tomar el promedio de cada cluster
    return [int(np.mean(c)) for c in clusters]

def _detect_table_lines(binary: np.ndarray) -> Tuple[List[int], List[int]]:
    edges = cv2.Canny(255 - binary, 50, 150, apertureSize=3)
    # Hough probabilístico para líneas largas
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=max(60, binary.shape[1]//10),
                            minLineLength=binary.shape[1]//6, maxLineGap=10)
    ys, xs = [], []
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0]:
            if abs(y1 - y2) < 3:  # horizontal
                ys.extend([y1, y2])
            if abs(x1 - x2) < 3:  # vertical
                xs.extend([x1, x2])
    y_lines = _cluster_1d(ys, gap=8)
    x_lines = _cluster_1d(xs, gap=8)
    return y_lines, x_lines

def _ocr_cell(img: np.ndarray, lang: str) -> str:
    try:
        cfg = f"--psm 7 --oem 1 -c user_defined_dpi={max(300, OCR_DPI)}"
        text = pytesseract.image_to_string(img, lang=lang, config=cfg)
        return correct_ocr_text(text).replace('\n', ' ').strip()
    except Exception:
        return ""

def extract_tables(binary: np.ndarray, color: np.ndarray, lang: str) -> List[Dict[str, Any]]:
    tables: List[Dict[str, Any]] = []
    try:
        y_lines, x_lines = _detect_table_lines(binary)
        if len(y_lines) >= 3 and len(x_lines) >= 2:
            y_pairs = list(zip(y_lines, y_lines[1:]))
            x_pairs = list(zip(x_lines, x_lines[1:]))

            pad = 3
            rows_text: List[List[str]] = []
            for (y1, y2) in y_pairs:
                row = []
                for (x1, x2) in x_pairs:
                    if y2 - y1 < 12 or x2 - x1 < 12:
                        row.append("")
                        continue
                    cell = color[max(y1+pad,0):max(y2-pad,0), max(x1+pad,0):max(x2-pad,0)]
                    if cell.size == 0:
                        row.append("")
                        continue
                    cell_gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
                    _, cell_bin = cv2.threshold(cell_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    cell_for_ocr = 255 - cell_bin if np.mean(cell_bin) < 127 else cell_bin
                    text = _ocr_cell(cell_for_ocr, lang)
                    row.append(text)
                rows_text.append(row)

            headers = rows_text[0] if rows_text and any(t.strip() for t in rows_text[0]) else []
            body = rows_text[1:] if headers else rows_text

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
    except Exception as e:
        logger.debug(f"table extraction failed: {e}")
    return tables

# ----------------------------
# Extracción de campos clave
# ----------------------------
KEY_PATTERNS: Dict[str, List[re.Pattern]] = {
    'expediente': [re.compile(r'\bEXPEDIENTE[:\s]+([A-Z0-9\-]+)', re.I)],
    'juzgado': [re.compile(r'\bJUZGADO[:\s]+(.+)', re.I)],
    'juez': [re.compile(r'\bJUE[ZS][:\s]+(.+)', re.I)],
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
                val = re.sub(r'\s{2,}', ' ', val)
                val = re.sub(r'[\|#]+$', '', val).strip()
                kv[key] = val
                break
    return kv

# ----------------------------
# Procesado de una página (worker)
# ----------------------------
@dataclass
class PageResult:
    index: int
    text: str
    tables: List[Dict[str, Any]]

def ocr_png_worker(png_path: str, index: int, language: str, want_tables: bool) -> PageResult:
    try:
        _fix_orientation_inplace(png_path)
        binary, color = enhance_image_for_ocr(png_path)
        if binary is None:
            text = pytesseract.image_to_string(png_path, lang=language)
            return PageResult(index=index, text=correct_ocr_text(text), tables=[])
        # Guardar binaria para Tesseract
        tmp = png_path + ".proc.png"
        cv2.imwrite(tmp, binary)
        text = perform_advanced_ocr(tmp, language=language)
        tables: List[Dict[str, Any]] = []
        if want_tables and color is not None:
            tables = extract_tables(binary, color, language)
        try:
            os.remove(tmp)
        except Exception:
            pass
        return PageResult(index=index, text=text, tables=tables)
    except Exception as e:
        logger.error(f"worker page {index} error: {e}")
        return PageResult(index=index, text=f"(Error OCR en página {index}: {e})", tables=[])

# ----------------------------
# Procesamiento de PDFs
# ----------------------------
def process_text_pdf(pdf_path: str, output_path: str) -> Tuple[bool, int, str, Dict[str, Any]]:
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
        logger.error(f"text PDF error: {e}")
        return False, 0, str(e), {}

def process_bitmap_pdf(pdf_path: str, output_path: str, language: str = 'spa', want_tables: bool = False
                      ) -> Tuple[bool, int, str, Dict[str, Any]]:
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                info = pdfinfo_from_path(pdf_path)
                total_pages = int(info.get("Pages", 0))
            except Exception:
                reader = PdfReader(pdf_path)
                total_pages = len(reader.pages)

            # Convertir páginas una por una y despachar OCR en paralelo
            futures = []
            results_map: Dict[int, PageResult] = {}
            with ProcessPoolExecutor(max_workers=OCR_WORKERS) as ex:
                for i in range(1, total_pages + 1):
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
                        results_map[i] = PageResult(i, "(Página vacía o no legible)", [])
                        continue
                    png_path = png_paths[0]
                    futures.append(ex.submit(ocr_png_worker, png_path, i, language, want_tables))

                for fut in as_completed(futures):
                    pr: PageResult = fut.result()
                    results_map[pr.index] = pr

            # Unir texto en orden
            all_text = ""
            all_tables: List[Dict[str, Any]] = []
            for i in range(1, total_pages + 1):
                pr = results_map.get(i, PageResult(i, "(sin resultado)", []))
                all_text += get_page_separator(i, total_pages)
                all_text += pr.text
                if pr.tables:
                    all_tables.extend(pr.tables)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(all_text)

            meta = {'key_values': extract_key_values(all_text), 'tables': all_tables}
            return True, total_pages, all_text, meta
    except Exception as e:
        logger.error(f"bitmap PDF error: {e}")
        return False, 0, str(e), {}

# ----------------------------
# Entrypoints
# ----------------------------
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '3.0.0-adv',
        'dpi': OCR_DPI,
        'lang_default': OCR_LANG_DEFAULT,
        'workers': OCR_WORKERS
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
            logger.warning(f"Load custom corrections failed: {e}")

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

        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        file.save(file_path)

        output_filename = f"{timestamp}_{os.path.splitext(filename)[0]}.txt"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        logger.info(f"Processing {temp_filename} | lang={language} | dpi={OCR_DPI} | tables={want_tables} | workers={OCR_WORKERS}")

        load_custom_corrections()
        start_time = datetime.now()

        is_bitmap = contains_bitmap(file_path) or force_bitmap
        if is_bitmap:
            logger.info("Mode: enhanced OCR (bitmap/scan + advanced math)")
            success, pages_processed, text, meta = process_bitmap_pdf(file_path, output_path, language, want_tables)
            method = 'enhanced_ocr_advanced'
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
            md_stream = io.BytesIO(md_bytes); md_stream.seek(0)
            return send_file(md_stream, as_attachment=True,
                             download_name=f"{os.path.splitext(filename)[0]}.md",
                             mimetype='text/markdown')
        else:
            return send_file(output_path, as_attachment=True,
                             download_name=f"{os.path.splitext(filename)[0]}.txt")

    except Exception as e:
        logger.error(f"process error: {str(e)}")
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
                        method = 'enhanced_ocr_advanced'
                    else:
                        success, pages, text, meta = process_text_pdf(file_path, output_path)
                        method = 'text_extraction_corrected'

                    with open(os.path.join(batch_folder, f"{os.path.splitext(filename)[0]}.json"), "w", encoding="utf-8") as jf:
                        json.dump({
                            'filename': filename,
                            'pages': pages,
                            'method': method,
                            'key_values': meta.get('key_values', {}),
                            'tables': meta.get('tables', []),
                        }, jf, ensure_ascii=False, indent=2)

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

        for tmp in temp_files:
            try:
                os.remove(tmp)
            except Exception:
                pass

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

def _bootstrap():
    try:
        load_custom_corrections()
    except Exception:
        pass

_bootstrap()

if __name__ == '__main__':
    # En producción: aumenta timeout para PDFs grandes
    # gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 900 app:app
    app.run(host='0.0.0.0', port=5000, debug=False)
