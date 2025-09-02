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
from typing import Tuple, List, Dict, Any, Optional, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

import pytesseract
from PyPDF2 import PdfReader
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image, ImageEnhance, ImageFilter

# ----------------------------
# Logging Configuration
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ocr.peruvian_legal")

# ----------------------------
# App & CORS
# ----------------------------
app = Flask(__name__)
CORS(app, origins=['*'], methods=['GET', 'POST', 'OPTIONS'])

# ----------------------------
# Enhanced Configuration
# ----------------------------
class Config:
    # Directories
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', '/app/input')
    OUTPUT_FOLDER = os.getenv('OUTPUT_FOLDER', '/app/output')
    TEMP_FOLDER = os.getenv('TEMP_FOLDER', '/app/temp')
    TRAINING_FOLDER = os.getenv('TRAINING_FOLDER', '/app/training')
    
    # File handling
    ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'}
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_FILE_SIZE_MB', '100')) * 1024 * 1024
    
    # OCR settings - optimized for Spanish legal documents
    TESSERACT_CMD = os.getenv('TESSERACT_CMD', '/usr/bin/tesseract')
    OCR_LANG_DEFAULT = os.getenv('OCR_LANG', 'spa')  # Spanish only for better accuracy
    OCR_DPI = int(os.getenv('OCR_DPI', '400'))  # Slightly lower for speed
    OCR_SCALE = float(os.getenv('OCR_SCALE', '2.5'))  # Optimized scaling
    OSD_ENABLED = os.getenv('OSD_ENABLED', '1') == '1'
    OCR_WORKERS = int(os.getenv('OCR_WORKERS', min(4, os.cpu_count() or 1)))  # Cap workers
    
    # Performance settings
    ENABLE_GPU = os.getenv('ENABLE_GPU', '0') == '1'
    MEMORY_LIMIT_MB = int(os.getenv('MEMORY_LIMIT_MB', '2048'))
    
    # Quality thresholds
    MIN_CONFIDENCE_THRESHOLD = float(os.getenv('MIN_CONFIDENCE', '60.0'))
    TEXT_EXTRACTION_THRESHOLD = int(os.getenv('TEXT_THRESHOLD', '800'))  # Increased for better detection

app.config.from_object(Config)

# Create directories
for folder in [Config.UPLOAD_FOLDER, Config.OUTPUT_FOLDER, Config.TEMP_FOLDER, Config.TRAINING_FOLDER]:
    Path(folder).mkdir(parents=True, exist_ok=True)

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_CMD

# ----------------------------
# Enhanced Legal Patterns for Peru
# ----------------------------
PERUVIAN_LEGAL_CORRECTIONS: Dict[str, str] = {
    # Common OCR errors in Spanish legal documents
    'FISCALÍA': 'FISCALÍA', 'FISCALIA': 'FISCALÍA', 'FlSCALÍA': 'FISCALÍA',
    'MINISTERIO': 'MINISTERIO', 'MlNlSTERlO': 'MINISTERIO',
    'PÚBLICO': 'PÚBLICO', 'PUBLICO': 'PÚBLICO', 'PÚBLlCO': 'PÚBLICO',
    'INVESTIGACIÓN': 'INVESTIGACIÓN', 'INVESTIGACION': 'INVESTIGACIÓN',
    'PROVIDENCIA': 'PROVIDENCIA', 'PROVlDENCIA': 'PROVIDENCIA',
    'CARPETA': 'CARPETA', 'CARFETA': 'CARPETA',
    'EXPEDIENTE': 'EXPEDIENTE', 'EXPEDlENTE': 'EXPEDIENTE',
    'RESOLUCIÓN': 'RESOLUCIÓN', 'RESOLUCION': 'RESOLUCIÓN',
    'DISTRITO': 'DISTRITO', 'DlSTRlTO': 'DISTRITO',
    'AREQUIPA': 'AREQUIPA', 'AREQUlPA': 'AREQUIPA',
    'CHIVAY': 'CHIVAY', 'CHlVAY': 'CHIVAY',
    'PENAL': 'PENAL', 'PENAЛ': 'PENAL',
    'CIVIL': 'CIVIL', 'ClVlL': 'CIVIL',
    'CORPORATIVA': 'CORPORATIVA', 'CORPORATlVA': 'CORPORATIVA',
    'DELITO': 'DELITO', 'DELlTO': 'DELITO',
    'CONTRA': 'CONTRA', 'CÖNTRA': 'CONTRA',
    'PATRIMONIO': 'PATRIMONIO', 'PATRlMONlO': 'PATRIMONIO',
    'USURPACIÓN': 'USURPACIÓN', 'USURPACION': 'USURPACIÓN',
    'APROPIACIÓN': 'APROPIACIÓN', 'APROPIACION': 'APROPIACIÓN',
    'CÓDIGO': 'CÓDIGO', 'CODIGO': 'CÓDIGO',
    'ARTÍCULO': 'ARTÍCULO', 'ARTICULO': 'ARTÍCULO',
    'TÍTULO': 'TÍTULO', 'TITULO': 'TÍTULO',
    'PROCEDIMIENTO': 'PROCEDIMIENTO', 'PROCEDlMlENTO': 'PROCEDIMIENTO',
}

PERUVIAN_LEGAL_PATTERNS: List[Tuple[str, str]] = [
    # Carpeta Fiscal patterns
    (r'CARPETA\s+FISCAL[:\s]*(\d{10}-\d{4}-\d{1,3}-\d{1,2})', r'CARPETA FISCAL: \1'),
    (r'CARPETA\s+FlSCAL[:\s]*(\d{10}-\d{4}-\d{1,3}-\d{1,2})', r'CARPETA FISCAL: \1'),
    
    # Expediente patterns
    (r'EXP(?:EDIENTE)?\.?\s*N[°º]?\s*(\d{4,5}-\d{4}-\d{1,3}-\w{2,4}-\w{2})', r'EXPEDIENTE N° \1'),
    (r'EXP(?:EDlENTE)?\.?\s*N[°º]?\s*(\d{4,5}-\d{4}-\d{1,3}-\w{2,4}-\w{2})', r'EXPEDIENTE N° \1'),
    
    # Resolution patterns
    (r'RESOLUCIÓN\s*N[°º]?\s*(\d+[-\s]?\d*)', r'RESOLUCIÓN N° \1'),
    (r'RESOLUCION\s*N[°º]?\s*(\d+[-\s]?\d*)', r'RESOLUCIÓN N° \1'),
    
    # Fiscal office patterns
    (r'FISCALÍA\s+PROVINCIAL\s+(PENAL|CIVIL|FAMILIA)', r'FISCALÍA PROVINCIAL \1'),
    (r'FlSCALÍA\s+PROVlNClAL\s+(PENAL|ClVlL|FAMlLlA)', r'FISCALÍA PROVINCIAL \1'),
    
    # Date patterns (dd/mm/yyyy, dd-mm-yyyy, dd de mmm del yyyy)
    (r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', r'\1/\2/\3'),
    (r'(\d{1,2})\s+de\s+(\w+)\s+del?\s+(\d{4})', r'\1 de \2 del \3'),
    
    # License plate patterns
    (r'\b([A-Z]{3})[-\s]?(\d{3})\b', r'\1-\2'),
    (r'\b([A-Z]{2})[-\s]?(\d{4})\b', r'\1-\2'),
    
    # Legal article references
    (r'Art\.?\s*(\d+)[°º]?', r'Artículo \1°'),
    (r'Inc\.?\s*(\d+)', r'Inciso \1'),
    
    # Clean up multiple spaces and line breaks
    (r'\s{2,}', ' '),
    (r'\n{3,}', '\n\n'),
]

# Field extraction patterns for Peruvian legal documents
PERUVIAN_KEY_PATTERNS: Dict[str, List[re.Pattern]] = {
    'carpeta_fiscal': [
        re.compile(r'CARPETA\s+FISCAL[:\s]*(\d{10}-\d{4}-\d{1,3}-\d{1,2})', re.I),
        re.compile(r'CARPETA\s+FlSCAL[:\s]*(\d{10}-\d{4}-\d{1,3}-\d{1,2})', re.I),
    ],
    'expediente': [
        re.compile(r'EXP(?:EDIENTE)?\.?\s*N[°º]?\s*[:\s]*(\d{4,5}-\d{4}-\d{1,3}-\w{2,4}-\w{2})', re.I),
        re.compile(r'EXP(?:EDlENTE)?\.?\s*N[°º]?\s*[:\s]*(\d{4,5}-\d{4}-\d{1,3}-\w{2,4}-\w{2})', re.I),
    ],
    'fiscalia': [
        re.compile(r'FISCALÍA\s+(PROVINCIAL|SUPERIOR|SUPREMA)\s+(.+?)(?:DISTRITO|$)', re.I),
        re.compile(r'FlSCALÍA\s+(PROVlNClAL|SUPERlOR|SUPREMA)\s+(.+?)(?:DlSTRlTO|$)', re.I),
    ],
    'distrito_fiscal': [
        re.compile(r'DISTRITO\s+FISCAL\s+DE\s+([A-ZÁÉÍÓÚÑ\s]+)', re.I),
        re.compile(r'DlSTRlTO\s+FlSCAL\s+DE\s+([A-ZÁÉÍÓÚÑ\s]+)', re.I),
    ],
    'delito': [
        re.compile(r'DELITO[:\s]+(.+?)(?:\n|$)', re.I),
        re.compile(r'DELlTO[:\s]+(.+?)(?:\n|$)', re.I),
        re.compile(r'CONTRA\s+EL\s+(PATRIMONIO|HONOR|VIDA)', re.I),
        re.compile(r'USURPACIÓN', re.I),
        re.compile(r'APROPIACIÓN\s+ILÍCITA', re.I),
    ],
    'imputado': [
        re.compile(r'IMPUTADO[:\s]+([A-ZÁÉÍÓÚÑ\s,]+?)(?:\n|DELITO|AGRAVIADO)', re.I),
        re.compile(r'lMPUTADO[:\s]+([A-ZÁÉÍÓÚÑ\s,]+?)(?:\n|DELlTO|AGRAVlADO)', re.I),
        re.compile(r'INVESTIGADO[:\s]+([A-ZÁÉÍÓÚÑ\s,]+?)(?:\n|DELITO|AGRAVIADO)', re.I),
    ],
    'agraviado': [
        re.compile(r'AGRAVIADO[:\s]+([A-ZÁÉÍÓÚÑ\s,]+?)(?:\n|DELITO|IMPUTADO)', re.I),
        re.compile(r'AGRAVlADO[:\s]+([A-ZÁÉÍÓÚÑ\s,]+?)(?:\n|DELlTO|lMPUTADO)', re.I),
    ],
    'fecha': [
        re.compile(r'(\d{1,2})\s+de\s+(\w+)\s+del?\s+(\d{4})', re.I),
        re.compile(r'FECHA[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{4})', re.I),
        re.compile(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', re.I),
    ],
    'resolucion': [
        re.compile(r'RESOLUCIÓN\s*N[°º]?\s*(\d+[-\s]?\d*)', re.I),
        re.compile(r'RESOLUCION\s*N[°º]?\s*(\d+[-\s]?\d*)', re.I),
    ],
    'juzgado': [
        re.compile(r'JUZGADO\s+([A-ZÁÉÍÓÚÑ\s]+?)(?:DE|$)', re.I),
        re.compile(r'JUZGADO\s+PENAL\s+([A-ZÁÉÍÓÚÑ\s\d]+)', re.I),
    ],
    'juez': [
        re.compile(r'JUE[ZS][:\s]+([A-ZÁÉÍÓÚÑ\s]+?)(?:\n|$)', re.I),
        re.compile(r'MAGISTRADO[:\s]+([A-ZÁÉÍÓÚÑ\s]+?)(?:\n|$)', re.I),
    ],
    'fiscal': [
        re.compile(r'FISCAL[:\s]+([A-ZÁÉÍÓÚÑ\s]+?)(?:\n|$)', re.I),
        re.compile(r'FlSCAL[:\s]+([A-ZÁÉÍÓÚÑ\s]+?)(?:\n|$)', re.I),
    ],
}

# ----------------------------
# Utility Functions
# ----------------------------
def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB."""
    return os.path.getsize(file_path) / (1024 * 1024)

def cleanup_temp_files(*file_paths: str) -> None:
    """Clean up temporary files safely."""
    for file_path in file_paths:
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.warning(f"Could not remove temp file {file_path}: {e}")

def memory_cleanup():
    """Force garbage collection to free memory."""
    gc.collect()

# ----------------------------
# Enhanced PDF Analysis
# ----------------------------
def analyze_pdf_content(pdf_path: str) -> Dict[str, Any]:
    """
    Comprehensive PDF analysis to determine processing strategy.
    Returns analysis results including text content, image presence, and processing recommendations.
    """
    analysis = {
        'has_text': False,
        'has_images': False,
        'text_length': 0,
        'page_count': 0,
        'text_density': 0.0,
        'processing_recommendation': 'ocr',  # Default to OCR for safety
        'confidence': 0.0,
        'sample_text': '',
        'file_size_mb': 0.0
    }
    
    try:
        analysis['file_size_mb'] = get_file_size_mb(pdf_path)
        
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            analysis['page_count'] = len(reader.pages)
            
            total_text = ""
            image_count = 0
            
            # Analyze first few pages for performance
            pages_to_check = min(3, analysis['page_count'])
            
            for i, page in enumerate(reader.pages[:pages_to_check]):
                try:
                    # Extract text
                    page_text = page.extract_text() or ""
                    clean_text = re.sub(r'\s+', ' ', page_text).strip()
                    total_text += clean_text + " "
                    
                    # Check for images/objects
                    page_obj = page.get_object()
                    if page_obj and '/XObject' in str(page_obj):
                        resources = page_obj.get('/Resources')
                        if resources and '/XObject' in resources:
                            xobjects = resources['/XObject'].get_object()
                            for obj_name in xobjects:
                                obj = xobjects[obj_name]
                                if obj.get('/Subtype') == '/Image':
                                    image_count += 1
                                    
                except Exception as e:
                    logger.debug(f"Error analyzing page {i}: {e}")
                    continue
            
            analysis['text_length'] = len(total_text.strip())
            analysis['has_text'] = analysis['text_length'] > 50
            analysis['has_images'] = image_count > 0
            analysis['text_density'] = analysis['text_length'] / max(1, analysis['page_count'])
            analysis['sample_text'] = total_text[:500].strip()
            
            # Determine processing strategy
            if analysis['text_length'] >= Config.TEXT_EXTRACTION_THRESHOLD and not analysis['has_images']:
                analysis['processing_recommendation'] = 'text_extraction'
                analysis['confidence'] = 0.9
            elif analysis['text_length'] >= 200 and analysis['text_density'] > 100:
                analysis['processing_recommendation'] = 'hybrid'
                analysis['confidence'] = 0.7
            else:
                analysis['processing_recommendation'] = 'ocr'
                analysis['confidence'] = 0.8
                
    except Exception as e:
        logger.error(f"PDF analysis failed: {e}")
        analysis['processing_recommendation'] = 'ocr'  # Safe fallback
        
    return analysis

# ----------------------------
# Advanced Image Processing
# ----------------------------
def gaussian_sharpen(image: np.ndarray, sigma: float = 1.0, alpha: float = 1.5) -> np.ndarray:
    """Apply unsharp mask for better text clarity."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + alpha, blurred, -alpha, 0)
    return sharpened

def adaptive_threshold_enhanced(gray: np.ndarray) -> np.ndarray:
    """Enhanced adaptive thresholding for varied lighting conditions."""
    # Try multiple thresholding methods and combine
    methods = [
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2),
    ]
    
    # Also try Otsu
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    methods.append(otsu)
    
    # Combine methods by taking the brightest pixels (assumes text is dark on light background)
    combined = np.maximum.reduce(methods)
    return combined

def remove_noise_morphology(binary: np.ndarray) -> np.ndarray:
    """Remove noise using morphological operations."""
    # Remove small noise
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Fill small holes in text
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def enhance_image_for_ocr_optimized(image_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Optimized image enhancement pipeline for Peruvian legal documents.
    Returns: (binary_image, color_image) or (None, None) if failed.
    """
    try:
        # Read image
        color = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if color is None:
            return None, None
        
        original_shape = color.shape[:2]
        
        # Smart scaling based on image size
        if Config.OCR_SCALE != 1.0:
            # Only scale up small images, scale down very large ones
            height, width = original_shape
            if height * width < 1000000:  # Less than 1MP
                scale = Config.OCR_SCALE
            elif height * width > 8000000:  # More than 8MP
                scale = 0.7  # Scale down for performance
            else:
                scale = 1.2  # Modest scaling for medium images
                
            if scale != 1.0:
                color = cv2.resize(color, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Sharpen the image
        gray = gaussian_sharpen(gray, sigma=1.0, alpha=0.3)
        
        # Denoise
        gray = cv2.fastNlMeansDenoising(gray, h=8)
        
        # Enhanced thresholding
        binary = adaptive_threshold_enhanced(gray)
        
        # Remove noise
        binary = remove_noise_morphology(binary)
        
        # Simple skew correction (only for significant skew)
        angle = estimate_skew_simple(binary)
        if abs(angle) > 1.0:  # Only correct significant skew
            (h, w) = binary.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            binary = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            color = cv2.warpAffine(color, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return binary, color
        
    except Exception as e:
        logger.error(f"Image enhancement failed: {e}")
        return None, None

def estimate_skew_simple(binary: np.ndarray) -> float:
    """Simple and fast skew estimation using Hough transform."""
    try:
        # Get edges
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)
        
        # Get lines
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=max(100, binary.shape[1]//10))
        
        if lines is None:
            return 0.0
        
        angles = []
        for rho_theta in lines[:50]:  # Limit to first 50 lines for speed
            rho, theta = rho_theta[0]
            angle = (theta * 180 / np.pi) - 90
            if -30 <= angle <= 30:  # Only consider reasonable angles
                angles.append(angle)
        
        if not angles:
            return 0.0
            
        # Return median angle
        return float(np.median(angles))
        
    except Exception:
        return 0.0

# ----------------------------
# OCR Processing
# ----------------------------
def correct_peruvian_legal_text(text: str) -> str:
    """Apply corrections specific to Peruvian legal documents."""
    corrected = text
    
    # Apply character-level corrections
    for wrong, correct in PERUVIAN_LEGAL_CORRECTIONS.items():
        corrected = re.sub(r'\b' + re.escape(wrong) + r'\b', correct, corrected, flags=re.IGNORECASE)
    
    # Apply pattern-based corrections
    for pattern, replacement in PERUVIAN_LEGAL_PATTERNS:
        corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE | re.MULTILINE)
    
    # Additional cleanup
    corrected = re.sub(r'[\r\n]+', '\n', corrected)  # Normalize line breaks
    corrected = re.sub(r'[ \t]+', ' ', corrected)    # Normalize spaces
    corrected = re.sub(r'\n\s*\n\s*\n+', '\n\n', corrected)  # Limit consecutive line breaks
    
    return corrected.strip()

def perform_ocr_optimized(image_path: str, language: str = 'spa') -> Tuple[str, float]:
    """
    Optimized OCR with multiple passes and confidence scoring.
    Returns: (text, average_confidence)
    """
    try:
        base_config = f"--dpi {Config.OCR_DPI} -c preserve_interword_spaces=1"
        
        # Configuration optimized for Spanish legal documents
        configs = [
            f"--psm 6 --oem 3 {base_config}",    # Uniform block of text
            f"--psm 4 --oem 3 {base_config}",    # Single column of text
            f"--psm 6 --oem 1 {base_config}",    # LSTM only
        ]
        
        best_text = ""
        best_confidence = 0.0
        
        for config in configs:
            try:
                # Get detailed data for confidence calculation
                data = pytesseract.image_to_data(image_path, lang=language, config=config, output_type=pytesseract.Output.DICT)
                
                # Calculate average confidence
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_confidence = np.mean(confidences) if confidences else 0.0
                
                if avg_confidence > best_confidence:
                    best_confidence = avg_confidence
                    best_text = pytesseract.image_to_string(image_path, lang=language, config=config)
                    
            except Exception as e:
                logger.debug(f"OCR pass failed with config {config}: {e}")
                continue
        
        # Fallback if all configs failed
        if not best_text:
            best_text = pytesseract.image_to_string(image_path, lang=language)
            best_confidence = 50.0  # Default confidence
        
        # Apply corrections
        corrected_text = correct_peruvian_legal_text(best_text)
        
        return corrected_text, float(best_confidence)
        
    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        return f"Error en OCR: {str(e)}", 0.0

# ----------------------------
# Field Extraction
# ----------------------------
def extract_peruvian_legal_fields(text: str) -> Dict[str, Any]:
    """Extract specific fields from Peruvian legal documents."""
    fields = {}
    
    for field_name, patterns in PERUVIAN_KEY_PATTERNS.items():
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                value = match.group(1).strip() if match.lastindex and match.lastindex >= 1 else match.group(0).strip()
                
                # Clean the extracted value
                value = re.sub(r'\s+', ' ', value)
                value = re.sub(r'[|#]+$', '', value).strip()
                
                if value:
                    fields[field_name] = value
                    break  # Use first match
    
    # Post-process specific fields
    if 'fecha' in fields:
        # Normalize date format
        date_text = fields['fecha']
        # Try to extract day, month, year
        date_match = re.search(r'(\d{1,2})\s+de\s+(\w+)\s+del?\s+(\d{4})', date_text, re.I)
        if date_match:
            day, month_name, year = date_match.groups()
            fields['fecha_normalizada'] = f"{day} de {month_name.lower()} del {year}"
    
    return fields

# ----------------------------
# Document Processing Functions
# ----------------------------
def get_page_separator(page_num: int, total_pages: int) -> str:
    """Generate page separator for multi-page documents."""
    return f"\n{'='*50}\nPÁGINA {page_num} DE {total_pages}\n{'='*50}\n\n"

@dataclass
class PageProcessingResult:
    index: int
    text: str
    confidence: float
    processing_time: float
    error: Optional[str] = None

def process_pdf_page_worker(args: Tuple[str, int, int, str]) -> PageProcessingResult:
    """
    Worker function for processing a single PDF page.
    Args: (png_path, page_index, total_pages, language)
    """
    png_path, page_index, total_pages, language = args
    start_time = datetime.now()
    
    try:
        # Optional orientation correction
        if Config.OSD_ENABLED:
            try:
                osd = pytesseract.image_to_osd(png_path)
                rotate_match = re.search(r"Rotate:\s+(\d+)", osd)
                if rotate_match:
                    rotation = int(rotate_match.group(1))
                    if rotation % 360 != 0:
                        img = Image.open(png_path)
                        img = img.rotate(360 - rotation, expand=True, fillcolor="white")
                        img.save(png_path)
            except Exception:
                pass  # Skip if OSD fails
        
        # Enhanced image processing
        binary, color = enhance_image_for_ocr_optimized(png_path)
        
        if binary is None:
            # Fallback to basic OCR
            text, confidence = perform_ocr_optimized(png_path, language)
        else:
            # Save processed image for OCR
            processed_path = png_path + ".processed.png"
            cv2.imwrite(processed_path, binary)
            
            try:
                text, confidence = perform_ocr_optimized(processed_path, language)
            finally:
                cleanup_temp_files(processed_path)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return PageProcessingResult(
            index=page_index,
            text=text,
            confidence=confidence,
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        error_msg = f"Error procesando página {page_index}: {str(e)}"
        logger.error(error_msg)
        
        return PageProcessingResult(
            index=page_index,
            text="",
            confidence=0.0,
            processing_time=processing_time,
            error=error_msg
        )

def process_text_pdf_optimized(pdf_path: str, output_path: str) -> Tuple[bool, int, str, Dict[str, Any]]:
    """
    Optimized text extraction from PDF with corrections.
    """
    try:
        logger.info("Processing as text-based PDF with enhanced corrections")
        
        all_text = ""
        total_pages = 0
        
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            total_pages = len(reader.pages)
            
            for i, page in enumerate(reader.pages, 1):
                try:
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        all_text += get_page_separator(i, total_pages)
                        corrected_text = correct_peruvian_legal_text(page_text)
                        all_text += corrected_text
                except Exception as e:
                    logger.warning(f"Error extracting text from page {i}: {e}")
                    all_text += get_page_separator(i, total_pages)
                    all_text += f"[Error extrayendo texto de la página {i}]"
        
        # Extract legal fields
        legal_fields = extract_peruvian_legal_fields(all_text)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(all_text)
        
        metadata = {
            'key_values': legal_fields,
            'processing_method': 'text_extraction_enhanced',
            'confidence': 95.0,  # High confidence for text extraction
            'total_pages': total_pages
        }
        
        return True, total_pages, all_text, metadata
        
    except Exception as e:
        logger.error(f"Text PDF processing failed: {e}")
        return False, 0, str(e), {}

def process_scanned_pdf_optimized(pdf_path: str, output_path: str, language: str = 'spa') -> Tuple[bool, int, str, Dict[str, Any]]:
    """
    Optimized OCR processing for scanned PDFs with parallel processing.
    """
    try:
        logger.info("Processing as scanned PDF with advanced OCR")
        
        # Get PDF info
        try:
            info = pdfinfo_from_path(pdf_path)
            total_pages = int(info.get("Pages", 0))
        except Exception:
            with open(pdf_path, 'rb') as f:
                reader = PdfReader(f)
                total_pages = len(reader.pages)
        
        if total_pages == 0:
            return False, 0, "PDF no contiene páginas", {}
        
        logger.info(f"Processing {total_pages} pages with {Config.OCR_WORKERS} workers")
        
        all_text = ""
        page_results = {}
        total_confidence = 0.0
        processed_pages = 0
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Prepare worker arguments
            worker_args = []
            temp_files = []
            
            for page_num in range(1, total_pages + 1):
                try:
                    # Convert single page to PNG
                    png_paths = convert_from_path(
                        pdf_path,
                        dpi=Config.OCR_DPI,
                        first_page=page_num,
                        last_page=page_num,
                        fmt="png",
                        output_folder=temp_dir,
                        paths_only=True,
                        thread_count=1,
                    )
                    
                    if png_paths:
                        png_path = png_paths[0]
                        temp_files.append(png_path)
                        worker_args.append((png_path, page_num, total_pages, language))
                    else:
                        # Handle missing page
                        page_results[page_num] = PageProcessingResult(
                            index=page_num,
                            text="[Página no disponible]",
                            confidence=0.0,
                            processing_time=0.0
                        )
                        
                except Exception as e:
                    logger.warning(f"Failed to convert page {page_num}: {e}")
                    page_results[page_num] = PageProcessingResult(
                        index=page_num,
                        text=f"[Error convirtiendo página {page_num}: {e}]",
                        confidence=0.0,
                        processing_time=0.0,
                        error=str(e)
                    )
            
            # Process pages in parallel
            if worker_args:
                with ProcessPoolExecutor(max_workers=Config.OCR_WORKERS) as executor:
                    future_to_page = {executor.submit(process_pdf_page_worker, args): args[1] for args in worker_args}
                    
                    for future in as_completed(future_to_page):
                        try:
                            result = future.result()
                            page_results[result.index] = result
                            
                            if result.confidence > Config.MIN_CONFIDENCE_THRESHOLD:
                                processed_pages += 1
                                total_confidence += result.confidence
                            
                        except Exception as e:
                            page_num = future_to_page[future]
                            logger.error(f"Worker failed for page {page_num}: {e}")
                            page_results[page_num] = PageProcessingResult(
                                index=page_num,
                                text=f"[Error procesando página {page_num}]",
                                confidence=0.0,
                                processing_time=0.0,
                                error=str(e)
                            )
            
            # Cleanup temp files
            cleanup_temp_files(*temp_files)
        
        # Combine results in order
        for page_num in range(1, total_pages + 1):
            result = page_results.get(page_num, PageProcessingResult(
                index=page_num,
                text="[Página no procesada]",
                confidence=0.0,
                processing_time=0.0
            ))
            
            all_text += get_page_separator(page_num, total_pages)
            all_text += result.text
            if result.error:
                all_text += f"\n[ERROR: {result.error}]"
        
        # Extract legal fields from combined text
        legal_fields = extract_peruvian_legal_fields(all_text)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(all_text)
        
        # Calculate average confidence
        avg_confidence = total_confidence / max(1, processed_pages)
        
        metadata = {
            'key_values': legal_fields,
            'processing_method': 'advanced_ocr',
            'confidence': round(avg_confidence, 2),
            'total_pages': total_pages,
            'processed_pages': processed_pages,
            'low_confidence_pages': total_pages - processed_pages
        }
        
        memory_cleanup()
        
        return True, total_pages, all_text, metadata
        
    except Exception as e:
        logger.error(f"Scanned PDF processing failed: {e}")
        memory_cleanup()
        return False, 0, str(e), {}

# ----------------------------
# Flask Routes
# ----------------------------
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({
        'error': f'Archivo muy grande. Máximo permitido: {Config.MAX_CONTENT_LENGTH // (1024*1024)}MB'
    }), 413

@app.errorhandler(Exception)
def handle_general_error(e):
    logger.error(f"Unhandled error: {e}")
    return jsonify({'error': 'Error interno del servidor'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with system information."""
    try:
        # Test Tesseract
        test_result = pytesseract.get_tesseract_version()
        tesseract_ok = True
    except Exception:
        tesseract_ok = False
    
    return jsonify({
        'status': 'healthy' if tesseract_ok else 'degraded',
        'timestamp': datetime.now().isoformat(),
        'version': '4.0.0-peruvian-legal-optimized',
        'tesseract_version': test_result if tesseract_ok else 'unavailable',
        'config': {
            'dpi': Config.OCR_DPI,
            'language': Config.OCR_LANG_DEFAULT,
            'workers': Config.OCR_WORKERS,
            'max_file_size_mb': Config.MAX_CONTENT_LENGTH // (1024*1024),
            'confidence_threshold': Config.MIN_CONFIDENCE_THRESHOLD
        },
        'supported_formats': list(Config.ALLOWED_EXTENSIONS),
        'optimizations': [
            'peruvian_legal_patterns',
            'enhanced_preprocessing',
            'parallel_processing',
            'memory_management',
            'smart_pdf_analysis'
        ]
    })

@app.route('/analyze', methods=['POST'])
def analyze_document():
    """Analyze document structure without full processing."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No se proporcionó archivo'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó archivo'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'Formato no permitido. Use: {", ".join(Config.ALLOWED_EXTENSIONS)}'}), 400
        
        # Save temp file
        filename = secure_filename(file.filename)
        temp_path = os.path.join(Config.TEMP_FOLDER, f"analyze_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}")
        file.save(temp_path)
        
        try:
            if filename.lower().endswith('.pdf'):
                analysis = analyze_pdf_content(temp_path)
            else:
                # For image files
                analysis = {
                    'has_text': False,
                    'has_images': True,
                    'processing_recommendation': 'ocr',
                    'confidence': 0.8,
                    'file_size_mb': get_file_size_mb(temp_path)
                }
            
            return jsonify({
                'success': True,
                'filename': filename,
                'analysis': analysis,
                'timestamp': datetime.now().isoformat()
            })
            
        finally:
            cleanup_temp_files(temp_path)
            
    except Exception as e:
        logger.error(f"Document analysis failed: {e}")
        return jsonify({'error': f'Error analizando documento: {str(e)}'}), 500

@app.route('/process', methods=['POST'])
def process_document():
    """Main document processing endpoint with enhanced options."""
    processing_start = datetime.now()
    temp_files = []
    
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({'error': 'No se proporcionó archivo'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó archivo'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'Formato no permitido. Use: {", ".join(Config.ALLOWED_EXTENSIONS)}'}), 400
        
        # Get parameters
        language = request.form.get('language', Config.OCR_LANG_DEFAULT)
        output_format = request.form.get('format', 'json')  # json, txt, md
        force_ocr = request.form.get('force_ocr', '0') == '1'
        extract_fields = request.form.get('extract_fields', '1') == '1'
        
        # Validate language
        if language not in ['spa', 'eng', 'spa+eng']:
            language = 'spa'
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_filename = f"{timestamp}_{filename}"
        input_path = os.path.join(Config.UPLOAD_FOLDER, temp_filename)
        file.save(input_path)
        temp_files.append(input_path)
        
        # Check file size
        file_size_mb = get_file_size_mb(input_path)
        if file_size_mb > Config.MAX_CONTENT_LENGTH / (1024*1024):
            return jsonify({'error': f'Archivo muy grande: {file_size_mb:.1f}MB'}), 413
        
        # Prepare output
        output_filename = f"{timestamp}_{os.path.splitext(filename)[0]}.txt"
        output_path = os.path.join(Config.OUTPUT_FOLDER, output_filename)
        
        logger.info(f"Processing {filename} ({file_size_mb:.1f}MB) - Language: {language}, Force OCR: {force_ocr}")
        
        # Determine processing method
        if filename.lower().endswith('.pdf') and not force_ocr:
            analysis = analyze_pdf_content(input_path)
            processing_method = analysis['processing_recommendation']
        else:
            processing_method = 'ocr'
        
        # Process document
        if processing_method == 'text_extraction':
            success, pages, text, metadata = process_text_pdf_optimized(input_path, output_path)
        else:
            success, pages, text, metadata = process_scanned_pdf_optimized(input_path, output_path, language)
        
        processing_time = (datetime.now() - processing_start).total_seconds()
        
        if not success:
            return jsonify({
                'success': False,
                'error': text,
                'processing_time': round(processing_time, 2)
            }), 500
        
        # Prepare response data
        response_data = {
            'success': True,
            'filename': filename,
            'pages_processed': pages,
            'processing_time': round(processing_time, 2),
            'processing_method': metadata.get('processing_method', processing_method),
            'language': language,
            'file_size_mb': round(file_size_mb, 2),
            'confidence': metadata.get('confidence', 0.0),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add extracted fields if requested
        if extract_fields and metadata.get('key_values'):
            response_data['extracted_fields'] = metadata['key_values']
        
        # Handle different output formats
        if output_format == 'json':
            response_data['text'] = text
            return jsonify(response_data)
        
        elif output_format == 'md':
            # Generate markdown
            md_content = generate_markdown_report(filename, text, metadata, response_data)
            md_bytes = md_content.encode('utf-8')
            md_stream = io.BytesIO(md_bytes)
            md_stream.seek(0)
            
            return send_file(
                md_stream,
                as_attachment=True,
                download_name=f"{os.path.splitext(filename)[0]}.md",
                mimetype='text/markdown'
            )
        
        else:  # txt format
            return send_file(
                output_path,
                as_attachment=True,
                download_name=f"{os.path.splitext(filename)[0]}.txt",
                mimetype='text/plain'
            )
    
    except Exception as e:
        processing_time = (datetime.now() - processing_start).total_seconds()
        logger.error(f"Document processing failed: {e}")
        return jsonify({
            'success': False,
            'error': f'Error procesando documento: {str(e)}',
            'processing_time': round(processing_time, 2)
        }), 500
    
    finally:
        # Cleanup
        cleanup_temp_files(*temp_files)
        memory_cleanup()

def generate_markdown_report(filename: str, text: str, metadata: Dict, response_data: Dict) -> str:
    """Generate a comprehensive markdown report."""
    md_parts = [
        f"# Procesamiento OCR - {filename}",
        "",
        "## Información del Procesamiento",
        f"- **Archivo**: {filename}",
        f"- **Método**: {response_data.get('processing_method', 'N/A')}",
        f"- **Páginas**: {response_data.get('pages_processed', 0)}",
        f"- **Tiempo**: {response_data.get('processing_time', 0)}s",
        f"- **Confianza**: {response_data.get('confidence', 0)}%",
        f"- **Idioma**: {response_data.get('language', 'spa')}",
        f"- **Tamaño**: {response_data.get('file_size_mb', 0)}MB",
        ""
    ]
    
    # Add extracted fields
    extracted_fields = metadata.get('key_values', {})
    if extracted_fields:
        md_parts.extend([
            "## Campos Extraídos",
            ""
        ])
        for field, value in extracted_fields.items():
            field_name = field.replace('_', ' ').title()
            md_parts.append(f"- **{field_name}**: {value}")
        md_parts.append("")
    
    # Add text content
    md_parts.extend([
        "## Contenido Extraído",
        "",
        "```",
        text,
        "```",
        "",
        f"---",
        f"*Procesado el {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}*"
    ])
    
    return "\n".join(md_parts)

@app.route('/batch', methods=['POST'])
def process_batch():
    """Batch processing endpoint for multiple documents."""
    try:
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No se proporcionaron archivos'}), 400
        
        # Parameters
        language = request.form.get('language', Config.OCR_LANG_DEFAULT)
        force_ocr = request.form.get('force_ocr', '0') == '1'
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_folder = os.path.join(Config.OUTPUT_FOLDER, f'batch_{timestamp}')
        os.makedirs(batch_folder, exist_ok=True)
        
        results = []
        temp_files = []
        
        logger.info(f"Processing batch of {len(files)} files")
        
        for i, file in enumerate(files, 1):
            if file.filename and allowed_file(file.filename):
                file_start = datetime.now()
                try:
                    filename = secure_filename(file.filename)
                    logger.info(f"Processing batch file {i}/{len(files)}: {filename}")
                    
                    # Save temp file
                    temp_path = os.path.join(Config.TEMP_FOLDER, f"{timestamp}_{filename}")
                    file.save(temp_path)
                    temp_files.append(temp_path)
                    
                    # Output paths
                    base_name = os.path.splitext(filename)[0]
                    txt_output = os.path.join(batch_folder, f"{base_name}.txt")
                    json_output = os.path.join(batch_folder, f"{base_name}.json")
                    
                    # Process file
                    if filename.lower().endswith('.pdf') and not force_ocr:
                        analysis = analyze_pdf_content(temp_path)
                        if analysis['processing_recommendation'] == 'text_extraction':
                            success, pages, text, metadata = process_text_pdf_optimized(temp_path, txt_output)
                        else:
                            success, pages, text, metadata = process_scanned_pdf_optimized(temp_path, txt_output, language)
                    else:
                        success, pages, text, metadata = process_scanned_pdf_optimized(temp_path, txt_output, language)
                    
                    processing_time = (datetime.now() - file_start).total_seconds()
                    
                    # Save metadata
                    file_metadata = {
                        'filename': filename,
                        'success': success,
                        'pages': pages,
                        'processing_time': round(processing_time, 2),
                        'method': metadata.get('processing_method', 'ocr'),
                        'confidence': metadata.get('confidence', 0.0),
                        'extracted_fields': metadata.get('key_values', {}),
                        'timestamp': datetime.now().isoformat(),
                        'file_size_mb': round(get_file_size_mb(temp_path), 2)
                    }
                    
                    with open(json_output, 'w', encoding='utf-8') as jf:
                        json.dump(file_metadata, jf, ensure_ascii=False, indent=2)
                    
                    results.append({
                        'filename': filename,
                        'success': success,
                        'pages': pages,
                        'processing_time': round(processing_time, 2),
                        'confidence': metadata.get('confidence', 0.0),
                        'error': None if success else 'Processing failed'
                    })
                    
                except Exception as e:
                    logger.error(f"Batch file {filename} failed: {e}")
                    results.append({
                        'filename': file.filename,
                        'success': False,
                        'pages': 0,
                        'processing_time': 0.0,
                        'confidence': 0.0,
                        'error': str(e)
                    })
        
        # Clean up temp files
        cleanup_temp_files(*temp_files)
        
        # Create summary
        successful = len([r for r in results if r['success']])
        failed = len(results) - successful
        
        batch_summary = {
            'batch_id': timestamp,
            'total_files': len(results),
            'successful': successful,
            'failed': failed,
            'language': language,
            'force_ocr': force_ocr,
            'results': results,
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'version': '4.0.0-peruvian-legal-optimized',
                'workers': Config.OCR_WORKERS,
                'dpi': Config.OCR_DPI
            }
        }
        
        # Create ZIP file
        zip_filename = f'lote_procesado_{timestamp}.zip'
        zip_path = os.path.join(Config.OUTPUT_FOLDER, zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add processed files
            for root, _, files in os.walk(batch_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, file)
            
            # Add summary
            zipf.writestr('resumen_lote.json', json.dumps(batch_summary, indent=2, ensure_ascii=False))
        
        logger.info(f"Batch processing completed: {successful}/{len(results)} successful")
        memory_cleanup()
        
        return send_file(zip_path, as_attachment=True, download_name=zip_filename)
    
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        return jsonify({'error': f'Error procesando lote: {str(e)}'}), 500

@app.route('/output/<filename>')
def get_output_file(filename):
    """Serve output files."""
    return send_from_directory(Config.OUTPUT_FOLDER, filename)

@app.route('/')
def home():
    """Simple home page."""
    return jsonify({
        'service': 'Servicio OCR para Documentos Legales Peruanos',
        'version': '4.0.0-optimized',
        'endpoints': {
            '/health': 'Estado del servicio',
            '/analyze': 'Analizar documento (POST)',
            '/process': 'Procesar documento (POST)',
            '/batch': 'Procesamiento por lotes (POST)'
        },
        'supported_formats': list(Config.ALLOWED_EXTENSIONS),
        'optimized_for': ['Documentos legales peruanos', 'Fiscalía', 'Juzgados', 'Expedientes']
    })

# ----------------------------
# Application Startup
# ----------------------------
def initialize_service():
    """Initialize the service with custom corrections and optimizations."""
    logger.info("Inicializando servicio OCR para documentos legales peruanos...")
    
    # Log configuration
    logger.info(f"Configuración: DPI={Config.OCR_DPI}, Workers={Config.OCR_WORKERS}, Lang={Config.OCR_LANG_DEFAULT}")
    logger.info(f"Memoria límite: {Config.MEMORY_LIMIT_MB}MB")
    logger.info(f"Confianza mínima: {Config.MIN_CONFIDENCE_THRESHOLD}%")
    
    # Test Tesseract
    try:
        version = pytesseract.get_tesseract_version()
        logger.info(f"Tesseract version: {version}")
    except Exception as e:
        logger.error(f"Tesseract test failed: {e}")
    
    logger.info("Servicio OCR listo para procesar documentos legales peruanos")

if __name__ == '__main__':
    initialize_service()
    
    # Production settings
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('DEBUG', 'False').lower() == 'true',
        threaded=True
    )