import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import pytesseract
from pdf2image import convert_from_path
import tempfile
from PyPDF2 import PdfReader
import re
import logging
from datetime import datetime
import zipfile
from PIL import Image, ImageEnhance, ImageFilter

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuración
app.config['UPLOAD_FOLDER'] = '/app/input'
app.config['OUTPUT_FOLDER'] = '/app/output'
app.config['TEMP_FOLDER'] = '/app/temp'
app.config['TRAINING_FOLDER'] = '/app/training'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# Crear directorios
for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER'], 
               app.config['TEMP_FOLDER'], app.config['TRAINING_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# Configurar Tesseract para Linux
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Diccionario de correcciones comunes para caracteres especiales
CHAR_CORRECTIONS = {
    'N0': 'N°',
    'NO': 'N°',
    'Np': 'N°',
    'N8': 'N°',
    'N6': 'N°',
    'Nro': 'Nro',
    'NRO': 'Nro',
    'nro': 'Nro',
    'Nr0': 'Nro',
    'Nrp': 'Nro',
    'Num': 'Núm',
    'NUM': 'Núm',
    'núm': 'Núm',
    'numero': 'número',
    'NUMERO': 'NÚMERO',
    'JUZGADO': 'JUZGADO',
    'CIVIL': 'CIVIL',
    'PENAL': 'PENAL',
    'EXPEDIENTE': 'EXPEDIENTE',
    'EXP': 'EXP',
    'RESOLUCION': 'RESOLUCIÓN',
    'RESOLUCIÖN': 'RESOLUCIÓN',
    'DIGITALIZACION': 'DIGITALIZACIÓN',
    'DIGITALIZACIÖN': 'DIGITALIZACIÓN'
}

# Patrones regex para identificar y corregir formatos específicos
LEGAL_PATTERNS = [
    (r'N[0O8p6]\s*(\d+)', r'N° \1'),  # N0 123 -> N° 123
    (r'Nr[0o]\s*(\d+)', r'Nro \1'),   # Nr0 123 -> Nro 123  
    (r'[Nn][úu]m\s*(\d+)', r'Núm \1'), # num 123 -> Núm 123
    (r'EXP\s*[\.:]?\s*N[0O8p6]\s*(\d+)', r'EXP N° \1'), # EXP N0 123 -> EXP N° 123
    (r'EXPEDIENTE\s*N[0O8p6]\s*(\d+)', r'EXPEDIENTE N° \1'),
    (r'RESOLUCI[ÖO]N\s*N[0O8p6]\s*(\d+)', r'RESOLUCIÓN N° \1'),
    (r'(\d{4})-(\d{4})-(\w{3})-(\w{2})-(\w{2})', r'\1-\2-\3-\4-\5'), # Formato de expediente
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def contains_bitmap(pdf_path):
    """Verificar si PDF contiene imágenes bitmap"""
    try:
        with open(pdf_path, 'rb') as f:
            pdf = PdfReader(f)
            pdf_text = ""
            for page in pdf.pages:
                pdf_text += page.extract_text() if page.extract_text() else ""
            
            if len(pdf_text.strip()) < 100:
                return True
                
            pattern = re.compile(r'/Subtype\s*/Image', re.IGNORECASE)
            for page in pdf.pages:
                page_obj = page.get_object() if hasattr(page, 'get_object') else page
                if pattern.search(str(page_obj)):
                    return True
                        
        return False
    except Exception as e:
        logger.warning(f"Error checking PDF content: {e}")
        return True

def get_page_separator(page_num, total_pages):
    """Generar separador de página con formato consistente"""
    return f"\n\n--- PÁGINA {page_num} DE {total_pages}\n\n"

def enhance_image_for_ocr(image_path):
    """Mejorar imagen específicamente para caracteres legales"""
    try:
        # Abrir imagen con PIL
        img = Image.open(image_path)
        
        # Aumentar contraste
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)
        
        # Aumentar nitidez
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(2.0)
        
        # Convertir a OpenCV
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Redimensionar para mejor OCR (300-400% del tamaño original)
        h, w = img_cv.shape[:2]
        img_cv = cv2.resize(img_cv, (int(w * 3.5), int(h * 3.5)), interpolation=cv2.INTER_CUBIC)
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Aplicar filtro de desenfoque gaussiano suave
        blurred = cv2.GaussianBlur(gray, (1, 1), 0)
        
        # Umbralización adaptativa para mejor reconocimiento de caracteres
        adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY, 11, 2)
        
        # Operaciones morfológicas para limpiar caracteres
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
        
    except Exception as e:
        logger.error(f"Error enhancing image: {e}")
        return None

def correct_ocr_text(text):
    """Aplicar correcciones específicas al texto OCR"""
    corrected_text = text
    
    # Aplicar correcciones de diccionario
    for wrong, correct in CHAR_CORRECTIONS.items():
        corrected_text = re.sub(r'\b' + re.escape(wrong) + r'\b', correct, corrected_text, flags=re.IGNORECASE)
    
    # Aplicar patrones regex
    for pattern, replacement in LEGAL_PATTERNS:
        corrected_text = re.sub(pattern, replacement, corrected_text, flags=re.IGNORECASE)
    
    return corrected_text

def perform_advanced_ocr(image_path, language='spa'):
    """Realizar OCR avanzado con múltiples configuraciones"""
    try:
        # Configuraciones de Tesseract optimizadas para documentos legales
        configs = [
            '--psm 6 --oem 1',  # Uniform text block
            '--psm 4 --oem 1',  # Single column of text
            '--psm 3 --oem 1',  # Fully automatic page segmentation
            '--psm 6 --oem 1 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNÑOPQRSTUVWXYZabcdefghijklmnñopqrstuvwxyzÁÉÍÓÚáéíóú°:-/',
        ]
        
        best_text = ""
        max_confidence = 0
        
        for config in configs:
            try:
                # Obtener texto con confianza
                data = pytesseract.image_to_data(image_path, lang=language, config=config, output_type=pytesseract.Output.DICT)
                
                # Calcular confianza promedio
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_confidence = np.mean(confidences) if confidences else 0
                
                if avg_confidence > max_confidence:
                    max_confidence = avg_confidence
                    text = pytesseract.image_to_string(image_path, lang=language, config=config)
                    best_text = correct_ocr_text(text)
                    
            except Exception as e:
                logger.warning(f"OCR config failed: {e}")
                continue
        
        # Si no se obtuvo buen resultado, usar configuración básica
        if not best_text.strip():
            text = pytesseract.image_to_string(image_path, lang=language)
            best_text = correct_ocr_text(text)
        
        return best_text
        
    except Exception as e:
        logger.error(f"Advanced OCR failed: {e}")
        return ""

def process_text_pdf(pdf_path, output_path):
    """Procesar PDFs que son principalmente texto"""
    try:
        text = ""
        with open(pdf_path, 'rb') as f:
            pdf = PdfReader(f)
            total_pages = len(pdf.pages)
            
            for i, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    # Agregar separador de página
                    text += get_page_separator(i, total_pages)
                    # Aplicar correcciones al texto extraído
                    corrected_text = correct_ocr_text(page_text)
                    text += corrected_text
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        return True, len(pdf.pages), text
    except Exception as e:
        logger.error(f"Error processing text PDF: {e}")
        return False, 0, str(e)

def process_bitmap_pdf(pdf_path, output_path, language='spa'):
    """Procesar PDFs que contienen imágenes bitmap con OCR mejorado"""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Convertir PDF a imágenes con mayor DPI para mejor calidad
            images = convert_from_path(pdf_path, dpi=400)
            all_text = ""
            total_pages = len(images)
            
            for i, image in enumerate(images):
                try:
                    # Agregar separador de página
                    page_separator = get_page_separator(i + 1, total_pages)
                    all_text += page_separator
                    
                    # Guardar imagen como PNG (mejor calidad que PPM)
                    png_file = f"{temp_dir}/page_{i}.png"
                    image.save(png_file, "PNG", quality=100, optimize=False)
                    
                    # Mejorar imagen para OCR
                    enhanced_img = enhance_image_for_ocr(png_file)
                    
                    if enhanced_img is not None:
                        # Guardar imagen mejorada
                        enhanced_file = f"{temp_dir}/enhanced_page_{i}.png"
                        cv2.imwrite(enhanced_file, enhanced_img)
                        
                        # Realizar OCR avanzado
                        text = perform_advanced_ocr(enhanced_file, language)
                    else:
                        # Fallback a OCR básico si la mejora falla
                        text = pytesseract.image_to_string(png_file, lang=language)
                        text = correct_ocr_text(text)
                    
                    all_text += text
                    
                except Exception as e:
                    logger.error(f"Error processing page {i + 1}: {e}")
                    error_separator = get_page_separator(i + 1, total_pages)
                    all_text += f"{error_separator}Error procesando página: {str(e)}"
            
            # Escribir texto final al archivo de salida
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(all_text)
                
            return True, len(images), all_text
            
    except Exception as e:
        logger.error(f"Error processing bitmap PDF: {e}")
        return False, 0, str(e)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0'
    })

@app.route('/ocr/train', methods=['POST'])
def train_corrections():
    """Endpoint para entrenar correcciones personalizadas"""
    try:
        data = request.get_json()
        if not data or 'corrections' not in data:
            return jsonify({'error': 'No corrections data provided'}), 400
        
        corrections_file = os.path.join(app.config['TRAINING_FOLDER'], 'custom_corrections.txt')
        
        # Guardar correcciones personalizadas
        with open(corrections_file, 'a', encoding='utf-8') as f:
            for wrong, correct in data['corrections'].items():
                f.write(f"{wrong}|{correct}\n")
        
        # Cargar correcciones existentes
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
    """Cargar correcciones personalizadas desde archivo"""
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
        
        # Parámetros opcionales
        language = request.form.get('language', 'spa')
        output_format = request.form.get('format', 'txt')  # txt o json
        
        # Guardar archivo
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        file.save(file_path)
        
        # Generar nombre de archivo de salida
        output_filename = f"{timestamp}_{os.path.splitext(filename)[0]}.txt"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        logger.info(f"Processing file: {temp_filename}")
        
        # Cargar correcciones personalizadas
        load_custom_corrections()
        
        # Verificar si PDF contiene bitmap y procesar en consecuencia
        start_time = datetime.now()
        
        if contains_bitmap(file_path):
            logger.info("PDF contains bitmap images, using enhanced OCR")
            success, pages_processed, text = process_bitmap_pdf(file_path, output_path, language)
            method = 'enhanced_ocr'
        else:
            logger.info("PDF contains text, using text extraction with corrections")
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
                # Devolver archivo de texto
                return send_file(output_path, 
                               as_attachment=True, 
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
        # Limpiar archivo temporal
        try:
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.warning(f"Could not remove temp file: {e}")

@app.route('/ocr/batch', methods=['POST'])
def process_batch():
    """Procesar múltiples PDFs con OCR mejorado"""
    try:
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No files provided'}), 400
        
        language = request.form.get('language', 'spa')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_folder = os.path.join(app.config['OUTPUT_FOLDER'], f'batch_{timestamp}')
        os.makedirs(batch_folder, exist_ok=True)
        
        # Cargar correcciones personalizadas
        load_custom_corrections()
        
        results = []
        temp_files = []
        
        for file in files:
            if file.filename != '' and allowed_file(file.filename):
                try:
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(app.config['TEMP_FOLDER'], f"{timestamp}_{filename}")
                    file.save(file_path)
                    temp_files.append(file_path)
                    
                    # Procesar archivo
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
        
        # Limpiar archivos temporales
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass
        
        # Crear ZIP con resultados
        zip_filename = f'ocr_batch_enhanced_{timestamp}.zip'
        zip_path = os.path.join(app.config['OUTPUT_FOLDER'], zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # Agregar archivos de texto
            for root, dirs, files in os.walk(batch_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, file)
            
            # Agregar resumen JSON
            summary_data = {
                'batch_id': timestamp,
                'processed_files': len(results),
                'successful': len([r for r in results if r['success']]),
                'failed': len([r for r in results if not r['success']]),
                'total_corrections_available': len(CHAR_CORRECTIONS),
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
            
            zipf.writestr('batch_summary.json', 
                         __import__('json').dumps(summary_data, indent=2, ensure_ascii=False))
        
        return send_file(zip_path, as_attachment=True, download_name=zip_filename)
        
    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}")
        return jsonify({'error': f'Batch processing failed: {str(e)}'}), 500

@app.route('/output/<filename>')
def get_file(filename):
    """Descargar archivo de salida"""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/', methods=['GET'])
def home():
    return send_from_directory(app.root_path, 'index.html')

# Cargar correcciones personalizadas al iniciar
load_custom_corrections()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)