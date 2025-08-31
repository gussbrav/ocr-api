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

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuración
app.config['UPLOAD_FOLDER'] = '/app/input'
app.config['OUTPUT_FOLDER'] = '/app/output'
app.config['TEMP_FOLDER'] = '/app/temp'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# Crear directorios
for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER'], app.config['TEMP_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# Configurar Tesseract para Linux
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def contains_bitmap(pdf_path):
    """Verificar si PDF contiene imágenes bitmap (basado en tu código original)"""
    try:
        with open(pdf_path, 'rb') as f:
            pdf = PdfReader(f)
            pdf_text = ""
            for page in pdf.pages:
                pdf_text += page.extract_text() if page.extract_text() else ""
            
            # Si la extracción de texto produce muy poco texto, probablemente sea bitmap
            if len(pdf_text.strip()) < 100:
                return True
                
            # Buscar objetos de imagen en el PDF
            pattern = re.compile(r'/Subtype\s*/Image', re.IGNORECASE)
            for page in pdf.pages:
                page_obj = page.get_object() if hasattr(page, 'get_object') else page
                if pattern.search(str(page_obj)):
                    return True
                        
        return False
    except Exception as e:
        logger.warning(f"Error checking PDF content: {e}")
        return True  # Si no podemos determinar, asumimos que contiene bitmaps

def process_text_pdf(pdf_path, output_path):
    """Procesar PDFs que son principalmente texto (basado en tu código)"""
    try:
        text = ""
        with open(pdf_path, 'rb') as f:
            pdf = PdfReader(f)
            for page in pdf.pages:
                text += page.extract_text() + "\n\n"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        return True, len(pdf.pages), text
    except Exception as e:
        logger.error(f"Error processing text PDF: {e}")
        return False, 0, str(e)

def process_bitmap_pdf(pdf_path, output_path, language='spa'):
    """Procesar PDFs que contienen imágenes bitmap (mejorado con tu lógica)"""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Convertir PDF a imágenes
            images = convert_from_path(pdf_path, dpi=300)
            all_text = ""
            
            for i, image in enumerate(images):
                try:
                    # Guardar imagen como PPM
                    ppm_file = f"{temp_dir}/page_{i}.ppm"
                    image.save(ppm_file, "PPM")
                    
                    # Convertir PPM a TIFF con mejoras (tu lógica original)
                    tiff_file = ppm_file.replace('.ppm', '.tiff')
                    img = cv2.imread(ppm_file)
                    
                    # Duplicar tamaño (200%) para mejor OCR
                    h, w = img.shape[:2]
                    img = cv2.resize(img, (2*w, 2*h), interpolation=cv2.INTER_CUBIC)
                    
                    # Convertir a escala de grises
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Convertir a monocromático (binario)
                    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    
                    # Detectar ángulo de inclinación usando Hough Line Transform (tu implementación)
                    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
                    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
                    
                    if lines is not None:
                        angles = []
                        for line in lines:
                            for rho, theta in line:
                                # Solo considerar líneas horizontales o verticales
                                if abs(theta - 0) < 0.1 or abs(theta - np.pi/2) < 0.1 or abs(theta - np.pi) < 0.1:
                                    angles.append(theta)
                        
                        if angles:
                            median_angle = np.median(angles)
                            angle_degrees = np.degrees(median_angle)
                            
                            if angle_degrees > 45:
                                angle_degrees = 90 - angle_degrees
                            elif angle_degrees < -45:
                                angle_degrees = -90 - angle_degrees
                            
                            # Solo rotar si la inclinación es significativa
                            if abs(angle_degrees) > 0.5:
                                h, w = binary.shape
                                center = (w//2, h//2)
                                M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
                                binary = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC, 
                                                      borderMode=cv2.BORDER_CONSTANT, borderValue=255)
                    
                    # Guardar como TIFF
                    cv2.imwrite(tiff_file, binary)
                    
                    # OCR con Tesseract usando el idioma especificado
                    text = pytesseract.image_to_string(tiff_file, lang=language)
                    all_text += f"\n--- Página {i + 1} ---\n{text}\n\n"
                    
                except Exception as e:
                    logger.error(f"Error processing page {i + 1}: {e}")
                    all_text += f"\n--- Página {i + 1} ---\nError procesando página: {str(e)}\n\n"
            
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
        'version': '1.0.0'
    })

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
        
        # Verificar si PDF contiene bitmap y procesar en consecuencia
        start_time = datetime.now()
        
        if contains_bitmap(file_path):
            logger.info("PDF contains bitmap images, using OCR")
            success, pages_processed, text = process_bitmap_pdf(file_path, output_path, language)
            method = 'ocr'
        else:
            logger.info("PDF contains text, using text extraction")
            success, pages_processed, text = process_text_pdf(file_path, output_path)
            method = 'text_extraction'
        
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
                'error': text,  # text contiene el error en caso de fallo
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
    """Procesar múltiples PDFs"""
    try:
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No files provided'}), 400
        
        language = request.form.get('language', 'spa')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_folder = os.path.join(app.config['OUTPUT_FOLDER'], f'batch_{timestamp}')
        os.makedirs(batch_folder, exist_ok=True)
        
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
                        method = 'ocr'
                    else:
                        success, pages, text = process_text_pdf(file_path, output_path)
                        method = 'text_extraction'
                    
                    results.append({
                        'filename': filename,
                        'success': success,
                        'pages': pages,
                        'method': method,
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
        zip_filename = f'ocr_batch_{timestamp}.zip'
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
    # Sirve el index.html que está en /app (raíz del contenedor)
    return send_from_directory(app.root_path, 'index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)