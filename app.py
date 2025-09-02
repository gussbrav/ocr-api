import os
import io
import cv2
import gc
import re
import json
import math
import uuid
import zipfile
import logging
import tempfile
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Any, Optional, Union
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor

import aiohttp
import asyncio
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, String, Text, DateTime, JSON, Integer, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pytesseract
from PyPDF2 import PdfReader, PdfWriter
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image, ImageEnhance, ImageOps
import redis
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import jwt
from jwt.exceptions import InvalidTokenError
from passlib.context import CryptContext

# ----------------------------
# Configuración y Constantes
# ----------------------------
class Config:
    # Directorios
    UPLOAD_FOLDER = "/app/data/uploads"
    OUTPUT_FOLDER = "/app/data/outputs"
    TEMP_FOLDER = "/app/data/temp"
    TRAINING_FOLDER = "/app/data/training"
    MODEL_FOLDER = "/app/data/models"
    
    # Redis
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB = int(os.getenv("REDIS_DB", 0))
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/ocr_database.db")
    
    # Security
    SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-key-change-in-production")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    
    # OCR Configuration
    OCR_LANG_DEFAULT = os.getenv("OCR_LANG", "spa+eng")
    OCR_DPI = int(os.getenv("OCR_DPI", "450"))
    OCR_SCALE = float(os.getenv("OCR_SCALE", "2.5"))
    OCR_WORKERS = int(os.getenv("OCR_WORKERS", max(1, os.cpu_count() - 1)))
    OCR_TIMEOUT = int(os.getenv("OCR_TIMEOUT", "300"))
    
    # API Limits
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    RATE_LIMIT_REQUESTS = 100
    RATE_LIMIT_PERIOD = 3600  # 1 hour

# Crear directorios
for folder in [Config.UPLOAD_FOLDER, Config.OUTPUT_FOLDER, 
               Config.TEMP_FOLDER, Config.TRAINING_FOLDER, Config.MODEL_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# ----------------------------
# Database Setup
# ----------------------------
Base = declarative_base()

class OCRJob(Base):
    __tablename__ = "ocr_jobs"
    
    id = Column(String(36), primary_key=True, index=True)
    filename = Column(String(255))
    status = Column(String(20), default="processing")  # processing, completed, failed
    language = Column(String(10), default="spa")
    pages_processed = Column(Integer, default=0)
    total_pages = Column(Integer, default=0)
    processing_time = Column(Float, default=0.0)
    method = Column(String(50))
    key_values = Column(JSON)
    tables = Column(JSON)
    error = Column(Text)
    user_id = Column(String(36))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class User(Base):
    __tablename__ = "users"
    
    id = Column(String(36), primary_key=True)
    username = Column(String(50), unique=True, index=True)
    email = Column(String(100), unique=True, index=True)
    hashed_password = Column(String(255))
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    api_key = Column(String(64), unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)

# Database engine
engine = create_engine(Config.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

# ----------------------------
# Redis Setup
# ----------------------------
redis_client = redis.Redis(
    host=Config.REDIS_HOST,
    port=Config.REDIS_PORT,
    db=Config.REDIS_DB,
    decode_responses=True
)

# ----------------------------
# Security
# ----------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# ----------------------------
# Metrics
# ----------------------------
REQUEST_COUNTER = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
PROCESSING_TIME = Histogram('api_processing_time_seconds', 'API processing time')
OCR_TIME = Histogram('ocr_processing_time_seconds', 'OCR processing time per page')

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(
    title="DeepSeek OCR API",
    description="API avanzada para OCR de documentos legales peruanos",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Modelos Pydantic
# ----------------------------
class OCRRequest(BaseModel):
    language: str = Field(default=Config.OCR_LANG_DEFAULT, description="Idioma para OCR")
    format: str = Field(default="json", description="Formato de salida: json, text, markdown")
    tables: bool = Field(default=False, description="Extraer tablas")
    legal_mode: bool = Field(default=True, description="Modo especializado para documentos legales")
    enhance: bool = Field(default=True, description="Mejorar imagen antes de OCR")

class OCRResponse(BaseModel):
    success: bool
    job_id: str
    status: str
    message: str
    estimated_time: Optional[float] = None

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    filename: str
    pages_processed: int
    total_pages: int
    processing_time: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

# ----------------------------
# Utilidades
# ----------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: SessionLocal = Depends(get_db)):
    try:
        payload = jwt.decode(credentials.credentials, Config.SECRET_KEY, algorithms=[Config.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    
    user = db.query(User).filter(User.username == username).first()
    if user is None or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    
    return user

def rate_limit(user_id: str, endpoint: str):
    key = f"rate_limit:{user_id}:{endpoint}"
    current = redis_client.incr(key)
    if current == 1:
        redis_client.expire(key, Config.RATE_LIMIT_PERIOD)
    
    if current > Config.RATE_LIMIT_REQUESTS:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

# ----------------------------
# Configuración Tesseract
# ----------------------------
pytesseract.pytesseract.tesseract_cmd = os.environ.get('TESSERACT_CMD', '/usr/bin/tesseract')

# ----------------------------
# Correcciones para documentos legales peruanos (mejorado)
# ----------------------------
LEGAL_CORRECTIONS = {
    # Términos legales
    'Nro': 'Nro', 'NRO': 'Nro', 'nro': 'Nro', 'Nr0': 'Nro', 'Nrp': 'Nro',
    'Num': 'Núm', 'NUM': 'Núm', 'núm': 'Núm', 'numero': 'número', 'NUMERO': 'NÚMERO',
    'EXPEDIENTE': 'EXPEDIENTE', 'EXP': 'EXP', 'RESOLUCION': 'RESOLUCIÓN',
    'FISCALIA': 'FISCALÍA', 'INVESTIGACION': 'INVESTIGACIÓN', 'APROPIACION': 'APROPIACIÓN',
    'USURPACION': 'USURPACIÓN', 'NOTIFICACION': 'NOTIFICACIÓN', 'REPROGRAMACION': 'REPROGRAMACIÓN',
    'INSPECCION': 'INSPECCIÓN', 'COMISARIA': 'COMISARÍA', 'OFICIOSE': 'OFÍCIOSE',
    
    # Nombres comunes
    'MERCADO': 'MERCADO', 'VARCARCEL': 'VARCARCEL', 'SAPACAYO': 'SAPACAYO',
    'MOLLO': 'MOLLO', 'BACA': 'BACA', 'ALEJANDRINA': 'ALEJANDRINA',
    
    # Lugares
    'CHIVAY': 'CHIVAY', 'CAYLLOMA': 'CAYLLOMA', 'AREQUIPA': 'AREQUIPA',
}

LEGAL_PATTERNS = [
    (r'N[0O8p6]\s*(\d+)', r'N° \1'),
    (r'Nr[0o]\s*(\d+)', r'Nro \1'),
    (r'EXP\s*[\.:]?\s*N[0O8p6]\s*(\d+)', r'EXP N° \1'),
    (r'ART[\.\s]*(\d+)', r'ART. \1'),
    (r'CODIGO\s*(PENAL|PROCESAL|CIVIL)', r'CÓDIGO \1'),
]

PERU_LEGAL_PATTERNS = {
    'fiscalia': [re.compile(r'\bFISCAL[ÍI]A.*?[:\s]+(.+?)(?=\n|$)', re.I)],
    'expediente': [re.compile(r'\bEXPEDIENTE[:\s]*([\w\d\-\.]+)', re.I)],
    'providencia': [re.compile(r'\bPROVIDENCIA[:\s]*N[°ºo]?\s*([\d\-]+)', re.I)],
    'imputados': [re.compile(r'\bIMPUTADO[SD]?[:\s]+(.+?)(?=\bAGRAVIADO\b|\bDELITO\b|$)', re.I)],
    'agraviado': [re.compile(r'\bAGRAVIADO[:\s]+(.+?)(?=\bDELITO\b|\bIMPUTADO\b|$)', re.I)],
    'delito': [re.compile(r'\bDELITO[:\s]+(.+?)(?=\bMODALIDAD\b|\bAGRAVIADO\b|$)', re.I)],
    'fecha': [re.compile(r'\b(\d{1,2}\s+de\s+[a-z]+\s+de\s+\d{4})', re.I)],
}

# ----------------------------
# Procesamiento de Imágenes (Mejorado)
# ----------------------------
class AdvancedImageProcessor:
    @staticmethod
    def enhance_image(image_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Mejora avanzada de imágenes para OCR"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None, None

            # Convertir a escala de grises
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Mejorar contraste con CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Reducción de ruido
            denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
            
            # Binarización adaptativa
            binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            return binary, img
            
        except Exception as e:
            logger.error(f"Image enhancement error: {e}")
            return None, None

    @staticmethod
    def detect_skew(image: np.ndarray) -> float:
        """Detección de inclinación con múltiples métodos"""
        try:
            # Método 1: Hough Lines
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
            
            angles = []
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                    if abs(angle) < 45:  # Solo ángulos razonables
                        angles.append(angle)
            
            if angles:
                return np.median(angles)
            return 0.0
            
        except Exception:
            return 0.0

# ----------------------------
# OCR Avanzado
# ----------------------------
class DeepSeekOCR:
    def __init__(self):
        self.processor = AdvancedImageProcessor()
    
    async def process_page(self, image_path: str, language: str = "spa") -> str:
        """Procesamiento de una página con OCR avanzado"""
        start_time = datetime.now()
        
        try:
            # Mejorar imagen
            enhanced_img, color_img = self.processor.enhance_image(image_path)
            
            if enhanced_img is None:
                # Fallback a OCR básico
                text = pytesseract.image_to_string(image_path, lang=language)
            else:
                # Guardar imagen mejorada temporalmente
                temp_path = f"{image_path}_enhanced.png"
                cv2.imwrite(temp_path, enhanced_img)
                
                # OCR con múltiples configuraciones
                configs = [
                    "--psm 6 --oem 1",
                    "--psm 4 --oem 1", 
                    "--psm 11 --oem 1"
                ]
                
                texts = []
                for config in configs:
                    try:
                        text = pytesseract.image_to_string(temp_path, lang=language, config=config)
                        texts.append(text.strip())
                    except Exception:
                        continue
                
                # Elegir el mejor texto (más largo y con mejor formato)
                if texts:
                    text = max(texts, key=lambda x: (len(x), -x.count('\n')))
                else:
                    text = pytesseract.image_to_string(temp_path, lang=language)
                
                # Limpiar archivo temporal
                try:
                    os.remove(temp_path)
                except:
                    pass
            
            # Aplicar correcciones
            corrected_text = self.apply_corrections(text)
            
            # Métricas
            processing_time = (datetime.now() - start_time).total_seconds()
            OCR_TIME.observe(processing_time)
            
            return corrected_text
            
        except Exception as e:
            logger.error(f"OCR processing error: {e}")
            return f"Error processing page: {str(e)}"
    
    def apply_corrections(self, text: str) -> str:
        """Aplicar correcciones específicas para documentos legales"""
        for wrong, correct in LEGAL_CORRECTIONS.items():
            text = re.sub(rf'\b{re.escape(wrong)}\b', correct, text, flags=re.IGNORECASE)
        
        for pattern, replacement in LEGAL_PATTERNS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def extract_legal_info(self, text: str) -> Dict[str, str]:
        """Extraer información legal específica"""
        info = {}
        for field, patterns in PERU_LEGAL_PATTERNS.items():
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    info[field] = match.group(1).strip()
                    break
        return info

# ----------------------------
# Procesamiento de PDFs
# ----------------------------
class PDFProcessor:
    def __init__(self):
        self.ocr_engine = DeepSeekOCR()
    
    async def process_pdf(self, pdf_path: str, output_path: str, language: str = "spa", 
                         extract_tables: bool = False, legal_mode: bool = True) -> Dict[str, Any]:
        """Procesar PDF completo"""
        start_time = datetime.now()
        
        try:
            # Detectar tipo de PDF
            is_scanned = self.is_scanned_pdf(pdf_path)
            
            if not is_scanned:
                # PDF con texto, extraer directamente
                text = self.extract_text_from_pdf(pdf_path)
                legal_info = self.ocr_engine.extract_legal_info(text)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                return {
                    "success": True,
                    "text": text,
                    "legal_info": legal_info,
                    "pages": 1,
                    "method": "text_extraction"
                }
            else:
                # PDF escaneado, procesar con OCR
                return await self.process_scanned_pdf(pdf_path, output_path, language, extract_tables, legal_mode)
                
        except Exception as e:
            logger.error(f"PDF processing error: {e}")
            return {
                "success": False,
                "error": str(e),
                "pages": 0,
                "method": "failed"
            }
        finally:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"PDF processing completed in {processing_time:.2f} seconds")
    
    def is_scanned_pdf(self, pdf_path: str) -> bool:
        """Determinar si el PDF es escaneado"""
        try:
            with open(pdf_path, 'rb') as f:
                reader = PdfReader(f)
                if len(reader.pages) == 0:
                    return True
                
                # Verificar si tiene texto extraíble
                text = reader.pages[0].extract_text() or ""
                return len(text.strip()) < 100  # Poco texto = probablemente escaneado
                
        except Exception:
            return True
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extraer texto de PDF con texto nativo"""
        text = ""
        try:
            with open(pdf_path, 'rb') as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n\n"
        except Exception as e:
            logger.error(f"Text extraction error: {e}")
        
        return text
    
    async def process_scanned_pdf(self, pdf_path: str, output_path: str, language: str,
                                extract_tables: bool, legal_mode: bool) -> Dict[str, Any]:
        """Procesar PDF escaneado con OCR"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Convertir PDF a imágenes
                images = convert_from_path(pdf_path, dpi=Config.OCR_DPI, output_folder=temp_dir, 
                                         fmt='png', paths_only=True)
                
                total_pages = len(images)
                all_text = ""
                legal_info = {}
                
                # Procesar páginas en paralelo
                tasks = []
                for i, image_path in enumerate(images, 1):
                    task = self.ocr_engine.process_page(image_path, language)
                    tasks.append((i, task))
                
                # Esperar resultados
                for i, task in tasks:
                    try:
                        page_text = await task
                        all_text += f"\n\n--- PÁGINA {i} ---\n\n{page_text}"
                        
                        # Extraer info legal de cada página
                        page_info = self.ocr_engine.extract_legal_info(page_text)
                        legal_info.update(page_info)
                        
                    except Exception as e:
                        logger.error(f"Page {i} processing error: {e}")
                        all_text += f"\n\n--- PÁGINA {i} (ERROR) ---\n\nError processing page: {str(e)}"
                
                # Guardar texto completo
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(all_text)
                
                return {
                    "success": True,
                    "text": all_text,
                    "legal_info": legal_info,
                    "pages": total_pages,
                    "method": "advanced_ocr"
                }
                
        except Exception as e:
            logger.error(f"Scanned PDF processing error: {e}")
            return {
                "success": False,
                "error": str(e),
                "pages": 0,
                "method": "failed"
            }

# ----------------------------
# Endpoints de la API
# ----------------------------
pdf_processor = PDFProcessor()

@app.post("/api/v1/ocr", response_model=OCRResponse)
async def process_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: str = Form(Config.OCR_LANG_DEFAULT),
    format: str = Form("json"),
    tables: bool = Form(False),
    legal_mode: bool = Form(True),
    current_user: User = Depends(get_current_user)
):
    """Procesar documento con OCR"""
    rate_limit(current_user.id, "ocr")
    
    # Validar archivo
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "Solo se permiten archivos PDF")
    
    # Crear job ID
    job_id = str(uuid.uuid4())
    filename = f"{job_id}_{file.filename}"
    file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
    output_path = os.path.join(Config.OUTPUT_FOLDER, f"{job_id}.txt")
    
    # Guardar archivo
    try:
        contents = await file.read()
        if len(contents) > Config.MAX_FILE_SIZE:
            raise HTTPException(413, "Archivo demasiado grande")
        
        with open(file_path, 'wb') as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(500, f"Error saving file: {str(e)}")
    
    # Guardar job en base de datos
    db = SessionLocal()
    try:
        job = OCRJob(
            id=job_id,
            filename=file.filename,
            language=language,
            status="processing",
            user_id=current_user.id
        )
        db.add(job)
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(500, f"Error creating job: {str(e)}")
    finally:
        db.close()
    
    # Procesar en background
    background_tasks.add_task(
        process_job_background,
        job_id, file_path, output_path, language, format, tables, legal_mode
    )
    
    return OCRResponse(
        success=True,
        job_id=job_id,
        status="processing",
        message="Documento en proceso",
        estimated_time=60.0  # Estimación inicial
    )

@app.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str, current_user: User = Depends(get_current_user)):
    """Obtener estado del job"""
    db = SessionLocal()
    try:
        job = db.query(OCRJob).filter(OCRJob.id == job_id, OCRJob.user_id == current_user.id).first()
        if not job:
            raise HTTPException(404, "Job no encontrado")
        
        result = None
        if job.status == "completed":
            output_path = os.path.join(Config.OUTPUT_FOLDER, f"{job_id}.txt")
            if os.path.exists(output_path):
                with open(output_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                result = {
                    "text": text,
                    "key_values": job.key_values or {},
                    "tables": job.tables or []
                }
        
        return JobStatusResponse(
            job_id=job.id,
            status=job.status,
            progress=min(100, int((job.pages_processed / job.total_pages * 100)) if job.total_pages > 0 else 0),
            filename=job.filename,
            pages_processed=job.pages_processed,
            total_pages=job.total_pages,
            processing_time=job.processing_time,
            result=result,
            error=job.error
        )
    finally:
        db.close()

@app.get("/api/v1/jobs/{job_id}/download")
async def download_result(job_id: str, current_user: User = Depends(get_current_user)):
    """Descargar resultado del OCR"""
    db = SessionLocal()
    try:
        job = db.query(OCRJob).filter(OCRJob.id == job_id, OCRJob.user_id == current_user.id).first()
        if not job or job.status != "completed":
            raise HTTPException(404, "Resultado no disponible")
        
        output_path = os.path.join(Config.OUTPUT_FOLDER, f"{job_id}.txt")
        if not os.path.exists(output_path):
            raise HTTPException(404, "Archivo de resultado no encontrado")
        
        return FileResponse(
            output_path,
            filename=f"ocr_result_{job.filename}.txt",
            media_type="text/plain"
        )
    finally:
        db.close()

@app.get("/metrics")
async def metrics():
    """Endpoint de métricas Prometheus"""
    return StreamingResponse(
        io.BytesIO(generate_latest()),
        media_type=CONTENT_TYPE_LATEST
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# ----------------------------
# Funciones de Background
# ----------------------------
async def process_job_background(job_id: str, file_path: str, output_path: str, 
                               language: str, format: str, tables: bool, legal_mode: bool):
    """Procesar job en background"""
    db = SessionLocal()
    try:
        start_time = datetime.now()
        
        # Procesar PDF
        result = await pdf_processor.process_pdf(
            file_path, output_path, language, tables, legal_mode
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Actualizar job
        job = db.query(OCRJob).filter(OCRJob.id == job_id).first()
        if job:
            job.status = "completed" if result["success"] else "failed"
            job.pages_processed = result.get("pages", 0)
            job.total_pages = result.get("pages", 0)
            job.processing_time = processing_time
            job.method = result.get("method", "")
            job.key_values = result.get("legal_info", {})
            job.error = result.get("error")
            
            db.commit()
        
        # Limpiar archivo temporal
        try:
            os.remove(file_path)
        except:
            pass
            
    except Exception as e:
        logger.error(f"Background processing error: {e}")
        if job:
            job.status = "failed"
            job.error = str(e)
            db.commit()
    finally:
        db.close()

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        timeout_keep_alive=300,
        log_level="info"
    )