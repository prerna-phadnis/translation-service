from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import io
import logging
from PIL import Image
import pytesseract
from deep_translator import GoogleTranslator, MyMemoryTranslator
from deep_translator.exceptions import TranslationNotFound, LanguageNotSupportedException
import fitz

# ================================
# CONFIGURATION
# ================================

class Settings:
    APP_NAME = "Chinese to English Translation Service"
    APP_DESCRIPTION = "Translate Chinese text from images and PDFs to English using deep_translator"
    VERSION = "1.0.0"
    HOST = "0.0.0.0"
    PORT = 8000
    DEBUG = True
    
    # Translation settings - Fixed language codes for deep_translator
    SOURCE_LANGUAGE = "zh-CN"  # Chinese Simplified
    SOURCE_LANGUAGE_TRADITIONAL = "zh-TW"  # Chinese Traditional
    TARGET_LANGUAGE = "en"
    TESSERACT_CONFIG = "--oem 3 --psm 6 -l chi_sim+chi_tra"
    
    # File processing limits
    MAX_BATCH_SIZE = 10
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    PDF_ZOOM_FACTOR = 2.0

settings = Settings()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================
# DATA MODELS
# ================================

class TranslationResponse(BaseModel):
    success: bool
    original_text: str
    translated_text: str
    detected_language: str
    confidence: float = 0.0
    page_number: Optional[int] = None
    provider_used: Optional[str] = None

class MultiPageResponse(BaseModel):
    success: bool
    total_pages: int
    translations: List[TranslationResponse]


class TranslationService:
    def __init__(self):
        # Initialize translators with proper language codes for deep_translator
        try:
            self.primary_translator = GoogleTranslator(
                source=settings.SOURCE_LANGUAGE,  # zh-CN
                target=settings.TARGET_LANGUAGE   # en
            )
            logger.info("Primary translator (GoogleTranslator) initialized with zh-CN -> en")
        except Exception as e:
            logger.warning(f"Failed to initialize primary translator: {e}")
            self.primary_translator = None
        
        try:
            self.backup_translator = MyMemoryTranslator(
                source=settings.SOURCE_LANGUAGE,  # zh-CN
                target=settings.TARGET_LANGUAGE   # en
            )
            logger.info("Backup translator (MyMemoryTranslator) initialized with zh-CN -> en")
        except Exception as e:
            logger.warning(f"Failed to initialize backup translator: {e}")
            self.backup_translator = None
        
        # Additional translator for traditional Chinese
        try:
            self.traditional_translator = GoogleTranslator(
                source=settings.SOURCE_LANGUAGE_TRADITIONAL,  # zh-TW
                target=settings.TARGET_LANGUAGE  # en
            )
            logger.info("Traditional Chinese translator initialized with zh-TW -> en")
        except Exception as e:
            logger.warning(f"Failed to initialize traditional Chinese translator: {e}")
            self.traditional_translator = None
        
        # Auto-detect translator
        try:
            self.auto_translator = GoogleTranslator(
                source='auto',
                target=settings.TARGET_LANGUAGE
            )
            logger.info("Auto-detect translator initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize auto-detect translator: {e}")
            self.auto_translator = None
    
    def detect_language(self, text: str) -> Dict[str, Any]:
        """Detect if text contains Chinese characters and determine type"""
        try:
            if not text or not text.strip():
                return {"language": "unknown", "confidence": 0.0, "chinese_type": "none"}
            
            # Count different types of characters
            simplified_chars = 0
            traditional_chars = 0
            chinese_punctuation = sum(1 for char in text if char in '，。！？；：「」『』（）【】《》')
            
            # Common simplified vs traditional character indicators
            simplified_indicators = '个从众会学国'  # Common simplified chars
            traditional_indicators = '個從眾會學國'  # Their traditional counterparts
            
            for char in text:
                if '\u4e00' <= char <= '\u9fff':  # CJK range
                    if char in simplified_indicators:
                        simplified_chars += 2  # Weight more heavily
                    elif char in traditional_indicators:
                        traditional_chars += 2
                    else:
                        simplified_chars += 1  # Default to simplified for common chars
            
            total_chinese = simplified_chars + traditional_chars + chinese_punctuation
            total_chars = len([char for char in text if char.isalpha() or '\u4e00' <= char <= '\u9fff' or char in '，。！？；：'])
            
            if total_chars == 0:
                return {"language": "unknown", "confidence": 0.0, "chinese_type": "none"}
            
            chinese_ratio = total_chinese / total_chars
            
            # Determine Chinese type
            if chinese_ratio > 0.15:
                if traditional_chars > simplified_chars * 0.3:  # If traditional chars are significant
                    chinese_type = "traditional"
                    detected_lang = "zh-TW"
                else:
                    chinese_type = "simplified"
                    detected_lang = "zh-CN"
                    
                return {
                    "language": detected_lang,
                    "confidence": min(chinese_ratio * 1.3, 1.0),
                    "chinese_type": chinese_type
                }
            else:
                return {
                    "language": "en",
                    "confidence": max(0.3, 1.0 - chinese_ratio),
                    "chinese_type": "none"
                }
                
        except Exception as e:
            logger.warning(f"Language detection failed: {str(e)}")
            return {"language": "unknown", "confidence": 0.0, "chinese_type": "none"}
    
    def translate_with_fallback(self, text: str, detected_language: str = "zh-CN") -> Dict[str, str]:
        """Try translation with multiple translators based on detected language"""
        if not text or not text.strip():
            return {"text": "", "provider": "none"}
        
        # Choose appropriate translator based on detected language
        translators_to_try = []
        
        if detected_language == "zh-TW" and self.traditional_translator:
            translators_to_try.append(("TraditionalGoogleTranslator", self.traditional_translator))
        
        if self.primary_translator:
            translators_to_try.append(("GoogleTranslator", self.primary_translator))
            
        if self.auto_translator:
            translators_to_try.append(("AutoGoogleTranslator", self.auto_translator))
            
        if self.backup_translator:
            translators_to_try.append(("MyMemoryTranslator", self.backup_translator))
        
        # Try each translator
        for translator_name, translator in translators_to_try:
            try:
                result = translator.translate(text.strip())
                if result and result.strip() and result.strip() != text.strip():
                    logger.debug(f"Translation successful with {translator_name}")
                    return {"text": result, "provider": translator_name}
            except (TranslationNotFound, LanguageNotSupportedException) as e:
                logger.warning(f"{translator_name} failed: {str(e)}")
            except Exception as e:
                logger.warning(f"{translator_name} error: {str(e)}")
        
        # If all translators fail, return original text
        logger.warning("All translators failed, returning original text")
        return {"text": text, "provider": "fallback"}
    
    def translate_text(self, text: str) -> Dict[str, Any]:
        """Translate Chinese text to English using deep_translator"""
        try:
            if not text or not text.strip():
                return {
                    "translated_text": "",
                    "detected_language": "unknown",
                    "confidence": 0.0,
                    "provider_used": "none"
                }
            
            # Detect language and Chinese type
            detection = self.detect_language(text)
            
            # Translate with appropriate translator
            translation_result = self.translate_with_fallback(
                text, 
                detection.get("language", "zh-CN")
            )
            
            return {
                "translated_text": translation_result["text"],
                "detected_language": detection["language"],
                "confidence": detection["confidence"],
                "provider_used": translation_result["provider"]
            }
            
        except Exception as e:
            logger.error(f"Translation service failed: {str(e)}")
            return {
                "translated_text": text if text else "",
                "detected_language": "unknown",
                "confidence": 0.0,
                "provider_used": "error"
            }

# ================================
# OCR SERVICE
# ================================

class OCRService:
    def __init__(self):
        self.config = settings.TESSERACT_CONFIG
        logger.info("OCR service initialized with Tesseract")
    
    def extract_text_from_image(self, image: Image.Image) -> str:
        """Extract Chinese text from image using OCR"""
        try:
            text = pytesseract.image_to_string(image, config=self.config)
            return text.strip()
        except Exception as e:
            logger.error(f"OCR extraction failed: {str(e)}")
            raise Exception(f"OCR processing failed: {str(e)}")
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results"""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise Exception(f"Image preprocessing failed: {str(e)}")

# ================================
# PDF SERVICE
# ================================

class PDFService:
    def __init__(self):
        self.zoom_factor = settings.PDF_ZOOM_FACTOR
        logger.info("PDF service initialized with PyMuPDF")
    
    def pdf_to_images(self, pdf_bytes: bytes) -> List[Image.Image]:
        """Convert PDF pages to images"""
        try:
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            images = []
            
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                # Convert page to image with high resolution for better OCR
                mat = fitz.Matrix(self.zoom_factor, self.zoom_factor)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("ppm")
                img = Image.open(io.BytesIO(img_data))
                images.append(img)
            
            pdf_document.close()
            logger.info(f"Successfully converted PDF to {len(images)} images")
            return images
            
        except Exception as e:
            logger.error(f"PDF processing failed: {str(e)}")
            raise Exception(f"PDF processing failed: {str(e)}")

# ================================
# FILE PROCESSOR
# ================================

class FileProcessor:
    def __init__(self):
        self.ocr_service = OCRService()
        self.translation_service = TranslationService()
        self.pdf_service = PDFService()
        logger.info("File processor initialized with all services")
    
    async def process_image(self, file: UploadFile) -> TranslationResponse:
        """Process single image file"""
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            image = self.ocr_service.preprocess_image(image)
            
            logger.info(f"Processing image: {file.filename}")
            original_text = self.ocr_service.extract_text_from_image(image)
            
            if not original_text:
                return TranslationResponse(
                    success=True,
                    original_text="",
                    translated_text="No text detected in image",
                    detected_language="unknown",
                    confidence=0.0,
                    provider_used="none"
                )
            
            # Translate text
            translation_result = self.translation_service.translate_text(original_text)
            
            return TranslationResponse(
                success=True,
                original_text=original_text,
                translated_text=translation_result["translated_text"],
                detected_language=translation_result["detected_language"],
                confidence=translation_result["confidence"],
                provider_used=translation_result["provider_used"]
            )
            
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            raise Exception(f"Image processing failed: {str(e)}")
    
    async def process_pdf(self, file: UploadFile) -> MultiPageResponse:
        """Process PDF file"""
        try:
            contents = await file.read()
            logger.info(f"Processing PDF: {file.filename}")
            
            # Convert PDF to images
            images = self.pdf_service.pdf_to_images(contents)
            translations = []
            
            # Process each page
            for page_num, image in enumerate(images, 1):
                logger.info(f"Processing page {page_num}/{len(images)}")
                
                original_text = self.ocr_service.extract_text_from_image(image)
                
                if original_text:
                    translation_result = self.translation_service.translate_text(original_text)
                    
                    translations.append(TranslationResponse(
                        success=True,
                        original_text=original_text,
                        translated_text=translation_result["translated_text"],
                        detected_language=translation_result["detected_language"],
                        confidence=translation_result["confidence"],
                        page_number=page_num,
                        provider_used=translation_result["provider_used"]
                    ))
                else:
                    translations.append(TranslationResponse(
                        success=True,
                        original_text="",
                        translated_text="No text detected on this page",
                        detected_language="unknown",
                        confidence=0.0,
                        page_number=page_num,
                        provider_used="none"
                    ))
            
            return MultiPageResponse(
                success=True,
                total_pages=len(images),
                translations=translations
            )
            
        except Exception as e:
            logger.error(f"PDF processing failed: {str(e)}")
            raise Exception(f"PDF processing failed: {str(e)}")

# ================================
# FASTAPI APPLICATION
# ================================

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.VERSION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize file processor
file_processor = FileProcessor()

# ================================
# API ENDPOINTS
# ================================

@app.post("/api/v1/translate/image", response_model=TranslationResponse)
async def translate_image(file: UploadFile = File(...)):
    """Translate Chinese text from uploaded image to English"""
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        logger.info(f"Received image translation request: {file.filename}")
        result = await file_processor.process_image(file)
        logger.info(f"Image translation completed successfully")
        return result
    except Exception as e:
        logger.error(f"Image translation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.post("/api/v1/translate/pdf", response_model=MultiPageResponse)
async def translate_pdf(file: UploadFile = File(...)):
    """Translate Chinese text from uploaded PDF to English"""
    
    # Validate file type
    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        logger.info(f"Received PDF translation request: {file.filename}")
        result = await file_processor.process_pdf(file)
        logger.info(f"PDF translation completed successfully")
        return result
    except Exception as e:
        logger.error(f"PDF translation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


if __name__ == "__main__":
    logger.info(f"Starting {settings.APP_NAME} v{settings.VERSION}")
    logger.info(f"Server will run on {settings.HOST}:{settings.PORT}")
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )