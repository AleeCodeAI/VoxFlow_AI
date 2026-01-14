from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
from tempfile import NamedTemporaryFile
import logging
from contextlib import asynccontextmanager
from core.transcriber import Transcriber, Transcription
from core.preprocessor import Preprocessor, PreprocessedResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the core components (singleton pattern)
transcriber = Transcriber()
preprocessor = Preprocessor()

# ============================================================================
# LIFESPAN EVENTS
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup and cleanup on shutdown"""
    # Startup
    logger.info("✅ Audio Preprocessor API is starting up...")
    logger.info(f"✅ Transcriber loaded with model: base")
    logger.info(f"✅ Preprocessor initialized")
    yield
    # Shutdown
    logger.info("👋 Shutting down Audio Preprocessor API...")

# ============================================================================
# INITIALIZE FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Audio Preprocessor API",
    description="API for transcribing and preprocessing audio files",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class DirectTextInput(BaseModel):
    """Model for when user pastes text directly"""
    name: str
    transcription: str

class ProcessRequest(BaseModel):
    """Model for processing a transcription"""
    id: str
    name: str
    transcription: str

class TranscriptionResponse(BaseModel):
    """Standardized response for transcription endpoints"""
    status: str
    message: str
    data: Transcription

class PreprocessingResponse(BaseModel):
    """Standardized response for preprocessing endpoints"""
    status: str
    message: str
    data: PreprocessedResult

class CombinedResponse(BaseModel):
    """Response for combined workflow"""
    status: str
    message: str
    transcription: Transcription
    preprocessed: PreprocessedResult

class ErrorResponse(BaseModel):
    """Standardized error response"""
    status: str
    message: str
    detail: Optional[str] = None

# ============================================================================
# NEW TOOL REQUEST/RESPONSE MODELS
# ============================================================================

class EmailRequest(BaseModel):
    """Model for email sending request"""
    to: str
    subject: str
    processed_data: str
    user_message: str
    sender: str

class EmailResponse(BaseModel):
    """Response for email sending"""
    status: str
    message: str
    email: str

class TextExtractionRequest(BaseModel):
    """Model for text extraction request"""
    processed_data: str

class TextExtractionData(BaseModel):
    """Extracted keywords and keypoints"""
    keywords: List[str]
    keypoints: List[str]

class TextExtractionResponse(BaseModel):
    """Response for text extraction"""
    status: str
    message: str
    data: TextExtractionData

class TranslationRequest(BaseModel):
    """Model for translation request"""
    language: str
    processed_data: str

class TranslationResponse(BaseModel):
    """Response for translation"""
    status: str
    message: str
    translated_data: str

# ============================================================================
# HEALTH CHECK ENDPOINT
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "✅ success",
        "service": "Audio Preprocessor API",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "✅ success",
        "transcriber": "✅ ready",
        "preprocessor": "✅ ready"
    }

# ============================================================================
# TRANSCRIPTION ENDPOINTS
# ============================================================================

@app.post("/transcribe/audio", response_model=TranscriptionResponse)
async def transcribe_audio_file(file: UploadFile = File(...)):
    """
    Endpoint for uploading and transcribing audio files.
    
    - Accepts: Audio files (mp3, wav, m4a, etc.)
    - Returns: Transcription object with status
    """
    try:
        logger.info(f"Received audio file: {file.filename}")
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Save uploaded file to temporary location
        suffix = os.path.splitext(file.filename)[1]
        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        logger.info(f"Saved to temporary file: {temp_path}")
        
        # Transcribe using your Transcriber class
        transcription_obj = transcriber.transcribe(temp_path)
        
        # Clean up temporary file
        os.unlink(temp_path)
        logger.info(f"✅ Transcription completed for: {file.filename}")
        
        return TranscriptionResponse(
            status="✅ success",
            message=f"Audio file '{file.filename}' transcribed successfully",
            data=transcription_obj
        )
        
    except Exception as e:
        logger.error(f"❌ Error transcribing audio: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail={
                "status": "❌ error",
                "message": "Transcription failed",
                "detail": str(e)
            }
        )

@app.post("/transcribe/text", response_model=TranscriptionResponse)
async def transcribe_direct_text(input_data: DirectTextInput):
    """
    Endpoint for directly pasted text (no audio transcription needed).
    
    - Accepts: JSON with 'name' and 'transcription' fields
    - Returns: Transcription object with status
    """
    try:
        logger.info(f"Received direct text input: {input_data.name}")
        
        from datetime import datetime
        import uuid
        
        # Create transcription object directly
        transcription_obj = Transcription(
            id=str(uuid.uuid4()),
            name=input_data.name,
            transcription=input_data.transcription,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Save to database
        db_path = r"D:\Projects\audio_preprocessor\backend\databases"
        jsonl_file = os.path.join(db_path, "transcriptions.jsonl")
        os.makedirs(db_path, exist_ok=True)
        
        with open(jsonl_file, 'a', encoding='utf-8') as f:
            f.write(transcription_obj.model_dump_json() + '\n')
            f.flush()
            os.fsync(f.fileno())
        
        logger.info(f"✅ Direct text saved as transcription: {transcription_obj.id}")
        
        return TranscriptionResponse(
            status="✅ success",
            message=f"Text '{input_data.name}' saved successfully",
            data=transcription_obj
        )
        
    except Exception as e:
        logger.error(f"❌ Error saving direct text: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "❌ error",
                "message": "Failed to save text",
                "detail": str(e)
            }
        )

# ============================================================================
# PREPROCESSING ENDPOINT
# ============================================================================

@app.post("/process", response_model=PreprocessingResponse)
async def process_transcription(request: ProcessRequest):
    """
    Endpoint for processing a transcription using LLM.
    
    - Accepts: Transcription object (id, name, transcription)
    - Returns: PreprocessedResult object with status
    """
    try:
        logger.info(f"Processing transcription ID: {request.id}")
        
        # Convert request to dict for preprocessor
        input_data = {
            "id": request.id,
            "name": request.name,
            "transcription": request.transcription
        }
        
        # Process using your Preprocessor class
        preprocessed_obj = preprocessor.preprocess(input_data)
        
        logger.info(f"✅ Processing completed for ID: {request.id}")
        
        return PreprocessingResponse(
            status="✅ success",
            message=f"Transcription '{request.name}' processed successfully",
            data=preprocessed_obj
        )
        
    except Exception as e:
        logger.error(f"❌ Error processing transcription: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "❌ error",
                "message": "Processing failed",
                "detail": str(e)
            }
        )

# ============================================================================
# NEW TOOL ENDPOINTS
# ============================================================================

@app.post("/send-email", response_model=EmailResponse)
async def send_email(request: EmailRequest):
    """
    Endpoint for sending emails via n8n webhook.
    
    - Accepts: Email details (to, subject, processed_data, user_message, sender)
    - Returns: Email sending status
    """
    try:
        logger.info(f"Sending email to: {request.to}")
        
        # Import the EmailSender class
        from tools.email_sender import EmailSender, Email
        
        # Create email sender instance
        email_sender = EmailSender()
        
        # Create email object
        email_data = Email(
            to=request.to,
            subject=request.subject,
            processed_data=request.processed_data,
            user_message=request.user_message,
            sender=request.sender
        )
        
        # Send email
        result = email_sender.send_email(email_data)
        
        logger.info(f"✅ Email sent successfully to: {request.to}")
        
        return EmailResponse(
            status="✅ success",
            message=f"Email sent successfully to {request.to}",
            email=request.to
        )
        
    except Exception as e:
        logger.error(f"❌ Error sending email: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "❌ error",
                "message": "Email sending failed",
                "detail": str(e)
            }
        )

@app.post("/extract-text", response_model=TextExtractionResponse)
async def extract_text(request: TextExtractionRequest):
    """
    Endpoint for extracting keywords and keypoints from processed data.
    
    - Accepts: Processed data text
    - Returns: Keywords and keypoints
    """
    try:
        logger.info("Extracting keywords and keypoints from processed data")
        
        # Import the TextExtracter class
        from tools.text_extracter import TextExtracter, ProcessedData
        
        # Create text extracter instance
        text_extracter = TextExtracter()
        
        # Create processed data object
        processed_data = ProcessedData(processed_data=request.processed_data)
        
        # Extract keywords and keypoints
        extraction_result = text_extracter.extract(processed_data.processed_data)
        
        logger.info(f"✅ Text extraction completed")
        
        return TextExtractionResponse(
            status="✅ success",
            message="Text extraction completed successfully",
            data=TextExtractionData(
                keywords=extraction_result.keywords,
                keypoints=extraction_result.keypoints
            )
        )
        
    except Exception as e:
        logger.error(f"❌ Error extracting text: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "❌ error",
                "message": "Text extraction failed",
                "detail": str(e)
            }
        )

@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """
    Endpoint for translating processed data to another language.
    
    - Accepts: Language code and processed data
    - Returns: Translated text
    """
    try:
        logger.info(f"Translating text to language: {request.language}")
        
        # Import the Translate class
        from tools.translator import Translate
        
        # Create translator instance
        translator = Translate()
        
        # Translate text
        translation_result = translator.translate(
            language=request.language,
            data=request.processed_data
        )
        
        logger.info(f"✅ Translation completed")
        
        return TranslationResponse(
            status="✅ success",
            message=f"Text translated successfully to {request.language}",
            translated_data=translation_result.translated_data
        )
        
    except Exception as e:
        logger.error(f"❌ Error translating text: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "❌ error",
                "message": "Translation failed",
                "detail": str(e)
            }
        )

# ============================================================================
# COMBINED WORKFLOW ENDPOINT (Optional - for convenience)
# ============================================================================

@app.post("/transcribe-and-process/audio", response_model=CombinedResponse)
async def transcribe_and_process_audio(file: UploadFile = File(...)):
    """
    Combined endpoint: Upload audio → Transcribe → Process
    Returns both transcription and preprocessed result with status.
    """
    try:
        # Step 1: Transcribe
        logger.info(f"Starting combined workflow for: {file.filename}")
        
        suffix = os.path.splitext(file.filename)[1]
        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        transcription_obj = transcriber.transcribe(temp_path)
        os.unlink(temp_path)
        
        # Step 2: Process
        input_data = {
            "id": transcription_obj.id,
            "name": transcription_obj.name,
            "transcription": transcription_obj.transcription
        }
        
        preprocessed_obj = preprocessor.preprocess(input_data)
        
        logger.info(f"✅ Combined workflow completed for: {file.filename}")
        
        return CombinedResponse(
            status="✅ success",
            message=f"Audio file '{file.filename}' transcribed and processed successfully",
            transcription=transcription_obj,
            preprocessed=preprocessed_obj
        )
        
    except Exception as e:
        logger.error(f"❌ Error in combined workflow: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "❌ error",
                "message": "Workflow failed",
                "detail": str(e)
            }
        )

# ============================================================================
# DATABASE QUERY ENDPOINTS (Optional - for retrieving saved data)
# ============================================================================

@app.get("/transcriptions/{transcription_id}")
async def get_transcription(transcription_id: str):
    """Retrieve a specific transcription by ID"""
    try:
        db_path = r"D:\Projects\audio_preprocessor\backend\databases"
        jsonl_file = os.path.join(db_path, "transcriptions.jsonl")
        
        if not os.path.exists(jsonl_file):
            raise HTTPException(
                status_code=404,
                detail={
                    "status": "❌ error",
                    "message": "No transcriptions found"
                }
            )
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                import json
                obj = json.loads(line)
                if obj.get('id') == transcription_id:
                    return {
                        "status": "✅ success",
                        "message": "Transcription retrieved successfully",
                        "data": obj
                    }
        
        raise HTTPException(
            status_code=404,
            detail={
                "status": "❌ error",
                "message": "Transcription not found"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "❌ error",
                "message": "Error retrieving transcription",
                "detail": str(e)
            }
        )

@app.get("/preprocessings/{preprocessing_id}")
async def get_preprocessing(preprocessing_id: str):
    """Retrieve a specific preprocessed result by ID"""
    try:
        db_path = r"D:\Projects\audio_preprocessor\backend\databases"
        jsonl_file = os.path.join(db_path, "preprocessings.jsonl")
        
        if not os.path.exists(jsonl_file):
            raise HTTPException(
                status_code=404,
                detail={
                    "status": "❌ error",
                    "message": "No preprocessings found"
                }
            )
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                import json
                obj = json.loads(line)
                if obj.get('id') == preprocessing_id:
                    return {
                        "status": "✅ success",
                        "message": "Preprocessing retrieved successfully",
                        "data": obj
                    }
        
        raise HTTPException(
            status_code=404,
            detail={
                "status": "❌ error",
                "message": "Preprocessing not found"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "❌ error",
                "message": "Error retrieving preprocessing",
                "detail": str(e)
            }
        )

# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  
        log_level="info"
    )