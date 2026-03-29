from pydantic import BaseModel
from pydantic_schemas.transcriber_schemas import Transcription
from pydantic_schemas.preprocessor_schemas import PreprocessedResult
from typing import List, Optional
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