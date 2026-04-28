from pydantic import BaseModel, Field
from datetime import datetime

# ====================== EXCEPTIONS ======================
class PreprocessorError(Exception):
    """Base exception for preprocessor errors"""
    pass

class LLMCallError(PreprocessorError):
    """Raised when LLM API call fails"""
    pass

class DatabaseError(PreprocessorError):
    """Raised when database operations fail"""
    pass

# ====================== SCHEMAS ======================
class PreprocessedResult(BaseModel):
    """Database model for storing preprocessed transcription results."""
    id: str = Field(description="Matches the original transcription ID")
    name: str = Field(description="Original audio filename")
    preprocessed_transcription: str = Field(description="The cleaned text produced by LLM")
    timestamp: datetime = Field(description="Time of preprocessing")

class PreprocessorReport(BaseModel):
    chunk_count: int = Field(default=0, description="Number of chunks created from the original transcription")
    chunks_processed: int = Field(default=0, description="Number of chunks successfully processed")
    llm_retries: int = Field(default=0, description="Number of times LLM calls were retried")
class LLMParsedResponse(BaseModel):
    """Response schema for structured output from LLM."""
    preprocessed_transcription: str = Field(description="The cleaned text")