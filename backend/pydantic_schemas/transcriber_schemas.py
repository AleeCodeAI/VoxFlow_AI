from pydantic import BaseModel, Field
from typing import List, Optional


class TranscriptionError(Exception):
    pass


class Transcription(BaseModel):
    """Database model for storing audio transcription results."""

    id: str = Field(description="Unique identifier for the transcription")
    name: str = Field(description="Name of the audio file transcribed")
    transcription: str = Field(description="The full transcription of the audio file")
    timestamp: str = Field(description="Time of transcription")


class Retries(BaseModel):
    chunk_no: int = Field(description="The chunk number that failed to transcribe")
    retries: int = Field(description="Number of retries attempted for this chunk")
    success: bool = Field(
        description="Whether the transcription was successful after retries"
    )


class TranscriptionReport(BaseModel):
    retries: List[Retries] = Field(
        description="List of retries for each chunk that failed to transcribe"
    )
    total_chunks: int = Field(
        description="Total number of chunks that were supposed to be transcribed"
    )
    chunks_completed: int = Field(
        description="Total number of chunks that were successfully transcribed"
    )
    total_time_ms: Optional[int] = Field(
        default=None, description="Total pipeline execution time in milliseconds"
    )
