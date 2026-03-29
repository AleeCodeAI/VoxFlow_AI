from pydantic import BaseModel, Field

class TranscriptionError(Exception):
    pass

class Transcription(BaseModel):
    """Database model for storing audio transcription results."""
    id: str = Field(description="Unique identifier for the transcription")
    name: str = Field(description="Name of the audio file transcribed")
    transcription: str = Field(description="The full transcription of the audio file")
    timestamp: str = Field(description="Time of transcription")
