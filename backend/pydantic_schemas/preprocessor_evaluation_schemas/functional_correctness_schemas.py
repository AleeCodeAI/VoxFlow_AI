from pydantic import BaseModel, Field
from datetime import datetime

class PreprocessorEvaluationResult(BaseModel):
    """
    Stores evaluation metrics for a single preprocessing execution.
    """
    id: str = Field(description="Unique identifier for the evaluation result")
    file_name: str = Field(description="Name of the file being evaluated")
    chunk_completeness: bool = Field(description="Whether all chunks were processed completely")
    llm_retries: int = Field(description="Number of retries made by the LLM during processing")
    output_existence: bool = Field(description="Indicates if the output file exists after processing")
    session_integrity: bool = Field(description="Indicates if the session data remains intact after processing")
    timestamp: datetime = Field(description="Timestamp of when the evaluation was performed")

class TranscriptionInput(BaseModel):
    """
    Represents a single transcription from the input JSONL file.
    """
    id: str
    name: str
    transcription: str
    timestamp: str
