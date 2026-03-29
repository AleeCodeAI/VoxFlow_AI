from pydantic import BaseModel, Field
from datetime import datetime

# ============================= Error Message Schema =============================
class ErrorMessage(BaseModel):
    """Represents an error encountered during transcription"""
    id: str = Field(description="Unique identifier for the transcription task")
    file_name: str = Field(description="Name of the audio file")
    error_message: str = Field(description="Detailed error message")
    timestamp: datetime = Field(description="Timestamp of when the error occurred")

# ============================= Transcription Evaluation Result Schema =============================
class TranscriptionEvaluationResult(BaseModel):
    """Evaluation result for a single test file"""
    id: str = Field(description="Unique identifier for the transcription task")
    file_name: str = Field(description="Name of the audio file")
    expected_valid: bool = Field(description="Whether the file was expected to be valid")
    input_validation_passed: bool = Field(description="Whether input validation behaved correctly")
    transcription_completed: bool = Field(description="Whether transcription completed successfully")
    output_saved: bool = Field(description="Whether output was saved to database")
    all_chunks_processed: bool = Field(description="Whether all audio chunks were processed")
    retry_count: int = Field(description="Number of retries attempted")
    errors: list[ErrorMessage] = Field(description="List of errors encountered")
    
    @property
    def success(self):
        """
        Overall success indicator
        For invalid files: success means correctly rejected
        For valid files: success means fully processed
        """
        if not self.expected_valid:
            return self.input_validation_passed
        
        return (self.input_validation_passed and 
                self.transcription_completed and 
                self.output_saved and 
                self.all_chunks_processed)
    
    @property
    def is_expected_rejection(self):
        """Invalid file that was correctly rejected"""
        return not self.expected_valid and self.input_validation_passed
    
    @property
    def is_unexpected_failure(self):
        """
        Something went wrong that shouldn't have
        Either invalid file accepted OR valid file failed processing
        """
        if not self.expected_valid:
            return not self.input_validation_passed
        else:
            return not self.success


class EvaluationSummary(BaseModel):
    """Aggregate metrics across all test cases"""
    total_files: int
    valid_files_count: int
    invalid_files_count: int
    
    overall_success_rate: float
    valid_files_success_rate: float
    invalid_files_rejection_rate: float
    
    input_validation_accuracy: float
    completion_rate: float
    output_save_rate: float
    chunk_processing_rate: float
    average_retries: float
    
    unexpected_failures: int
    expected_rejections: int
    total_errors: int
    timestamp: datetime