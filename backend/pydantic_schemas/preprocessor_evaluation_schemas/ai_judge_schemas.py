from pydantic import BaseModel, Field
from datetime import datetime

# =========================== Error Handling Schemas ===========================
class EvaluationError(BaseModel):
    """Custom exception model for evaluation errors"""
    error_type: str = Field(description="Type of error that occurred")
    message: str = Field(description="Detailed error message")
    timestamp: str = Field(description="When the error occurred")
    context: dict = Field(default_factory=dict, description="Additional context about the error")

# =========================== Evaluation Result Schemas ===========================
class Result(BaseModel):
    """Complete evaluation result with metadata"""
    id: str = Field(description="Unique identifier for the evaluation result")
    file_name: str = Field(description="Name of the file being evaluated")
    meaning_preservation: str = Field(description="Score for meaning preservation: HIGH | MODERATE | LOW")
    information_loss: str = Field(description="The amount of information lost during preprocessing: HIGH | MODERATE | LOW")
    preprocessing_quality: str = Field(description="How well the preprocessing was done: GOLDEN | ACCEPTABLE | POOR")
    hallucination: str = Field(description="How much AI hallucinated while preprocessing: HIGH | MODERATE | LOW")
    confidence: float = Field(description="Confidence level of AI in the Output")
    reasoning: str = Field(description="Detailed reasoning behind the values given")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class AIResult(BaseModel):
    """AI evaluation result without metadata"""
    meaning_preservation: str = Field(description="Score for meaning preservation: HIGH | MODERATE | LOW")
    information_loss: str = Field(description="The amount of information lost during preprocessing: HIGH | MODERATE | LOW")
    preprocessing_quality: str = Field(description="How well the preprocessing was done: GOLDEN | ACCEPTABLE | POOR")
    hallucination: str = Field(description="How much AI hallucinated while preprocessing: HIGH | MODERATE | LOW")
    confidence: float = Field(description="Confidence level of AI in the Output")
    reasoning: str = Field(description="Detailed reasoning behind the values given")