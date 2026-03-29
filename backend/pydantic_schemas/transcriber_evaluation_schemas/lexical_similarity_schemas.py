from pydantic import BaseModel, Field

class NormalizedObject(BaseModel):
    """
    Represents a normalized transcription-reference pair.
    
    This model stores both the transcription and its reference text after
    normalization (lowercase, punctuation removal, etc.) for evaluation.
    """
    id: str = Field(description="Unique identifier for the transcription")
    file_name: str = Field(description="Name of the audio file")
    transcription: str = Field(description="Normalized transcription text")
    reference: str = Field(description="Normalized reference text")


class LexicalMetrics(BaseModel):
    """
    Stores lexical evaluation metrics for a transcription.
    
    Contains various metrics like WER, CER, and n-gram similarity along with
    a quality label to assess transcription accuracy.
    """
    id: str = Field(description="Unique identifier for the transcription evaluation")
    file_name: str = Field(description="Name of the audio file evaluated")
    wer: float = Field(description="Word Error Rate")
    cer: float = Field(description="Character Error Rate")
    ngram: float = Field(description="N-gram Similarity Score")
    quality_label: str = Field(description="Quality Label: OK / Acceptable / Bad")
    timestamp: str = Field(description="Timestamp of the evaluation")