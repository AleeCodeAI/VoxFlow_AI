from pydantic import BaseModel, Field
from typing import List

class TextExtraction(BaseModel):
    keywords: List[str] = Field(
        description="A list of specific keywords found in the text, specifically focusing on terms related to the Rajya Sabha (e.g., Chairman, MP, Bill, Session).")
    keypoints: List[str] = Field(
        description="A list of bullet points summarizing the main actions or discussions involving the Rajya Sabha within this data.")

class ProcessedData(BaseModel):
    processed_data: str = Field(description="the processed data as input to TextAnalyzer")