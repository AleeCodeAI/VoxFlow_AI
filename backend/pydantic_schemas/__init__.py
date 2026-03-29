# ======================== SCHEMAS FOR CORE ============================
from .transcriber_schemas import Transcription, TranscriptionError
from .preprocessor_schemas import (PreprocessedResult, 
                                   LLMParsedResponse,
                                   PreprocessorError,
                                   LLMCallError,
                                   DatabaseError)

# ======================== SCHEMAS FOR EVALUATION ============================
from .preprocessor_evaluation_schemas.ai_judge_schemas import EvaluationError, Result, AIResult
from .preprocessor_evaluation_schemas.functional_correctness_schemas import PreprocessorEvaluationResult, TranscriptionInput

from .transcriber_evaluation_schemas.lexical_similarity_schemas import NormalizedObject, LexicalMetrics
from .transcriber_evaluation_schemas.functional_correctness_schemas import TranscriptionEvaluationResult, EvaluationSummary, ErrorMessage

# ======================== SCHEMAS FOR TOOLS ============================
from .tools_schemas.email_sender_schemas import Email
from .tools_schemas.text_extractor_schemas import TextExtraction, ProcessedData
from .tools_schemas.translator_schemas import TranslationOutput

if __name__ == "__main__":
    print("THESE ARE THE PYDANTIC SCHEMAS FOR THE ENTIRE PROJECT")
    print("==" * 30)

    sections = {
        "Transcriber Schemas": [
            Transcription, TranscriptionError,
        ],
        "Preprocessor Schemas": [
            PreprocessedResult, LLMParsedResponse,
            PreprocessorError, LLMCallError, DatabaseError,
        ],
        "Preprocessor Evaluation (AI Judge)": [
            EvaluationError, Result, AIResult,
        ],
        "Preprocessor Evaluation (Functional)": [
            PreprocessorEvaluationResult, TranscriptionInput,
        ],
        "Transcriber Evaluation": [
            NormalizedObject, LexicalMetrics,
            TranscriptionEvaluationResult, EvaluationSummary, ErrorMessage,
        ],
        "Tools Schemas": [
            Email, TextExtraction, ProcessedData, TranslationOutput,
        ],
    }

    for title, schemas in sections.items():
        print(f"\n{title}:")
        for schema in schemas:
            print(f"  • {schema.__name__}")