from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
import logging
from core.preprocessor.preprocessor import Preprocessor
from pydantic_schemas import ProcessRequest, PreprocessingResponse
from configs import MainSettings

settings = MainSettings()
limiter = Limiter(key_func=get_remote_address)

router = APIRouter()
logger = logging.getLogger(__name__)
preprocessor = Preprocessor()


@router.post("/process", response_model=PreprocessingResponse)
@limiter.limit(settings.PREPROCESS_RATE_LIMIT)
async def process_transcription(request: Request, input_data: ProcessRequest):
    """
    Endpoint for processing a transcription using LLM.

    - Accepts: Transcription object (id, name, transcription)
    - Returns: PreprocessedResult object with status
    """
    try:
        logger.info(f"Processing transcription ID: {input_data.id}")

        input_dict = {
            "id": input_data.id,
            "name": input_data.name,
            "transcription": input_data.transcription,
        }

        preprocessed_obj = preprocessor.preprocess(input_dict)
        logger.info(f"Processing completed for ID: {input_data.id}")

        return PreprocessingResponse(
            status="success",
            message=f"Transcription '{input_data.name}' processed successfully",
            data=preprocessed_obj,
        )

    except Exception as e:
        logger.error(f"Error processing transcription: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Processing failed",
                "detail": str(e),
            },
        )