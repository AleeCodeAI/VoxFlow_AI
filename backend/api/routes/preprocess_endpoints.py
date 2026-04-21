from fastapi import APIRouter, HTTPException
import logging
from core.preprocessor.preprocessor import Preprocessor
from pydantic_schemas import ProcessRequest, PreprocessingResponse

router = APIRouter()
logger = logging.getLogger(__name__)
preprocessor = Preprocessor()


@router.post("/preprocess", response_model=PreprocessingResponse)
async def process_transcription(request: ProcessRequest):
    """
    Endpoint for processing a transcription using LLM.

    - Accepts: Transcription object (id, name, transcription)
    - Returns: PreprocessedResult object with status
    """
    try:
        logger.info(f"Processing transcription ID: {request.id}")

        # Convert request to dict for preprocessor
        input_data = {
            "id": request.id,
            "name": request.name,
            "transcription": request.transcription,
        }

        # Process using your Preprocessor class
        preprocessed_obj = preprocessor.preprocess(input_data)

        logger.info(f"Processing completed for ID: {request.id}")

        return PreprocessingResponse(
            status="success",
            message=f"Transcription '{request.name}' processed successfully",
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