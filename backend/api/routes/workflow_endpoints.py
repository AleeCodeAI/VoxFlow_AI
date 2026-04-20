from fastapi import APIRouter, File, UploadFile, HTTPException
import os
from tempfile import NamedTemporaryFile
import logging
from core.transcriber.transcriber import Transcriber
from core.preprocessor.preprocessor import Preprocessor
from pydantic_schemas import CombinedResponse

router = APIRouter()
logger = logging.getLogger(__name__)

transcriber = Transcriber()
preprocessor = Preprocessor()


@router.post("/transcribe-and-process/audio", response_model=CombinedResponse)
async def transcribe_and_process_audio(file: UploadFile = File(...)):
    try:
        logger.info(f"Starting combined workflow for: {file.filename}")

        suffix = os.path.splitext(file.filename)[1]
        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        transcription_obj, report = transcriber.transcribe(temp_path)
        os.unlink(temp_path)

        input_data = {
            "id": transcription_obj.id,
            "name": transcription_obj.name,
            "transcription": transcription_obj.transcription,
        }

        preprocessed_obj = preprocessor.preprocess(input_data)

        logger.info(f"Combined workflow completed for: {file.filename}")

        return CombinedResponse(
            status="success",
            message=f"Audio file '{file.filename}' transcribed and processed successfully",
            transcription=transcription_obj,
            preprocessed=preprocessed_obj,
        )

    except Exception as e:
        logger.error(f"Error in combined workflow: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Workflow failed",
                "detail": str(e),
            },
        )