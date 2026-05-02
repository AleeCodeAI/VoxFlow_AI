from datetime import datetime
import uuid
from fastapi import APIRouter, File, UploadFile, HTTPException
import os
from tempfile import NamedTemporaryFile
import logging
from pydantic_schemas import (
    DirectTextInput,
    TranscriptionResponse,
    Transcription,
)
from core.transcriber.transcriber import Transcriber

from databases.database import get_session
from databases.transcriber_repository import TranscriptionRepository

transcription_repo = TranscriptionRepository()

router = APIRouter()
logger = logging.getLogger(__name__)
transcriber = Transcriber()


@router.post("/transcribe/audio", response_model=TranscriptionResponse)
async def transcribe_audio_file(file: UploadFile = File(...)):
    """
    Endpoint for uploading and transcribing audio files.

    - Accepts: Audio files (mp3, wav, m4a, etc.)
    - Returns: Transcription object with status
    """
    try:
        logger.info(f"Received audio file: {file.filename}")

        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        # Save uploaded file to temporary location
        suffix = os.path.splitext(file.filename)[1]
        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        logger.info(f"Saved to temporary file: {temp_path}")

        # Transcribe using your Transcriber class
        transcription_obj, report = transcriber.transcribe(temp_path)

        # Clean up temporary file
        os.unlink(temp_path)
        logger.info(f"Transcription completed for: {file.filename}")

        return TranscriptionResponse(
            status="success",
            message=f"Audio file '{file.filename}' transcribed successfully",
            data=transcription_obj,
        )

    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Transcription failed",
                "detail": str(e),
            },
        )


@router.post("/transcribe/text", response_model=TranscriptionResponse)
async def transcribe_direct_text(input_data: DirectTextInput):
    try:
        logger.info(f"Received direct text input: {input_data.name}")

        transcription_obj = Transcription(
            id=str(uuid.uuid4()),
            name=input_data.name,
            transcription=input_data.transcription,
            timestamp=datetime.now(),
        )

        for session in get_session():
            result = transcription_repo.save(
                session=session,
                session_id=transcription_obj.id,
                audio_name=transcription_obj.name,
                transcription_text=transcription_obj.transcription,
            )

        logger.info(f"Direct text saved as transcription: {result.id}")

        return TranscriptionResponse(
            status="success",
            message=f"Text '{input_data.name}' saved successfully",
            data=result,
        )

    except Exception as e:
        logger.error(f"Error saving direct text: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Failed to save text",
                "detail": str(e),
            },
        )