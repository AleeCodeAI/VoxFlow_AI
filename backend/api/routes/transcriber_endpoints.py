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
    """
    Endpoint for directly pasted text (no audio transcription needed).

    - Accepts: JSON with 'name' and 'transcription' fields
    - Returns: Transcription object with status
    """
    try:
        logger.info(f"Received direct text input: {input_data.name}")

        from datetime import datetime
        import uuid

        # Create transcription object directly
        transcription_obj = Transcription(
            id=str(uuid.uuid4()),
            name=input_data.name,
            transcription=input_data.transcription,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        # Save to database
        db_path = os.getenv(
                    "DATABASE_PATH",
                     os.path.join(os.path.dirname(__file__), "..", "..", "databases"),
)
        jsonl_file = os.path.join(db_path, "transcriptions.jsonl")
        os.makedirs(db_path, exist_ok=True)

        with open(jsonl_file, "a", encoding="utf-8") as f:
            f.write(transcription_obj.model_dump_json() + "\n")
            f.flush()
            os.fsync(f.fileno())

        logger.info(f"Direct text saved as transcription: {transcription_obj.id}")

        return TranscriptionResponse(
            status="success",
            message=f"Text '{input_data.name}' saved successfully",
            data=transcription_obj,
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