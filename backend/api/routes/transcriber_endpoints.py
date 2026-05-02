import json
import uuid
from datetime import datetime
from tempfile import NamedTemporaryFile
import os
import logging

from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic_schemas import DirectTextInput, TranscriptionResponse, Transcription
from core.transcriber.transcriber import Transcriber
from core.transcriber.transcription_cache import TranscriptionCache
from databases.database import get_session
from databases.transcriber_repository import TranscriptionRepository

router = APIRouter()
logger = logging.getLogger(__name__)
transcriber = Transcriber()
transcription_repo = TranscriptionRepository()
cache = TranscriptionCache()


@router.post("/transcribe/audio", response_model=TranscriptionResponse)
async def transcribe_audio_file(file: UploadFile = File(...)):
    """
    Endpoint for uploading and transcribing audio files.
    Checks Redis cache first using SHA256 hash of file bytes.
    On cache miss, transcribes and caches the full transcription object.

    - Accepts: Audio files (mp3, wav, m4a, etc.)
    - Returns: Transcription object with status
    """
    try:
        logger.info(f"Received audio file: {file.filename}")

        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        content = await file.read()
        file_hash = cache.compute_hash(content)

        cached = cache.get(file_hash)
        if cached:
            logger.info(f"Cache hit for file: {file.filename}")
            transcription_obj = Transcription(**json.loads(cached))
            return TranscriptionResponse(
                status="success",
                message=f"Audio file '{file.filename}' retrieved from cache",
                data=transcription_obj,
            )

        logger.info(f"Cache miss for file: {file.filename}, transcribing...")

        suffix = os.path.splitext(file.filename)[1]
        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        transcription_obj, report = transcriber.transcribe(temp_path)
        os.unlink(temp_path)

        cache.set(file_hash, transcription_obj.model_dump_json())

        logger.info(f"Transcription completed and cached for: {file.filename}")

        return TranscriptionResponse(
            status="success",
            message=f"Audio file '{file.filename}' transcribed successfully",
            data=transcription_obj,
        )

    except HTTPException:
        raise
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

        for session in get_session():
            result = transcription_repo.save(
                session=session,
                session_id=str(uuid.uuid4()),
                audio_name=input_data.name,
                transcription_text=input_data.transcription,
            )

        logger.info(f"Direct text saved as transcription: {result.id}")

        return TranscriptionResponse(
            status="success",
            message=f"Text '{input_data.name}' saved successfully",
            data=result,
        )

    except HTTPException:
        raise
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