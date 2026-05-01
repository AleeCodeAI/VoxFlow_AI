from fastapi import APIRouter, HTTPException
from databases.database import get_session
from databases.transcriber_repository import TranscriptionRepository
from databases.preprocessor_repository import PreprocessingRepository

router = APIRouter()

transcription_repo = TranscriptionRepository()
preprocessing_repo = PreprocessingRepository()


@router.get("/transcriptions/{transcription_id}")
async def get_transcription(transcription_id: str):
    try:
        for session in get_session():
            result = transcription_repo.get(session, transcription_id)

        if not result:
            raise HTTPException(
                status_code=404,
                detail={"status": "error", "message": "Transcription not found"},
            )

        return {
            "status": "success",
            "message": "Transcription retrieved successfully",
            "data": result.model_dump(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Error retrieving transcription",
                "detail": str(e),
            },
        )


@router.get("/preprocessings/{preprocessing_id}")
async def get_preprocessing(preprocessing_id: str):
    try:
        for session in get_session():
            result = preprocessing_repo.get(session, preprocessing_id)

        if not result:
            raise HTTPException(
                status_code=404,
                detail={"status": "error", "message": "Preprocessing not found"},
            )

        return {
            "status": "success",
            "message": "Preprocessing retrieved successfully",
            "data": result.model_dump(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Error retrieving preprocessing",
                "detail": str(e),
            },
        )