from fastapi import APIRouter, HTTPException
import os
import json

router = APIRouter()


def get_db_path():
    return os.getenv(
        "DATABASE_PATH",
        os.path.join(os.path.dirname(__file__), "..", "..", "databases")
    )


def read_jsonl(file_path):
    """Safely read JSONL file line by line."""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


@router.get("/transcriptions/{transcription_id}")
async def get_transcription(transcription_id: str):
    try:
        db_path = get_db_path()
        jsonl_file = os.path.join(db_path, "transcriptions.jsonl")

        if not os.path.exists(jsonl_file):
            raise HTTPException(
                status_code=404,
                detail={"status": "error", "message": "No transcriptions found"},
            )

        for obj in read_jsonl(jsonl_file):
            if obj.get("id") == transcription_id:
                return {
                    "status": "success",
                    "message": "Transcription retrieved successfully",
                    "data": obj,
                }

        raise HTTPException(
            status_code=404,
            detail={"status": "error", "message": "Transcription not found"},
        )

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
        db_path = get_db_path()
        jsonl_file = os.path.join(db_path, "preprocessings.jsonl")

        if not os.path.exists(jsonl_file):
            raise HTTPException(
                status_code=404,
                detail={"status": "error", "message": "No preprocessings found"},
            )

        for obj in read_jsonl(jsonl_file):
            if obj.get("id") == preprocessing_id:
                return {
                    "status": "success",
                    "message": "Preprocessing retrieved successfully",
                    "data": obj,
                }

        raise HTTPException(
            status_code=404,
            detail={"status": "error", "message": "Preprocessing not found"},
        )

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