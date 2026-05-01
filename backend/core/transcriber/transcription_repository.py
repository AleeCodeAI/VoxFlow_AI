import os
import logging
from langfuse.decorators import observe

from databases.database import get_session
from databases.transcriber_repository import TranscriptionRepository as DBTranscriptionRepository

logging.basicConfig(level=logging.INFO, format="%(message)s")


class TranscriptionRepository:

    def __init__(self):
        self.db_repository = DBTranscriptionRepository()

    @observe(name="save-transcription", as_type="span")
    def save(self, audio_file, transcription_text, session_id):
        """
        Save the transcription to the database.

        Args:
            audio_file: Path to the original audio file
            transcription_text: The complete transcription text
            session_id: Unique identifier to link with preprocessing

        Returns:
            Transcription: The saved transcription object with metadata
        """
        audio_name = os.path.basename(audio_file)

        for session in get_session():
            return self.db_repository.save(
                session=session,
                session_id=session_id,
                audio_name=audio_name,
                transcription_text=transcription_text,
            )