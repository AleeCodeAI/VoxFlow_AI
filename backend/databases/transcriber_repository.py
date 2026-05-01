from datetime import datetime
from pydantic_schemas import Transcription
from databases.models import TranscriptionModel


class TranscriptionRepository:

    def save(self, session, session_id: str, audio_name: str, transcription_text: str) -> Transcription:
        """
        Save or overwrite a transcription in the database.

        Args:
            session: SQLAlchemy session
            session_id: Unique identifier for the transcription
            audio_name: Original audio filename
            transcription_text: The full transcription text

        Returns:
            Transcription: Pydantic object of the saved transcription
        """
        timestamp = datetime.now()

        existing = session.get(TranscriptionModel, session_id)

        if existing:
            existing.name = audio_name
            existing.transcription = transcription_text
            existing.timestamp = timestamp
            db_obj = existing
        else:
            db_obj = TranscriptionModel(
                id=session_id,
                name=audio_name,
                transcription=transcription_text,
                timestamp=timestamp,
            )
            session.add(db_obj)

        session.flush()

        return Transcription(
            id=db_obj.id,
            name=db_obj.name,
            transcription=db_obj.transcription,
            timestamp=db_obj.timestamp,
        )

    def get(self, session, session_id: str) -> Transcription | None:
        """
        Retrieve a transcription by ID.

        Args:
            session: SQLAlchemy session
            session_id: Unique identifier for the transcription

        Returns:
            Transcription | None: Pydantic object if found, None otherwise
        """
        db_obj = session.get(TranscriptionModel, session_id)

        if not db_obj:
            return None

        return Transcription(
            id=db_obj.id,
            name=db_obj.name,
            transcription=db_obj.transcription,
            timestamp=db_obj.timestamp,
        )