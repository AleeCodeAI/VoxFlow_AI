from databases.database import get_session
from databases.models import TranscriptionModel


class TranscriberEvalQueryRepository:

    def exists(self, session_id: str) -> bool:
        """
        Check if a transcription exists in the database by ID.

        Args:
            session_id: Unique identifier for the transcription

        Returns:
            bool: True if the transcription exists, False otherwise
        """
        for session in get_session():
            result = session.get(TranscriptionModel, session_id)

        return result is not None