from databases.database import get_session
from databases.models import PreprocessingModel


class PreprocessorEvalQueryRepository:

    def exists(self, preprocessing_id: str) -> bool:
        """
        Check if a preprocessed transcription exists in the database by ID.

        Args:
            preprocessing_id: Unique identifier for the preprocessed transcription

        Returns:
            bool: True if the preprocessed transcription exists, False otherwise
        """
        for session in get_session():
            result = session.get(PreprocessingModel, preprocessing_id)

        return result is not None