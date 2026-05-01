from langfuse.decorators import observe

from utils.color import Logger
from databases.database import get_session
from databases.preprocessor_repository import PreprocessingRepository as DBPreprocessingRepository
from pydantic_schemas import DatabaseError


class PreprocessedRepository(Logger):
    name = "PreprocessedRepository"
    color = Logger.GREEN

    def __init__(self):
        self.db_repository = DBPreprocessingRepository()

    @observe(name="save-preprocessed", as_type="span")
    def save(self, session_id, audio_name, clean_text):
        """
        Save the preprocessed transcription to the database.

        Args:
            session_id: Unique identifier matching the original transcription
            audio_name: Original audio filename
            clean_text: The LLM-cleaned transcription text

        Returns:
            PreprocessedResult: The saved result object with metadata

        Raises:
            DatabaseError: If database operation fails
        """
        try:
            for session in get_session():
                result = self.db_repository.save(
                    session=session,
                    session_id=session_id,
                    audio_name=audio_name,
                    clean_text=clean_text,
                )

            self.log(f"Cleaned text for {audio_name} saved to database")
            self.log(f"Text length: {len(clean_text)} characters")
            return result

        except Exception as e:
            error_msg = f"Failed to save preprocessed data for {audio_name}: {str(e)}"
            self.log(error_msg)
            raise DatabaseError(error_msg) from e