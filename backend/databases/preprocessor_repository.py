from datetime import datetime
from pydantic_schemas import PreprocessedResult, DatabaseError
from databases.models import PreprocessingModel


class PreprocessingRepository:

    def save(self, session, session_id: str, audio_name: str, clean_text: str) -> PreprocessedResult:
        """
        Save or overwrite a preprocessed transcription in the database.

        Args:
            session: SQLAlchemy session
            session_id: Unique identifier matching the original transcription
            audio_name: Original audio filename
            clean_text: The LLM-cleaned transcription text

        Returns:
            PreprocessedResult: Pydantic object of the saved result

        Raises:
            DatabaseError: If database operation fails
        """
        try:
            timestamp = datetime.now()

            existing = session.get(PreprocessingModel, session_id)

            if existing:
                existing.name = audio_name
                existing.preprocessed_transcription = clean_text
                existing.timestamp = timestamp
                db_obj = existing
            else:
                db_obj = PreprocessingModel(
                    id=session_id,
                    name=audio_name,
                    preprocessed_transcription=clean_text,
                    timestamp=timestamp,
                )
                session.add(db_obj)

            session.flush()

            return PreprocessedResult(
                id=db_obj.id,
                name=db_obj.name,
                preprocessed_transcription=db_obj.preprocessed_transcription,
                timestamp=db_obj.timestamp,
            )

        except Exception as e:
            raise DatabaseError(f"Failed to save preprocessed data for {audio_name}: {str(e)}") from e

    def get(self, session, session_id: str) -> PreprocessedResult | None:
        """
        Retrieve a preprocessed result by ID.

        Args:
            session: SQLAlchemy session
            session_id: Unique identifier for the preprocessing

        Returns:
            PreprocessedResult | None: Pydantic object if found, None otherwise
        """
        db_obj = session.get(PreprocessingModel, session_id)

        if not db_obj:
            return None

        return PreprocessedResult(
            id=db_obj.id,
            name=db_obj.name,
            preprocessed_transcription=db_obj.preprocessed_transcription,
            timestamp=db_obj.timestamp,
        )