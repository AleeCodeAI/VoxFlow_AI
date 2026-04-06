# preprocessed_repository.py

import os
from datetime import datetime
from pydantic_schemas import PreprocessedResult, DatabaseError
from langfuse.decorators import observe
from utils.color import Logger


class PreprocessedRepository(Logger):
    name = "PreprocessedRepository"
    color = Logger.GREEN

    def __init__(self, db_path=None):
        self.db_path = db_path or os.getenv(
            "DATABASE_PATH",
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "databases"
            ),
        )

    @observe(name="save-preprocessed", as_type="span")
    def save(self, session_id, audio_name, clean_text):
        """
        Save the preprocessed transcription to JSONL database.

        Args:
            session_id: Unique identifier matching the original transcription
            audio_name: Original audio filename
            clean_text: The LLM-cleaned transcription text

        Returns:
            PreprocessedResult: The saved result object with metadata

        Raises:
            DatabaseError: If file operations fail
        """
        try:
            jsonl_file = os.path.join(self.db_path, "preprocessings.jsonl")
            os.makedirs(self.db_path, exist_ok=True)

            result_obj = PreprocessedResult(
                id=session_id,
                name=audio_name,
                preprocessed_transcription=clean_text,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

            with open(jsonl_file, "a", encoding="utf-8") as f:
                f.write(result_obj.model_dump_json() + "\n")
                f.flush()
                os.fsync(f.fileno())

            self.log(f"Cleaned text for {audio_name} saved to {jsonl_file}")
            self.log(f"Text length: {len(clean_text)} characters")
            return result_obj

        except Exception as e:
            error_msg = f"Failed to save preprocessed data for {audio_name}: {str(e)}"
            self.log(error_msg)
            raise DatabaseError(error_msg) from e


if __name__ == "__main__":
    repo = PreprocessedRepository()
    print(repo.db_path)
