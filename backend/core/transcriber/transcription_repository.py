import os
from datetime import datetime
from pydantic_schemas import Transcription
from langfuse.decorators import observe

class TranscriptionRepository:

    def __init__(self, db_path=None):
        self.db_path = db_path or os.getenv(
            "DATABASE_PATH",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "databases")
        )

    @observe(name="save-transcription", as_type="span")
    def save(self, audio_file, transcription_text, session_id):
        """
        Save the transcription to JSONL database with the provided session ID.

        Args:
            audio_file: Path to the original audio file
            transcription_text: The complete transcription text
            session_id: Unique identifier to link with preprocessing

        Returns:
            Transcription: The saved transcription object with metadata
        """
        jsonl_file = os.path.join(self.db_path, "transcriptions.jsonl")
        os.makedirs(self.db_path, exist_ok=True)

        transcription_obj = Transcription(
            id=session_id,
            name=os.path.basename(audio_file),
            transcription=transcription_text,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        with open(jsonl_file, 'a', encoding='utf-8') as f:
            f.write(transcription_obj.model_dump_json() + '\n')
            f.flush()
            os.fsync(f.fileno())

        return transcription_obj