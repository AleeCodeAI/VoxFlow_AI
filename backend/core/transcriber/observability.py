import os
from langfuse import Langfuse
from langfuse.decorators import langfuse_context
from configs import MainSettings

class ObservabilityManager:
    def __init__(self):
        self.langfuse_settings = MainSettings()
        self.langfuse = Langfuse(
            secret_key=self.langfuse_settings.LANGFUSE_SECRET_KEY,
            public_key=self.langfuse_settings.LANGFUSE_PUBLIC_KEY,
            host=self.langfuse_settings.LANGFUSE_HOST,
        )

    def update_trace(self, session_id, audio_file, model, chunk_length_ms):
        langfuse_context.update_current_trace(
            session_id=session_id,
            tags=["transcription", "audio", "whisper"],
            metadata={
                "audio_file": os.path.basename(audio_file),
                "model": model,
                "chunk_length_ms": chunk_length_ms,
            },
        )

    def score_success(self, comment="Successfully completed transcription"):
        self.langfuse.score(
            trace_id=langfuse_context.get_current_trace_id(),
            name="transcription-success",
            value=1,
            comment=comment,
        )

    def flush(self):
        self.langfuse.flush()
