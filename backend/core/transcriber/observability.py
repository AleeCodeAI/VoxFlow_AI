import os
from langfuse import Langfuse
from langfuse.decorators import langfuse_context


class ObservabilityManager:
    def __init__(self):
        self.langfuse = Langfuse(
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            host=os.getenv("LANGFUSE_HOST"),
        )

    def update_trace(self, session_id, audio_file, model, chunk_length_ms, max_workers):
        langfuse_context.update_current_trace(
            session_id=session_id,
            tags=["transcription", "audio", "whisper"],
            metadata={
                "audio_file": os.path.basename(audio_file),
                "model": model,
                "chunk_length_ms": chunk_length_ms,
                "max_workers": max_workers,
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
