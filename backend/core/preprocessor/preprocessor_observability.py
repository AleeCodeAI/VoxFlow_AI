import os
from langfuse import Langfuse
from langfuse.decorators import langfuse_context


class PreprocessorObservability:
    def __init__(self):
        self.langfuse = Langfuse(
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            host=os.getenv("LANGFUSE_HOST"),
        )

    def update_trace(self, session_id, audio_name, transcription_length, chunk_size):
        langfuse_context.update_current_trace(
            session_id=session_id,
            tags=["preprocessing", "audio"],
            metadata={
                "audio_name": audio_name,
                "transcription_length": transcription_length,
                "chunk_size": chunk_size,
            },
        )

    def score_success(self, comment="Successfully completed preprocessing"):
        self.langfuse.score(
            trace_id=langfuse_context.get_current_trace_id(),
            name="preprocessing-success",
            value=1,
            comment=comment,
        )

    def score_failure(self, reason: str):
        self.langfuse.score(
            trace_id=langfuse_context.get_current_trace_id(),
            name="preprocessing-failure",
            value=0,
            comment=reason,
        )

    def flush(self):
        self.langfuse.flush()
