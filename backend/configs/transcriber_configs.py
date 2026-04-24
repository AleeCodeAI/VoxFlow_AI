from pydantic_settings import BaseSettings 

class TranscriberConfig(BaseSettings):
    """ Configuration settings for the Transcriber component. """

    MODEL: str = "small"
    MAX_RETRIES: int = 3 # remember: this is max retries for each chunk (in transcribe_chunk) and the exporting of chunks (in process_chunks). S
    CHUNK_LENGTH_MS: int = 90_000

    