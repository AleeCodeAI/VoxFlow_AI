from pydantic_settings import BaseSettings 

class TranscriberConfig(BaseSettings):
    """ Configuration settings for the Transcriber component. """

    MODEL: str = "small"
    MAX_WORKERS: int = 4
    CHUNK_LENGTH_MS: int = 90_000

    