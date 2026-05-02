import hashlib
import redis
from configs import MainSettings


class TranscriptionCache:

    def __init__(self):
        self.settings = MainSettings()
        self.client = redis.Redis(
            host=self.settings.REDIS_HOST,
            port=self.settings.REDIS_PORT,
            db=self.settings.REDIS_DB,
            decode_responses=True,
        )
        self.ttl = self.settings.REDIS_TTL

    def compute_hash(self, file_bytes: bytes) -> str:
        """
        Compute SHA256 hash of audio file bytes.

        Args:
            file_bytes: Raw bytes of the audio file

        Returns:
            str: SHA256 hex digest of the file
        """
        return hashlib.sha256(file_bytes).hexdigest()

    def get(self, file_hash: str) -> str | None:
        """
        Retrieve cached transcription by file hash.

        Args:
            file_hash: SHA256 hash of the audio file

        Returns:
            str: Cached transcription text or None if not found
        """
        return self.client.get(f"transcription:{file_hash}")

    def set(self, file_hash: str, transcription: str) -> None:
        """
        Store transcription in cache with TTL.

        Args:
            file_hash: SHA256 hash of the audio file
            transcription: Transcription text to cache
        """
        self.client.setex(
            name=f"transcription:{file_hash}",
            time=self.ttl,
            value=transcription,
        )