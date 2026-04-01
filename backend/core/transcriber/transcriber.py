# transcriber.py

import os
import uuid
import time
import logging
from tempfile import TemporaryDirectory

import whisper
from pydub import AudioSegment
from pydantic_schemas import TranscriptionError
from dotenv import load_dotenv

from utils.color import Logger
from langfuse.decorators import observe

from core.transcriber.audio_processor import AudioProcessor
from core.transcriber.transcription_repository import TranscriptionRepository
from core.transcriber.observability import ObservabilityManager

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

load_dotenv(override=True)

MODEL = "small"


class Transcriber(Logger):
    name = "Transcriber"
    color = Logger.BLUE

    def __init__(self):
        self.whisper = whisper.load_model(MODEL)
        self.max_workers = 4
        self.chunk_length_ms = 90_000

        self.audio_processor = AudioProcessor(
            whisper_model=self.whisper,
            chunk_length_ms=self.chunk_length_ms,
            max_workers=self.max_workers
        )
        self.repository = TranscriptionRepository()
        self.observability = ObservabilityManager()

        self.log(f"Loaded Whisper model '{MODEL}', workers: {self.max_workers}, chunk length: {self.chunk_length_ms}ms")

    @observe(name="audio-transcription")
    def transcribe(self, audio_file):
        """
        Main transcription workflow that processes audio file into text.
        Splits long audio into chunks, transcribes in parallel, and combines results.
        Creates a Langfuse trace with session tracking and scores the result.

        Args:
            audio_file: Path to the audio file to transcribe

        Returns:
            tuple: (Transcription, TranscriptionReport)
        """
        start_time = time.time()

        valid_formats = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.opus', '.wma', '.aac', '.webm']
        file_ext = os.path.splitext(audio_file)[1].lower()

        if not os.path.exists(audio_file):
            raise TranscriptionError(f"Audio file not found: {audio_file}")

        if file_ext not in valid_formats:
            raise TranscriptionError(f"Invalid file format '{file_ext}'. Supported formats: {', '.join(valid_formats)}")

        session_id = str(uuid.uuid4())

        self.observability.update_trace(
            session_id=session_id,
            audio_file=audio_file,
            model=MODEL,
            chunk_length_ms=self.chunk_length_ms,
            max_workers=self.max_workers
        )

        self.log(f"Loading audio file: {audio_file}")
        audio = AudioSegment.from_file(audio_file)

        chunks = self.audio_processor.split_audio_chunks(audio)

        with TemporaryDirectory() as tmpdir:
            transcriptions, report = self.audio_processor.process_chunks_parallel(chunks, tmpdir)

        final_text = " ".join([transcriptions.get(i, "") for i in range(len(chunks))])

        result = self.repository.save(audio_file, final_text, session_id)
        self.log(f"Transcription {result.id} saved")

        self.observability.score_success()

        report.total_time_ms = round((time.time() - start_time) * 1000)

        return result, report


if __name__ == "__main__":
    transcriber = Transcriber()

    try:
        result_obj, report = transcriber.transcribe(r"D:\Projects\audio_preprocessor\backend\evaluations\test_data\transcriber\valids\test3.wav")
        print("\n" + "="*40)
        print("FINAL TRANSCRIPTION OBJECT")
        print("="*40)
        print(f"ID: {result_obj.id}")
        print(f"File: {result_obj.name}")
        print(f"Transcription: {result_obj.transcription[:200]}...")

        transcriber.observability.flush()
        print("\nLangfuse traces flushed.")
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")