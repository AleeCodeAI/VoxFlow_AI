import os
import threading
import uuid
import logging
from datetime import datetime
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor, as_completed

import whisper
from pydub import AudioSegment
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from color import Logger
from langfuse.decorators import observe, langfuse_context
from langfuse import Langfuse

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

load_dotenv(override=True)

MODEL = "small"

class TranscriptionError(Exception):
    pass

class Transcription(BaseModel):
    """Database model for storing audio transcription results."""
    id: str = Field(description="Unique identifier for the transcription")
    name: str = Field(description="Name of the audio file transcribed")
    transcription: str = Field(description="The full transcription of the audio file")
    timestamp: str = Field(description="Time of transcription")

class Transcriber(Logger):
    name = "Transcriber"
    color = Logger.BLUE

    def __init__(self):
        """
        Initialize the Transcriber with Whisper model and Langfuse observability.
        Sets up thread-safe transcription with configurable chunk processing.
        """
        self.whisper = whisper.load_model(MODEL)
        self.max_workers = 4
        self.chunk_length_ms = 90_000
        self.lock = threading.Lock()
        
        self.langfuse = Langfuse(
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            host=os.getenv("LANGFUSE_HOST")
        )
        
        self.log(f"Loaded Whisper model '{MODEL}', workers: {self.max_workers}, chunk length: {self.chunk_length_ms}ms")

    def transcribe_chunk(self, chunk_data):
        """
        Transcribe a single audio chunk using Whisper model with thread safety.
        Uses a lock to prevent concurrent access to the model which can cause crashes.
        Note: Not decorated with @observe because it runs in parallel threads where
        Langfuse context is not maintained.
        
        Args:
            chunk_data: Tuple of (chunk_index, chunk_file_path)
            
        Returns:
            tuple: (chunk_index, transcription_text)
        """
        idx, chunk_file = chunk_data
        self.log(f"Transcribing chunk {idx + 1}")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with self.lock:
                    result = self.whisper.transcribe(chunk_file, fp16=False)
                
                transcription_text = result["text"]
                return idx, transcription_text
            except Exception as e:
                if attempt < max_retries - 1:
                    self.log(f"Retry {attempt + 1}/{max_retries - 1} for chunk {idx + 1}: {str(e)}")
                else:
                    self.log(f"Error in chunk {idx + 1} after {max_retries} attempts: {str(e)}")
                    raise TranscriptionError(f"Failed to transcribe chunk {idx + 1}: {str(e)}")

    @observe(name="save-transcription", as_type="span")
    def save_transcription(self, audio_file, transcription_text, session_id):
        """
        Save the transcription to JSONL database with the provided session ID.
        
        Args:
            audio_file: Path to the original audio file
            transcription_text: The complete transcription text
            session_id: Unique identifier to link with preprocessing
            
        Returns:
            Transcription: The saved transcription object with metadata
        """
        db_path = r"D:\Projects\audio_preprocessor\backend\databases"
        jsonl_file = os.path.join(db_path, "transcriptions.jsonl")
        os.makedirs(db_path, exist_ok=True)
        
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
            
        self.log(f"Transcription {transcription_obj.id} saved to {jsonl_file}")
        return transcription_obj

    @observe(name="split-audio-chunks", as_type="span")
    def split_audio_chunks(self, audio):
        """
        Split audio into chunks and merge tiny trailing fragments.
        Prevents tensor errors from sub-second audio fragments.
        
        Args:
            audio: AudioSegment object to be split
            
        Returns:
            list: List of AudioSegment chunks ready for transcription
        """
        raw_chunks = [audio[i:i + self.chunk_length_ms] 
                     for i in range(0, len(audio), self.chunk_length_ms)]
        
        chunks = []
        for chunk in raw_chunks:
            if len(chunk) < 1000 and len(chunks) > 0:
                chunks[-1] = chunks[-1] + chunk
                self.log("Merged a tiny trailing fragment into the previous chunk.")
            else:
                chunks.append(chunk)

        self.log(f"Audio split into {len(chunks)} valid chunks")
        
        langfuse_context.update_current_observation(
            metadata={
                "total_chunks": len(chunks),
                "chunk_length_ms": self.chunk_length_ms,
                "audio_duration_ms": len(audio)
            }
        )
        
        return chunks

    @observe(name="process-chunks-parallel", as_type="span")
    def process_chunks_parallel(self, chunks, tmpdir):
        """
        Export audio chunks to temporary files and transcribe them in parallel.
        
        Args:
            chunks: List of AudioSegment chunks
            tmpdir: Temporary directory path for chunk files
            
        Returns:
            dict: Dictionary mapping chunk indices to transcription text
        """
        transcriptions = {}
        chunk_files = []
        
        for idx, chunk in enumerate(chunks):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    chunk_file = os.path.join(tmpdir, f"chunk_{idx}.mp3")
                    chunk.export(chunk_file, format="mp3")
                    chunk_files.append((idx, chunk_file))
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        self.log(f"Retry {attempt + 1}/{max_retries - 1} exporting chunk {idx}: {str(e)}")
                    else:
                        self.log(f"Error exporting chunk {idx} after {max_retries} attempts: {str(e)}")
                        raise TranscriptionError(f"Failed to export chunk {idx}: {str(e)}")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.transcribe_chunk, chunk_data): chunk_data 
                      for chunk_data in chunk_files}
            
            for future in as_completed(futures):
                idx, transcription_text = future.result()
                transcriptions[idx] = transcription_text
                self.log(f"Completed chunk {idx + 1}/{len(chunks)}")

        langfuse_context.update_current_observation(
            metadata={
                "total_chunks_processed": len(transcriptions),
                "max_workers": self.max_workers
            }
        )
        
        return transcriptions

    @observe(name="audio-transcription")
    def transcribe(self, audio_file):
        """
        Main transcription workflow that processes audio file into text.
        Splits long audio into chunks, transcribes in parallel, and combines results.
        Creates a Langfuse trace with session tracking and scores the result.
        
        Args:
            audio_file: Path to the audio file to transcribe
            
        Returns:
            Transcription: The final transcription object saved to database
        """

        valid_formats = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.opus', '.wma', '.aac']
        file_ext = os.path.splitext(audio_file)[1].lower()
        
        if not os.path.exists(audio_file):
            raise TranscriptionError(f"Audio file not found: {audio_file}")
        
        if file_ext not in valid_formats:
            raise TranscriptionError(f"Invalid file format '{file_ext}'. Supported formats: {', '.join(valid_formats)}")
        
        session_id = str(uuid.uuid4())
        
        langfuse_context.update_current_trace(
            session_id=session_id,
            tags=["transcription", "audio", "whisper"],
            metadata={
                "audio_file": os.path.basename(audio_file),
                "model": MODEL,
                "chunk_length_ms": self.chunk_length_ms,
                "max_workers": self.max_workers
            }
        )

        self.log(f"Loading audio file: {audio_file}")
        audio = AudioSegment.from_file(audio_file)

        chunks = self.split_audio_chunks(audio)

        with TemporaryDirectory() as tmpdir:
            transcriptions = self.process_chunks_parallel(chunks, tmpdir)

        final_text = " ".join([transcriptions.get(i, "") for i in range(len(chunks))])
        
        result = self.save_transcription(audio_file, final_text, session_id)
        
        self.langfuse.score(
            trace_id=langfuse_context.get_current_trace_id(),
            name="transcription-success",
            value=1,
            comment="Successfully completed transcription"
        )
        
        return result

if __name__ == "__main__":
    transcriber = Transcriber()
    
    try:
        result_obj = transcriber.transcribe("test3.mp3")
        print("\n" + "="*40)
        print("FINAL TRANSCRIPTION OBJECT")
        print("="*40)
        print(f"ID: {result_obj.id}")
        print(f"File: {result_obj.name}")
        print(f"Transcription: {result_obj.transcription[:200]}...")
        
        transcriber.langfuse.flush()
        print("\nLangfuse traces flushed.")
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        