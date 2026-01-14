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

# Assuming your custom Logger is in a file named color.py
from color import Logger

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

MODEL = "small"

class Transcription(BaseModel):  
    id: str = Field(description="this is the id of the transcription")
    name: str = Field(description="Name of the audio file transcribed")
    transcription: str = Field(description="the full transcription of the audio file")
    timestamp: str = Field(description="Time of transcription")

class Transcriber(Logger):
    name = "Transcriber"
    color = Logger.BLUE

    def __init__(self):
        # Load model once
        self.whisper = whisper.load_model(MODEL)
        self.max_workers = 4
        self.chunk_length_ms = 90_000 
        # The Lock ensures only one thread uses the Whisper model at a time
        self.lock = threading.Lock()
        self.log(f"Loaded Whisper model '{MODEL}', workers: {self.max_workers}, chunk length: {self.chunk_length_ms}ms")

    def transcribe_chunk(self, chunk_data):
        """Transcribe a single chunk of audio safely using a thread lock."""
        idx, chunk_file = chunk_data
        self.log(f"Transcribing chunk {idx + 1}")
        
        try:
            # We use the lock here because the model can crash if multiple 
            # threads hit the CPU math layers at the exact same millisecond.
            with self.lock:
                # fp16=False prevents warnings on CPU
                result = self.whisper.transcribe(chunk_file, fp16=False)
            
            return idx, result["text"]
        except Exception as e:
            self.log(f"❌ Error in chunk {idx + 1}: {str(e)}")
            return idx, "[Error transcribing this section]"
    
    def save_transcription(self, audio_file, transcription_text):
        db_path = r"D:\Projects\audio_preprocessor\backend\databases"
        jsonl_file = os.path.join(db_path, "transcriptions.jsonl")
        
        os.makedirs(db_path, exist_ok=True)
        
        transcription_obj = Transcription(
            id=str(uuid.uuid4()),
            name=os.path.basename(audio_file),
            transcription=transcription_text,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Save with UTF-8 to handle all languages (Asian/European)
        with open(jsonl_file, 'a', encoding='utf-8') as f:
            f.write(transcription_obj.model_dump_json() + '\n')
            f.flush() 
            os.fsync(f.fileno()) 
            
        self.log(f"Transcription {transcription_obj.id} saved to {jsonl_file}")
        return transcription_obj

    def transcribe(self, audio_file):
        """
        Transcribe audio in chunks with protection against tiny audio fragments.
        """
        self.log(f"Loading audio file: {audio_file}")
        audio = AudioSegment.from_file(audio_file)

        # 1. Initial split
        raw_chunks = [audio[i:i + self.chunk_length_ms] for i in range(0, len(audio), self.chunk_length_ms)]
        
        # 2. Merge tiny trailing chunks (less than 1 second) to prevent tensor errors
        chunks = []
        for chunk in raw_chunks:
            if len(chunk) < 1000 and len(chunks) > 0:
                chunks[-1] = chunks[-1] + chunk
                self.log("Merged a tiny trailing fragment into the previous chunk.")
            else:
                chunks.append(chunk)

        self.log(f"Audio split into {len(chunks)} valid chunks")

        transcriptions = {}

        with TemporaryDirectory() as tmpdir:
            chunk_files = []
            for idx, chunk in enumerate(chunks):
                chunk_file = os.path.join(tmpdir, f"chunk_{idx}.mp3")
                chunk.export(chunk_file, format="mp3")
                chunk_files.append((idx, chunk_file))

            # Process chunks
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self.transcribe_chunk, chunk_data): chunk_data 
                          for chunk_data in chunk_files}
                
                for future in as_completed(futures):
                    idx, transcription_text = future.result()
                    transcriptions[idx] = transcription_text
                    self.log(f"Completed chunk {idx + 1}/{len(chunks)}")

        # Combine results in the correct order
        final_text = " ".join([transcriptions.get(i, "") for i in range(len(chunks))])
        
        return self.save_transcription(audio_file, final_text)

if __name__ == "__main__":
    transcriber = Transcriber()
    # Replace with your actual file name for testing
    try:
        result_obj = transcriber.transcribe("test2.mp3")
        print("\n==== Final Structured Output ====")
        print(f"ID: {result_obj.id}")
        print(f"File: {result_obj.name}")
        print(f"Transcription: {result_obj.transcription[:200]}...") # Show first 200 chars
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")