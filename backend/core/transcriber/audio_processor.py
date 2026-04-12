import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic_schemas import TranscriptionError, Retries, TranscriptionReport
from langfuse.decorators import observe, langfuse_context
from utils.color import Logger


class AudioProcessor(Logger):
    name = "AudioProcessor"
    color = Logger.BLUE

    def __init__(self, whisper_model, chunk_length_ms=90_000, max_workers=4):
        self.whisper = whisper_model
        self.chunk_length_ms = chunk_length_ms
        self.max_workers = max_workers
        self.lock = threading.Lock()

    def transcribe_chunk(self, chunk_data):
        """
        Transcribe a single audio chunk using Whisper model with thread safety.
        Uses a lock to prevent concurrent access to the model which can cause crashes.
        Note: Not decorated with @observe because it runs in parallel threads where
        Langfuse context is not maintained.

        Args:
            chunk_data: Tuple of (chunk_index, chunk_file_path)

        Returns:
            tuple: (chunk_index, transcription_text, retries_entry)
        """
        idx, chunk_file = chunk_data
        self.log(f"Transcribing chunk {idx + 1}")

        max_retries = 3
        retries_count = 0
        for attempt in range(max_retries):
            try:
                with self.lock:
                    result = self.whisper.transcribe(chunk_file, fp16=False)

                transcription_text = result["text"]
                retries_entry = Retries(
                    chunk_no=idx + 1, retries=retries_count, success=True
                )
                return idx, transcription_text, retries_entry
            except Exception as e:
                if attempt < max_retries - 1:
                    retries_count += 1
                    self.log(
                        f"Retry {attempt + 1}/{max_retries - 1} for chunk {idx + 1}: {str(e)}"
                    )
                else:
                    self.log(
                        f"Error in chunk {idx + 1} after {max_retries} attempts: {str(e)}"
                    )
                    retries_entry = Retries(
                        chunk_no=idx + 1, retries=retries_count, success=False
                    )
                    raise TranscriptionError(
                        f"Failed to transcribe chunk {idx + 1}: {str(e)}", retries_entry
                    )

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
        raw_chunks = [
            audio[i : i + self.chunk_length_ms]
            for i in range(0, len(audio), self.chunk_length_ms)
        ]

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
                "audio_duration_ms": len(audio),
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
            tuple: (transcriptions dict, TranscriptionReport)
        """
        transcriptions = {}
        chunk_files = []
        retries_list = []

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
                        self.log(
                            f"Retry {attempt + 1}/{max_retries - 1} exporting chunk {idx}: {str(e)}"
                        )
                    else:
                        self.log(
                            f"Error exporting chunk {idx} after {max_retries} attempts: {str(e)}"
                        )
                        raise TranscriptionError(
                            f"Failed to export chunk {idx}: {str(e)}"
                        )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.transcribe_chunk, chunk_data): chunk_data
                for chunk_data in chunk_files
            }

            for future in as_completed(futures):
                idx, transcription_text, retries_entry = future.result()
                transcriptions[idx] = transcription_text
                retries_list.append(retries_entry)
                self.log(f"Completed chunk {idx + 1}/{len(chunks)}")

        langfuse_context.update_current_observation(
            metadata={
                "total_chunks_processed": len(transcriptions),
                "max_workers": self.max_workers,
            }
        )

        report = TranscriptionReport(
            retries=retries_list,
            total_chunks=len(chunks),
            chunks_completed=len(transcriptions),
        )

        return transcriptions, report
