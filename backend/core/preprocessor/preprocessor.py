import logging
from dotenv import load_dotenv

from utils.color import Logger
from langfuse.decorators import observe

from pydantic_schemas import (PreprocessorError, 
                              LLMCallError, 
                              DatabaseError, 
                              PreprocessorReport)

from core.preprocessor.preprocessor_repository import PreprocessedRepository
from core.preprocessor.preprocessor_llm import PreprocessorLLM
from core.preprocessor.preprocessor_observability import PreprocessorObservability

from configs import PreprocessorConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")

load_dotenv(override=True)


class Preprocessor(Logger):
    name = "Preprocessor"
    color = Logger.GREEN

    def __init__(self):
        try:
            self.llm = PreprocessorLLM()
            self.repository = PreprocessedRepository()
            self.observability = PreprocessorObservability()
            self.preprocessor_config = PreprocessorConfig()
            self.system_prompt = self.preprocessor_config.PREPROCESSOR_SYSTEM_PROMPT
            self.user_prompt_with_context = self.preprocessor_config.PREPROCESSOR_USER_PROMPT_WITH_CONTEXT
            self.user_prompt_no_context = self.preprocessor_config.PREPROCESSOR_USER_PROMPT_NO_CONTEXT

            self.chunk_count = 0
            self.chunks_processed = 0

            self.log("Initialized Preprocessor")
        except Exception as e:
            logging.error(f"Failed to initialize Preprocessor: {str(e)}")
            raise

    @observe(name="make-messages", as_type="span")
    def make_messages(self, previous_chunk, current_chunk):
        """
        Construct the messages array for LLM with appropriate context.

        Args:
            previous_chunk: Previously cleaned text for context (empty string if first chunk)
            current_chunk: Current text chunk to be cleaned

        Returns:
            list: Messages array with system and user prompts
        """
        if previous_chunk:
            user_content = self.user_prompt_with_context.format(
                previous_chunk=previous_chunk, current_chunk=current_chunk
            )
        else:
            user_content = self.user_prompt_no_context.format(
                current_chunk=current_chunk
            )

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

    @observe(name="chunk-transcription", as_type="span")
    def chunk_transcription(self, transcription, chunk_size):
        """
        Split long transcription into smaller chunks at sentence boundaries.

        Args:
            transcription: Full transcription text to be chunked
            chunk_size: Maximum character count per chunk

        Returns:
            list: List of text chunks split at sentence boundaries
        """
        words = transcription.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1

            if current_length >= chunk_size and word[-1] in ".!?":
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        self.log(f"Split transcription into {len(chunks)} chunks")
        self.chunk_count = len(chunks) 
        return chunks

    @observe(name="audio-preprocessing")
    def preprocess(self, input_data, chunk_size=2000):
        """
        Main preprocessing workflow that cleans raw transcription text using LLM.
        Automatically chunks long texts and maintains context between chunks.
        Creates a Langfuse trace with session tracking and scores the result.

        Args:
            input_data: Dict or object containing transcription, id, and name
            chunk_size: Maximum characters per chunk (default 2000)

        Returns:
            PreprocessedResult: The final cleaned result saved to database

        Raises:
            PreprocessorError: If preprocessing fails at any stage
        """
        try:
            if isinstance(input_data, dict):
                raw_text = input_data.get("transcription", "")
                session_id = input_data.get("id", "")
                audio_name = input_data.get("name", "")
            else:
                raw_text = input_data.transcription
                session_id = input_data.id
                audio_name = input_data.name

            if not raw_text or not session_id:
                raise ValueError("Missing required fields: transcription or id")

            self.observability.update_trace(
                session_id=session_id,
                audio_name=audio_name,
                transcription_length=len(raw_text),
                chunk_size=chunk_size,
            )

            self.log(f"Starting preprocessing for ID: {session_id}")
            self.log(f"Transcription length: {len(raw_text)} characters")

            if len(raw_text) <= chunk_size:
                self.log("Processing in single pass...")
                final_combined_text = self.llm.call(self.make_messages("", raw_text))
            else:
                chunks = self.chunk_transcription(raw_text, chunk_size)
                preprocessed_chunks = []
                previous_preprocessed = ""

                for idx, chunk in enumerate(chunks):
                    self.log(f"Processing chunk {idx + 1}/{len(chunks)}")
                    current_clean = self.llm.call(
                        self.make_messages(previous_preprocessed, chunk),
                        chunk_idx=idx + 1,
                    )
                    self.chunks_processed += 1
                    preprocessed_chunks.append(current_clean)
                    previous_preprocessed = current_clean

                final_combined_text = " ".join(preprocessed_chunks)

            result = self.repository.save(session_id, 
                                          audio_name, 
                                          final_combined_text)
            
            self.observability.score_success()

            self.report = PreprocessorReport(  
                            chunk_count=self.chunk_count,
                            chunks_processed=self.chunks_processed,
                            llm_retries=self.llm.retries)
            
            return result

        except (LLMCallError, DatabaseError) as e:
            self.log(f"Preprocessing failed: {str(e)}")
            self.observability.score_failure(str(e))
            raise
        except Exception as e:
            error_msg = f"Unexpected error during preprocessing: {str(e)}"
            self.log(error_msg)
            self.observability.score_failure(error_msg)
            raise PreprocessorError(error_msg) from e


if __name__ == "__main__":
    preprocessor = Preprocessor()

    test_input = {
        "id": "d1f66d9b-414d-4b37-832d-6c494c0b8c53",
        "name": "test2.mp3",
        "transcription": """
Uh, so basically, I was thinking that, you know, maybe we could start by discussing the main idea... about the recent meeting we had with the clients. I mean, there were a lot of points raised, and I think it would be good to summarize them before we move on to the next steps. Also, I wanted to mention that the deadline for the project is coming up soon, so we need to make sure we're on track with our deliverables. Anyway, let me know your thoughts on this when you have a chance. Thanks!
""",
    }

    try:
        final_result = preprocessor.preprocess(test_input)

        print("\n" + "=" * 40)
        print("FINAL PREPROCESSED OBJECT")
        print("=" * 40)
        print(final_result.model_dump_json(indent=4))
    except PreprocessorError as e:
        print(f"\nPreprocessing failed: {str(e)}")
    finally:
        preprocessor.observability.flush()
        print("\nLangfuse traces flushed.")
