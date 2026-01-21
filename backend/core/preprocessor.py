import os
import logging
from datetime import datetime
from openai import OpenAI
from pydantic import BaseModel, Field 
from dotenv import load_dotenv
from core.color import Logger
from core.prompts import SYSTEM_PROMPT, USER_PROMPT_NO_CONTEXT, USER_PROMPT_WITH_CONTEXT
from langfuse.decorators import observe, langfuse_context
from langfuse import Langfuse

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

load_dotenv(override=True)

api_key = os.getenv("OPENROUTER_API_KEY")
url = os.getenv("OPENROUTER_URL")
gpt = os.getenv("GPT_MODEL")
deepseek = os.getenv("DEEPSEEK_MODEL")

class PreprocessedResult(BaseModel):
    """Database model for storing preprocessed transcription results."""
    id: str = Field(description="Matches the original transcription ID")
    name: str = Field(description="Original audio filename")
    preprocessed_transcription: str = Field(description="The cleaned text produced by LLM")
    timestamp: str = Field(description="Time of preprocessing")

class LLMParsedResponse(BaseModel):
    """Response schema for structured output from LLM."""
    preprocessed_transcription: str = Field(description="The cleaned text")

class Preprocessor(Logger):
    name = "Preprocessor"
    color = Logger.GREEN 

    def __init__(self):
        """
        Initialize the Preprocessor with OpenAI client, Langfuse observability, and prompts.
        Sets up connection to OpenRouter API and Langfuse for tracking.
        """
        self.client = OpenAI(api_key=api_key, base_url=url)
        self.langfuse = Langfuse(
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            host=os.getenv("LANGFUSE_HOST")
        )
        self.model = gpt 
        self.system_prompt = SYSTEM_PROMPT
        self.user_prompt_with_context = USER_PROMPT_WITH_CONTEXT
        self.user_prompt_no_context = USER_PROMPT_NO_CONTEXT
        self.log("Initialized Preprocessor")

    @observe(name="save-preprocessed", as_type="span")
    def save_preprocessed(self, session_id, audio_name, clean_text):
        """
        Save the preprocessed transcription to JSONL database.
        
        Args:
            session_id: Unique identifier matching the original transcription
            audio_name: Original audio filename
            clean_text: The LLM-cleaned transcription text
            
        Returns:
            PreprocessedResult: The saved result object with metadata
        """
        db_path = r"D:\Projects\audio_preprocessor\backend\databases"
        jsonl_file = os.path.join(db_path, "preprocessings.jsonl")
        os.makedirs(db_path, exist_ok=True)
        
        result_obj = PreprocessedResult(
            id=session_id,
            name=audio_name,
            preprocessed_transcription=clean_text,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        with open(jsonl_file, 'a', encoding='utf-8') as f:
            f.write(result_obj.model_dump_json() + '\n')
            f.flush()
            os.fsync(f.fileno())
            
        self.log(f"Cleaned text for {audio_name} saved to {jsonl_file}")
        return result_obj

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
                previous_chunk=previous_chunk,
                current_chunk=current_chunk
            )
        else:
            user_content = self.user_prompt_no_context.format(
                current_chunk=current_chunk
            )
        
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content}
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
            
            if current_length >= chunk_size and word[-1] in '.!?':
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        self.log(f"Split transcription into {len(chunks)} chunks")
        return chunks

    @observe(name="call-llm-engine", as_type="generation")
    def call_llm(self, messages, chunk_idx=None):
        """
        Call the LLM to clean transcription text with structured output parsing.
        Enriches Langfuse generation with token usage and cost data from OpenRouter.
        
        Args:
            messages: Array of message objects for the LLM
            chunk_idx: Optional chunk number for tracking in metadata
            
        Returns:
            str: The cleaned transcription text
        """
        langfuse_context.update_current_observation(
            model=self.model,
            input=messages
        )
        
        response = self.client.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=LLMParsedResponse
        )
        
        parsed_obj = getattr(response.choices[0].message, 'parsed', None)
        
        if parsed_obj is not None:
            content = parsed_obj.preprocessed_transcription
        else:
            self.log("Structured parsing failed. Falling back to raw content.")
            content = response.choices[0].message.content
        
        if response.usage:
            input_cost = float(response.usage.cost_details.get('upstream_inference_prompt_cost', 0.0))
            output_cost = float(response.usage.cost_details.get('upstream_inference_completions_cost', 0.0))
            total_cost = float(response.usage.cost)
            upstream_inference_cost = float(response.usage.cost_details.get('upstream_inference_cost', 0.0))
            
            cached_tokens = response.usage.prompt_tokens_details.cached_tokens
            reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens
            
            langfuse_context.update_current_observation(
                output=content,
                usage={
                    "input": response.usage.prompt_tokens,
                    "output": response.usage.completion_tokens,
                    "total": response.usage.total_tokens,
                    "unit": "TOKENS",
                    "input_cost": input_cost,
                    "output_cost": output_cost
                },
                metadata={
                    "chunk_index": chunk_idx,
                    "total_cost": total_cost,
                    "upstream_inference_cost": upstream_inference_cost,
                    "cached_tokens": cached_tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "is_byok": response.usage.is_byok
                }
            )
            
            self.log(f"Tokens: {response.usage.total_tokens} | Cost: ${total_cost:.8f}")
        
        return content

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
        """
        if isinstance(input_data, dict):
            raw_text = input_data.get("transcription", "")
            session_id = input_data.get("id", "")
            audio_name = input_data.get("name", "")
        else:
            raw_text = input_data.transcription
            session_id = input_data.id
            audio_name = input_data.name

        langfuse_context.update_current_trace(
            session_id=session_id,
            tags=["preprocessing", "audio"],
            metadata={
                "audio_name": audio_name,
                "transcription_length": len(raw_text),
                "chunk_size": chunk_size
            }
        )

        self.log(f"Starting preprocessing for ID: {session_id}")
        
        if len(raw_text) <= chunk_size:
            self.log("Processing in single pass...")
            final_combined_text = self.call_llm(self.make_messages("", raw_text))
        else:
            chunks = self.chunk_transcription(raw_text, chunk_size)
            preprocessed_chunks = []
            previous_preprocessed = ""
            
            for idx, chunk in enumerate(chunks):
                self.log(f"Processing chunk {idx + 1}/{len(chunks)}")
                current_clean = self.call_llm(
                    self.make_messages(previous_preprocessed, chunk),
                    chunk_idx=idx + 1
                )
                preprocessed_chunks.append(current_clean)
                previous_preprocessed = current_clean
            
            final_combined_text = " ".join(preprocessed_chunks)

        result = self.save_preprocessed(session_id, audio_name, final_combined_text)
        
        self.langfuse.score(
            trace_id=langfuse_context.get_current_trace_id(),
            name="preprocessing-success",
            value=1,
            comment="Successfully completed preprocessing"
        )
        
        return result

if __name__ == "__main__":
    preprocessor = Preprocessor()
    
    test_input = {
        "id": "d1f66d9b-414d-4b37-832d-6c494c0b8c53",
        "name": "test2.mp3",
        "transcription": """ 
Uh, so basically, I was thinking that, you know, maybe we could start by discussing the main idea of the project, um, which is to build a small app that, like, can take audio input from the user, and then, you know, convert it into text. And I mean, the thing is, we have to make sure that, like, the transcription is actually accurate because, you know, a lot of apps out there, they, um, just don't get it right, and, like, it ends up being really confusing for the user. So yeah, the first step, I guess, would be to figure out how we can, um, capture audio in real time, maybe using a microphone API or something like that, and then, you know, feed it into the processing pipeline. And, uh, after that, we'll need some kind of model that, um, can handle speech-to-text conversion, and I was, you know, thinking about Whisper, like, the OpenAI Whisper model, because it's, uh, pretty good and supports multiple languages, which is kind of important if we want this app to be, you know, accessible internationally.

Then, um, after we get the transcription, we have to, like, clean it up because, you know, people talk in a very informal way, they use filler words, they repeat things, and the raw transcription can be, like, messy and hard to read. So, the preprocessing step is really important. And, um, we need to, like, remove things like 'um,' 'uh,' 'you know,' 'basically,' and 'actually,' and, you know, fix grammar mistakes, punctuation, and also maybe combine sentences that are broken up by, like, pauses. And we also need to make sure we preserve, like, all the important details, numbers, technical terms, names, and stuff like that, because those are crucial. Like, if someone mentions a specific date or a price, we can't just lose that in the cleaning process, you know?

And, uh, another thing I was thinking about is, you know, context. Like, when someone talks for a long time, we don't want to just cut the transcription into pieces without considering the flow. So maybe we should, um, chunk the text but also feed the previous chunk as context to the LLM when we preprocess, so that, you know, sentences that depend on earlier statements still make sense. And, like, this will help maintain coherence and keep the meaning intact. I mean, if we just process small bites individually, the AI might lose track of who 'he' or 'she' is referring to from the previous paragraph, so keeping that window of context is, uh, definitely going to be a game changer for the final quality.

Also, um, in terms of the interface, I was thinking that the user could either upload an audio file, or, like, record live audio, and then the app would, you know, process it in real time. And, uh, maybe we could show the transcription updating live as it processes, which would be, you know, a nice feature. And we'll have to make sure the backend can handle parallel processing if multiple chunks are being sent to the LLM at once, but, like, not overload the system because, you know, the model can be heavy, especially on CPU. We should probably implement some kind of queue system, you know, so that if ten people use it at once, the server doesn't just, like, explode or something

"""
    }
    
    final_result = preprocessor.preprocess(test_input)
    
    print("\n" + "="*40)
    print("FINAL PREPROCESSED OBJECT")
    print("="*40)
    print(final_result.model_dump_json(indent=4))
    
    preprocessor.langfuse.flush()
    print("\nLangfuse traces flushed.")