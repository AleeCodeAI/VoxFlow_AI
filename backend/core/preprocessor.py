import os
import logging
from datetime import datetime
from openai import OpenAI
from pydantic import BaseModel, Field 
from dotenv import load_dotenv
from core.color import Logger
from core.prompts import SYSTEM_PROMPT, USER_PROMPT_NO_CONTEXT, USER_PROMPT_WITH_CONTEXT

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
    """The final object structure for the database and UI."""
    id: str = Field(description="Matches the original transcription ID")
    name: str = Field(description="Original audio filename")
    preprocessed_transcription: str = Field(description="The cleaned text produced by LLM")
    timestamp: str = Field(description="Time of preprocessing")

class LLMParsedResponse(BaseModel):
    """Temporary model to force OpenAI to return a specific JSON key."""
    preprocessed_transcription: str = Field(description="The cleaned text")

class Preprocessor(Logger):
    name = "Preprocessor"
    color = Logger.GREEN 

    def __init__(self):
        self.client = OpenAI(api_key=api_key, base_url=url)
        self.model = gpt 
        self.system_prompt = SYSTEM_PROMPT
        self.user_prompt_with_context = USER_PROMPT_WITH_CONTEXT
        self.user_prompt_no_context = USER_PROMPT_NO_CONTEXT
        self.log("Initialized Preprocessor")

    def save_preprocessed(self, session_id, audio_name, clean_text):
        """Appends the clean result to preprocessings.jsonl."""
        db_path = r"D:\Projects\audio_preprocessor\backend\databases"
        jsonl_file = os.path.join(db_path, "preprocessings.jsonl")
        
        os.makedirs(db_path, exist_ok=True)
        
        # Create the result object using the SAME ID and Name from Transcriber
        result_obj = PreprocessedResult(
            id=session_id,
            name=audio_name,
            preprocessed_transcription=clean_text,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Append logic with flush to ensure data is written immediately
        with open(jsonl_file, 'a', encoding='utf-8') as f:
            f.write(result_obj.model_dump_json() + '\n')
            f.flush()
            os.fsync(f.fileno())
            
        self.log(f"Cleaned text for {audio_name} saved to {jsonl_file}")
        return result_obj

    def make_messages(self, previous_chunk, current_chunk):
        """Create messages with context from previous chunk."""
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

    def chunk_transcription(self, transcription, chunk_size):
        """Split transcription into chunks at sentence boundaries."""
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

    def preprocess(self, input_data, chunk_size=2000):
        """
        Main entry point with bulletproof parsing and fallbacks.
        """
        # 1. Extract data from the input package
        if isinstance(input_data, dict):
            raw_text = input_data.get("transcription", "")
            session_id = input_data.get("id", "")
            audio_name = input_data.get("name", "")
        else:
            raw_text = input_data.transcription
            session_id = input_data.id
            audio_name = input_data.name

        self.log(f"Starting preprocessing for ID: {session_id}")
        
        # 2. Decide between Single-Pass or Chunked Processing
        if len(raw_text) <= chunk_size:
            self.log("Processing in single pass...")
            messages = self.make_messages("", raw_text)
            
            response = self.client.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=LLMParsedResponse
            )
            
            # --- SAFETY CHECK ---
            # We check if 'parsed' exists. If not, we use 'content' as a fallback string.
            parsed_obj = getattr(response.choices[0].message, 'parsed', None)
            
            if parsed_obj is not None:
                final_combined_text = parsed_obj.preprocessed_transcription
            else:
                self.log("⚠️ Structured parsing failed. Falling back to raw content.")
                # If .parsed is None, the model likely returned a plain string or malformed JSON
                final_combined_text = response.choices[0].message.content
        
        else:
            chunks = self.chunk_transcription(raw_text, chunk_size)
            preprocessed_chunks = []
            previous_preprocessed = ""
            
            for idx, chunk in enumerate(chunks):
                self.log(f"Processing chunk {idx + 1}/{len(chunks)}")
                messages = self.make_messages(previous_preprocessed, chunk)
                
                response = self.client.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    response_format=LLMParsedResponse
                )
                
                # --- SAFETY CHECK FOR CHUNKS ---
                parsed_obj = getattr(response.choices[0].message, 'parsed', None)
                
                if parsed_obj is not None:
                    current_clean = parsed_obj.preprocessed_transcription
                else:
                    self.log(f"⚠️ Chunk {idx+1} parsing failed. Using raw content.")
                    current_clean = response.choices[0].message.content
                
                preprocessed_chunks.append(current_clean)
                previous_preprocessed = current_clean # Context for next chunk
            
            final_combined_text = " ".join(preprocessed_chunks)

        # 3. Save to database and return the final object
        return self.save_preprocessed(session_id, audio_name, final_combined_text)

# --- Execution Block ---

if __name__ == "__main__":
    preprocessor = Preprocessor()
    
    # This matches the structure of your transcriptions.jsonl line
    test_input = {
        "id": "d1f66d9b-414d-4b37-832d-6c494c0b8c53",
        "name": "test2.mp3",
        "transcription": """ 
Uh, so basically, I was thinking that, you know, maybe we could start by discussing the main idea of the project, um, which is to build a small app that, like, can take audio input from the user, and then, you know, convert it into text. And I mean, the thing is, we have to make sure that, like, the transcription is actually accurate because, you know, a lot of apps out there, they, um, just don’t get it right, and, like, it ends up being really confusing for the user. So yeah, the first step, I guess, would be to figure out how we can, um, capture audio in real time, maybe using a microphone API or something like that, and then, you know, feed it into the processing pipeline. And, uh, after that, we’ll need some kind of model that, um, can handle speech-to-text conversion, and I was, you know, thinking about Whisper, like, the OpenAI Whisper model, because it’s, uh, pretty good and supports multiple languages, which is kind of important if we want this app to be, you know, accessible internationally.

Then, um, after we get the transcription, we have to, like, clean it up because, you know, people talk in a very informal way, they use filler words, they repeat things, and the raw transcription can be, like, messy and hard to read. So, the preprocessing step is really important. And, um, we need to, like, remove things like 'um,' 'uh,' 'you know,' 'basically,' and 'actually,' and, you know, fix grammar mistakes, punctuation, and also maybe combine sentences that are broken up by, like, pauses. And we also need to make sure we preserve, like, all the important details, numbers, technical terms, names, and stuff like that, because those are crucial. Like, if someone mentions a specific date or a price, we can't just lose that in the cleaning process, you know?

And, uh, another thing I was thinking about is, you know, context. Like, when someone talks for a long time, we don’t want to just cut the transcription into pieces without considering the flow. So maybe we should, um, chunk the text but also feed the previous chunk as context to the LLM when we preprocess, so that, you know, sentences that depend on earlier statements still make sense. And, like, this will help maintain coherence and keep the meaning intact. I mean, if we just process small bites individually, the AI might lose track of who 'he' or 'she' is referring to from the previous paragraph, so keeping that window of context is, uh, definitely going to be a game changer for the final quality.

Also, um, in terms of the interface, I was thinking that the user could either upload an audio file, or, like, record live audio, and then the app would, you know, process it in real time. And, uh, maybe we could show the transcription updating live as it processes, which would be, you know, a nice feature. And we’ll have to make sure the backend can handle parallel processing if multiple chunks are being sent to the LLM at once, but, like, not overload the system because, you know, the model can be heavy, especially on CPU. We should probably implement some kind of queue system, you know, so that if ten people use it at once, the server doesn't just, like, explode or something

"""
    }
    
    # Process
    final_result = preprocessor.preprocess(test_input)
    
    # Show output
    print("\n" + "="*40)
    print("FINAL PREPROCESSED OBJECT")
    print("="*40)
    print(final_result.model_dump_json(indent=4))