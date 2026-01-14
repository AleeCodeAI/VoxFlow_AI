import os 
from dotenv import load_dotenv
from openai import OpenAI 
from pydantic import BaseModel, Field 
from typing import List
from color import Logger
import logging 

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
url = os.getenv("OPENROUTER_URL")
deepseek = os.getenv("DEEPSEEK_MODEL")
gpt = os.getenv("GPT_MODEL")

class TextExtraction(BaseModel):
    keywords: List[str] = Field(
        description="A list of specific keywords found in the text, specifically focusing on terms related to the Rajya Sabha (e.g., Chairman, MP, Bill, Session).")
    keypoints: List[str] = Field(
        description="A list of bullet points summarizing the main actions or discussions involving the Rajya Sabha within this data.")

class ProcessedData(BaseModel):
    processed_data: str = Field(description="the processed data as input to TextAnalyzer")

SYSTEM_PROMPT = """ 
Role: You are a Precise Data Extraction Assistant. Your goal is to analyze text and transform it into a structured JSON format for reporting.

Task: Examine the provided text and perform two primary extractions:
1. Keywords: Identify the most significant nouns, technical terms, or names mentioned in the text.
2. Keypoints: Summarize the essential facts, actions, or takeaways from the text into a list of clear, concise bullet points.

Guidelines:
- Focus on "signal over noise"—exclude generic filler words.
- Ensure the keypoints capture the "Who, What, and Why" of the input.
- Maintain a neutral, professional tone.

Output Format:
You must return ONLY a JSON object. Do not include any conversational text before or after the JSON.

{
  "keywords": ["word1", "word2", "word3"],
  "keypoints": [
    "The primary action or fact identified in the text.",
    "A secondary supporting detail or result.",
    "A final takeaway or next step mentioned."
  ]
}
"""

USER_PROMPT = """ 
here is the data to extract the keywords and keypoints:

{processed_data}
"""

class TextExtracter(Logger):
    name = "TextAnalyzer"
    color = Logger.CYAN

    def __init__(self):
        self.client = OpenAI(api_key=api_key, base_url=url)
        self.system_prompt = SYSTEM_PROMPT
        self.user_prompt = USER_PROMPT
        self.model = deepseek
        self.log("Initialized TextAnalyzer")

    def make_messages(self, processed_data: ProcessedData):
        return [{"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt.format(processed_data=processed_data)}]
    
    def extract(self, processed_data: ProcessedData):
        self.log("Sending the processed data for extracttion")
        response = self.client.chat.completions.parse(model=self.model,
                                           messages=self.make_messages(processed_data),
                                           response_format=TextExtraction)
        
        parsed_obj = getattr(response.choices[0].message, 'parsed', None)
        if parsed_obj is not None:
            self.log(f"Extarction successfully finished with {len(parsed_obj.keywords)} keywords and {len(parsed_obj.keypoints)} keypoints")
            return parsed_obj
        else:
            unparsed_response = response.choices[0].message.content
            self.log(f"Extraction failed. Returning unparsed data: {unparsed_response[:50]}")
            return unparsed_response
        
if __name__ == "__main__":
    text_extracter = TextExtracter()

    example = ProcessedData(processed_data=""" 
Artificial Intelligence (AI) is essentially the quest to build machines that can perform tasks traditionally requiring human intelligence. Rather than following a rigid set of pre-programmed instructions, modern AI uses machine learning to find patterns in vast amounts of data, allowing it to "learn" and improve over time.
+1
How It Works
At the heart of today’s AI boom are Neural Networks, which are loosely inspired by the structure of the human brain. These systems process information through layers of math, enabling them to:
Recognize: Identifying faces in photos or tumors in medical scans.
Predict: Estimating stock market trends or suggesting your next favorite song.
Generate: Creating original text, images, and even code (Generative AI).
AI in Daily Life
You likely interact with AI dozens of times a day without realizing it. It powers the voice assistants on your phone, the fraud detection systems at your bank, and the algorithms that curate your social media feeds.
+1
The Human Element
While AI can process data at speeds no human could match, it still lacks true sentience, emotion, and common sense. It excels at "narrow" tasks—like playing chess or translating languages—but it doesn't "understand" the world the way we do. The future of AI isn't just about the technology itself, but about how we collaborate with these tools to solve complex problems like climate change and disease
""")
    result = text_extracter.extract(example)
    print(result)
