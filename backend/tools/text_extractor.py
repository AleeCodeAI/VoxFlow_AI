from openai import OpenAI
from pydantic_schemas import TextExtraction, ProcessedData
from configs import TEXT_EXTRACTOR_PROMPT, MainSettings
from utils.color import Logger
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")


class TextExtracter(Logger):
    name = "TextAnalyzer"
    color = Logger.CYAN

    def __init__(self):
        llm_settings = MainSettings()
        self.client = OpenAI(
            api_key=llm_settings.OPENROUTER_API_KEY,
            base_url=llm_settings.OPENROUTER_URL,
        )
        self.prompt = TEXT_EXTRACTOR_PROMPT
        self.model = llm_settings.DEEPSEEK_MODEL
        self.log("Initialized TextAnalyzer")

    def make_messages(self, processed_data: ProcessedData):
        return [
            {"role": "system", "content": self.prompt.system_prompt},
            {
                "role": "user",
                "content": self.prompt.render_user(
                    "default", processed_data=processed_data
                ),
            },
        ]

    def extract(self, processed_data: ProcessedData):
        self.log("Sending the processed data for extraction")
        response = self.client.chat.completions.parse(
            model=self.model,
            messages=self.make_messages(processed_data),
            response_format=TextExtraction,
        )

        parsed_obj = getattr(response.choices[0].message, "parsed", None)
        if parsed_obj is not None:
            self.log(
                f"Extraction successfully finished with {len(parsed_obj.keywords)} keywords and {len(parsed_obj.keypoints)} keypoints"
            )
            return parsed_obj
        else:
            unparsed_response = response.choices[0].message.content
            self.log(
                f"Extraction failed. Returning unparsed data: {unparsed_response[:50]}"
            )
            return unparsed_response


if __name__ == "__main__":
    text_extracter = TextExtracter()

    example = ProcessedData(
        processed_data="""
Artificial Intelligence (AI) is essentially the quest to build machines that can perform tasks traditionally requiring human intelligence. Rather than following a rigid set of pre-programmed instructions, modern AI uses machine learning to find patterns in vast amounts of data, allowing it to "learn" and improve over time.
+1
How It Works
At the heart of today's AI boom are Neural Networks, which are loosely inspired by the structure of the human brain. These systems process information through layers of math, enabling them to:
Recognize: Identifying faces in photos or tumors in medical scans.
Predict: Estimating stock market trends or suggesting your next favorite song.
Generate: Creating original text, images, and even code (Generative AI).
AI in Daily Life
You likely interact with AI dozens of times a day without realizing it. It powers the voice assistants on your phone, the fraud detection systems at your bank, and the algorithms that curate your social media feeds.
+1
The Human Element
While AI can process data at speeds no human could match, it still lacks true sentience, emotion, and common sense. It excels at "narrow" tasks—like playing chess or translating languages—but it doesn't "understand" the world the way we do. The future of AI isn't just about the technology itself, but about how we collaborate with these tools to solve complex problems like climate change and disease
"""
    )
    result = text_extracter.extract(example)
    print(result)
