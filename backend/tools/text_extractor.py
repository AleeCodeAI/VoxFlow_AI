from openai import OpenAI
from pydantic import ValidationError
from pydantic_schemas import TextExtraction, ProcessedData
from configs import TEXT_EXTRACTOR_PROMPT, MainSettings
from utils.color import Logger
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(message)s")


class TextExtracter(Logger):
    """
    TextExtracter
    -------------
    LLM-based text extraction pipeline that converts processed input text into structured
    TextExtraction schema using a primary + fallback model strategy.

    Features:
    - Manual JSON validation using Pydantic (no SDK parsing layer)
    - Retry mechanism with model fallback (DeepSeek → GPT)
    - Token usage + cost tracking (when available via provider)
    - Latency tracking per request + cumulative
    - Strict structured output enforcement via schema validation
    """

    name = "TextExtractor"
    color = Logger.CYAN

    def __init__(self):
        self.llm_settings = MainSettings()

        self.client = OpenAI(
            api_key=self.llm_settings.OPENROUTER_API_KEY,
            base_url=self.llm_settings.OPENROUTER_URL,
        )

        self.retries_limit = self.llm_settings.EXTRACTOR_LLM_RETRIES
        self.primary_model = self.llm_settings.GPT_MODEL
        self.fallback_model = self.llm_settings.GEMINI_MODEL

        self.prompt = TEXT_EXTRACTOR_PROMPT

        self.log("Initialized TextExtractor")

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

    def _extract_usage(self, response):
        usage = getattr(response, "usage", None)
        if not usage:
            return None

        cost_details = getattr(usage, "cost_details", None)

        return {
            "input_tokens": getattr(usage, "prompt_tokens", 0),
            "output_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
            "cost": getattr(usage, "cost", None),
            "input_cost": (
                cost_details.get("upstream_inference_prompt_cost", 0.0)
                if cost_details else None
            ),
            "output_cost": (
                cost_details.get("upstream_inference_completions_cost", 0.0)
                if cost_details else None
            ),
        }

    def _call_llm(self, model: str, messages: list):
        start_time = time.time()

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )

        latency = time.time() - start_time

        content = response.choices[0].message.content
        usage = self._extract_usage(response)

        return content, usage, latency

    def extract(self, processed_data: ProcessedData) -> TextExtraction:
        self.log("Starting text extraction pipeline")

        messages = self.make_messages(processed_data)

        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0.0
        total_latency = 0.0

        last_exception = None

        for attempt in range(self.retries_limit):
            model = self.primary_model if attempt == 0 else self.fallback_model

            try:
                self.log(f"Attempt {attempt + 1}/{self.retries_limit} using {model}")

                raw_output, usage, latency = self._call_llm(model, messages)

                total_latency += latency

                self.log(f"Latency: {latency:.2f}s")
                self.log(f"Raw output: {raw_output[:120]}")

                parsed = TextExtraction.model_validate_json(raw_output)

                if usage:
                    total_input_tokens += usage["input_tokens"]
                    total_output_tokens += usage["output_tokens"]

                    if usage["cost"]:
                        total_cost += usage["cost"]

                    self.log(
                        f"Tokens → in: {usage['input_tokens']}, "
                        f"out: {usage['output_tokens']}, "
                        f"cost: {usage['cost']}"
                    )

                self.log(
                    f"TOTAL → tokens in: {total_input_tokens}, "
                    f"tokens out: {total_output_tokens}, "
                    f"cost: {total_cost}, "
                    f"latency: {total_latency:.2f}s"
                )

                return parsed

            except ValidationError as ve:
                last_exception = ve
                self.log(f"Validation error: {ve}")

            except Exception as e:
                last_exception = e
                self.log(f"LLM error: {e}")

        self.log(f"All retries failed. Last error: {last_exception}")
        raise last_exception


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