from openai import OpenAI
from pydantic_schemas import AIResult, EvaluationError
from dotenv import load_dotenv
from datetime import datetime
import os
import time

load_dotenv(override=True)
api_key = os.getenv("OPENROUTER_API_KEY")
base_url = os.getenv("OPENROUTER_URL")
gpt = os.getenv("GPT_MODEL")

MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


class EvaluationClient:
    """Handles the OpenAI API call for AI Judge evaluation"""

    def __init__(self):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = gpt

    def call(self, messages):
        last_exception = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                    top_p=0.93,
                    response_format=AIResult,
                )
                return response.choices[0].message.parsed

            except Exception as e:
                last_exception = e
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)

        error = EvaluationError(
            error_type="LLMCallError",
            message=str(last_exception),
            timestamp=datetime.now().isoformat(),
            context={"model": self.model, "attempts": MAX_RETRIES},
        )
        raise Exception(error.model_dump_json())
