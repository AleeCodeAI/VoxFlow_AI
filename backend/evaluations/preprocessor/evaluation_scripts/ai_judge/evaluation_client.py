from openai import OpenAI
from configs import MainSettings
from pydantic_schemas import AIResult, EvaluationError
from datetime import datetime
import time


class EvaluationClient:
    """Handles the OpenAI API call for AI Judge evaluation"""

    def __init__(self):
        self.settings = MainSettings()
        self.retries_limit = self.settings.AI_JUDGE_LLM_RETRIES
        self.retry_delay = self.settings.AI_JUDGE_RETRY_DELAY
        self.client = OpenAI(
            api_key=self.settings.OPENROUTER_API_KEY,
            base_url=self.settings.OPENROUTER_URL,
        )
        self.model = self.settings.GPT_MODEL

    def call(self, messages):
        last_exception = None

        for attempt in range(self.retries_limit + 1):
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
                if attempt < self.retries_limit:
                    time.sleep(self.retry_delay)

        error = EvaluationError(
            error_type="LLMCallError",
            message=str(last_exception),
            timestamp=datetime.now().isoformat(),
            context={"model": self.model, "attempts": self.retries_limit + 1},
        )
        raise Exception(error.model_dump_json())