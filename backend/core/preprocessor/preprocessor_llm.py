from openai import OpenAI
from pydantic_schemas import LLMParsedResponse, LLMCallError
from configs import LLMConfig
from langfuse.decorators import observe, langfuse_context
from utils.color import Logger

class PreprocessorLLM(Logger):
    name = "PreprocessorLLM"
    color = Logger.GREEN

    def __init__(self):
        self.llm_config = LLMConfig()
        self.retries = 0
        self.api_key = self.llm_config.OPENROUTER_API_KEY
        self.url = self.llm_config.OPENROUTER_URL
        self.client = OpenAI(api_key=self.api_key, base_url=self.url)
        self.gemini = self.llm_config.GEMINI_MODEL
        self.gpt = self.llm_config.GPT_MODEL

    @observe(name="call-llm-engine", as_type="generation")
    def call(self, messages, chunk_idx=None):
        """
        Call the LLM to clean transcription text with structured output parsing.
        Tries GPT first, then falls back to Gemini for 2 retries.

        Args:
            messages: Array of message objects for the LLM
            chunk_idx: Optional chunk number for tracking in metadata

        Returns:
            str: The cleaned transcription text

        Raises:
            LLMCallError: If all 3 attempts fail
        """
        last_error = None

        for attempt in range(3):
            model = self.gpt if attempt == 0 else self.gemini

            try:
                langfuse_context.update_current_observation(model=model, input=messages)

                response = self.client.chat.completions.parse(
                    model=model,
                    messages=messages,
                    response_format=LLMParsedResponse
                )

                parsed_obj = getattr(response.choices[0].message, 'parsed', None)
                content = parsed_obj.preprocessed_transcription if parsed_obj else response.choices[0].message.content

                if response.usage:
                    input_cost = float(response.usage.cost_details.get('upstream_inference_prompt_cost') or 0.0)
                    output_cost = float(response.usage.cost_details.get('upstream_inference_completions_cost') or 0.0)
                    total_cost = float(response.usage.cost or 0.0)
                    upstream_inference_cost = float(response.usage.cost_details.get('upstream_inference_cost') or 0.0)
                    cached_tokens = getattr(response.usage.prompt_tokens_details, 'cached_tokens', 0) or 0
                    reasoning_tokens = getattr(response.usage.completion_tokens_details, 'reasoning_tokens', 0) or 0

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
                            "is_byok": response.usage.is_byok,
                            "model_used": model
                        }
                    )

                    self.log(f"Model: {model} | Tokens: {response.usage.total_tokens} | Cost: ${total_cost:.8f}")

                return content

            except Exception as e:
                self.retries += 1
                self.log(f"Attempt {attempt + 1}/3 failed with {model}: {str(e)}")
                last_error = e

        raise LLMCallError(f"All 3 attempts failed: {str(last_error)}") from last_error