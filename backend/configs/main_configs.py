from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[1]  

class MainSettings(BaseSettings):
    """Configuration for LLM-related settings."""
    
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        extra="ignore"
    )

    OPENROUTER_API_KEY: str
    OPENROUTER_URL: str

    LANGFUSE_SECRET_KEY: str
    LANGFUSE_PUBLIC_KEY: str
    LANGFUSE_HOST: str

    DEEPSEEK_MODEL: str = "deepseek/deepseek-r1"
    GPT_MODEL: str = "openai/gpt-4.1-nano"
    GEMINI_MODEL: str = "google/gemini-2.0-flash"

if __name__ == "__main__":
    config = MainSettings()
    print(config.dict())
    print(config.GPT_MODEL)