import pytest


# ---------------------------------------------------------------------------
# Patch @observe as a no-op passthrough BEFORE any app modules are imported.
# Using autouse=False here; the patch is applied via the module-level
# pytestmark in each test file, OR you can rely on the autouse fixture below.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_observe(mocker):
    """
    Replace langfuse's @observe decorator with a transparent passthrough so
    decorated methods behave like plain functions during tests.
    No Langfuse connections are attempted.
    """
    mocker.patch(
        "langfuse.decorators.observe",
        side_effect=lambda *args, **kwargs: (
            # @observe("name", as_type="span")  →  called with args
            (lambda fn: fn) if args and callable(args[0]) is False
            # @observe  →  called directly on the function
            else (args[0] if args and callable(args[0]) else lambda fn: fn)
        ),
    )


# ---------------------------------------------------------------------------
# Shared MainSettings mock — avoids needing a real .env in CI
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_settings(mocker):
    """Return a mock MainSettings with sensible defaults."""
    settings = mocker.MagicMock()
    settings.OPENROUTER_API_KEY = "test-api-key"
    settings.OPENROUTER_URL = "https://openrouter.ai/api/v1"
    settings.GEMINI_MODEL = "google/gemini-pro"
    settings.GPT_MODEL = "openai/gpt-4o"
    settings.PREPROCESSOR_LLM_RETRIES = 2
    settings.LANGFUSE_SECRET_KEY = "sk-test"
    settings.LANGFUSE_PUBLIC_KEY = "pk-test"
    settings.LANGFUSE_HOST = "https://cloud.langfuse.com"
    return settings