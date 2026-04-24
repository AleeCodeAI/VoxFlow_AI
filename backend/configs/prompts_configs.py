from prompts import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_PROMPT,
    PREPROCESSOR_SYSTEM_PROMPT,
    PREPROCESSOR_USER_PROMPT_WITH_CONTEXT,
    PREPROCESSOR_USER_PROMPT_NO_CONTEXT,
    TEXT_EXTRACTOR_SYSTEM_PROMPT,
    TEXT_EXTRACTOR_USER_PROMPT,
)
from pydantic_schemas import Prompt


JUDGE_PROMPT = Prompt(
    name="judge_prompt",
    version="1.0",
    system_prompt=JUDGE_SYSTEM_PROMPT,
    user_templates={
        "default": JUDGE_USER_PROMPT,
    }
)

PREPROCESSOR_PROMPT = Prompt(
    name="preprocessor_prompt",
    version="1.0",
    system_prompt=PREPROCESSOR_SYSTEM_PROMPT,
    user_templates={
        "with_context": PREPROCESSOR_USER_PROMPT_WITH_CONTEXT,
        "no_context": PREPROCESSOR_USER_PROMPT_NO_CONTEXT,
    }
)

TEXT_EXTRACTOR_PROMPT = Prompt(
    name="text_extractor_prompt",
    version="1.0",
    system_prompt=TEXT_EXTRACTOR_SYSTEM_PROMPT,
    user_templates={
        "default": TEXT_EXTRACTOR_USER_PROMPT,
    }
)