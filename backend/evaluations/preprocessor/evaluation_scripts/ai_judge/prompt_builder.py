from prompts import JUDGE_SYSTEM_PROMPT, JUDGE_USER_PROMPT
from typing import List, Dict


class PromptBuilder:
    """Builds the messages array for the AI Judge API call"""

    def __init__(self):
        self.system_prompt = JUDGE_SYSTEM_PROMPT
        self.user_prompt = JUDGE_USER_PROMPT

    def build(self, transcription: str, preprocessed_transcription: str) -> List[Dict]:
        system_message = {"role": "system", "content": self.system_prompt}
        user_content = self.user_prompt.format(
            transcription=transcription,
            preprocessed_transcription=preprocessed_transcription
        )
        user_message = {"role": "user", "content": user_content}
        return [system_message, user_message]