from configs import JUDGE_PROMPT
from typing import List, Dict


class PromptBuilder:
    """Builds the messages array for the AI Judge API call"""

    def __init__(self):
        self.prompt = JUDGE_PROMPT

    def build(self, transcription: str, preprocessed_transcription: str) -> List[Dict]:
        return [
            {"role": "system", "content": self.prompt.system_prompt},
            {"role": "user", "content": self.prompt.render_user(
                "default",
                transcription=transcription,
                preprocessed_transcription=preprocessed_transcription,
            )}
        ]