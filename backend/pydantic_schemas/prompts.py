from pydantic import BaseModel, Field
from typing import Dict


class Prompt(BaseModel):
    name: str = Field(description="Prompt name")
    version: str = Field(description="Prompt version")
    system_prompt: str = Field(description="System message")
    user_templates: Dict[str, str] = Field(
        description="Different user prompt templates by name"
    )

    def render_user(self, template: str, **kwargs) -> str:
        if template not in self.user_templates:
            raise ValueError(f"Template '{template}' not found in prompt '{self.name}'")
        return self.user_templates[template].format(**kwargs)