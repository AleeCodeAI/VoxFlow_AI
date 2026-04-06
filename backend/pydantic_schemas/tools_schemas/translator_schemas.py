from pydantic import BaseModel


class TranslationOutput(BaseModel):
    translated_data: str
