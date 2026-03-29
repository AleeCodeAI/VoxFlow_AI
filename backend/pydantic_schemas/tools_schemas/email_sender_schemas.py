from pydantic import BaseModel, Field

class Email(BaseModel):
    to: str = Field(description="The recipient's email")
    subject: str = Field(description="The subject of the email")
    processed_data: str = Field(description="The processed data that will be used to craft email")
    user_message: str = Field(description="The user's message along the email")
    sender: str = Field(description="Sender's name")