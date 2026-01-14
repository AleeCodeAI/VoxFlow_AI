import os
from pydantic import BaseModel, Field
import requests
from color import Logger
import logging 

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

class Email(BaseModel):
    to: str = Field(description="The recipient's email")
    subject: str = Field(description="The subject of the email")
    processed_data: str = Field(description="The processed data that will be used to craft email")
    user_message: str = Field(description="The user's message along the email")
    sender: str = Field(description="Sender's name")

class EmailSender(Logger):
    name = "EmailSender"
    color = Logger.YELLOW

    def __init__(self):
        self.url = "http://localhost:5678/webhook/66531120-ed67-4a06-b6a2-0a690273e957"
        self.log("Initialized EmailSender")
        
    def send_to_n8n(self, data):
        """Send data to n8n webhook with error handling."""
        try:
            response = requests.post(self.url, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            if not result.get("success", True):
                self.log(f"Email send failed: {result}")
            
            return result
        except requests.exceptions.Timeout:
            error_msg = "Request to n8n webhook timed out"
            self.log(f"An error occured: {error_msg}")
            return {"success": False, "error": error_msg}
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP error occurred: {e}"
            self.log(error_msg)
            return {"success": False, "error": error_msg}
        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to send to n8n: {e}"
            self.log(error_msg,)
            return {"success": False, "error": error_msg}
        except ValueError as e:
            error_msg = f"Invalid JSON response: {e}"
            self.log(error_msg)
            return {"success": False, "error": error_msg}

    def send_email(self, email_data: Email):
        """Send email via n8n webhook."""
        self.log(f"Preparing email to {email_data.to} with subject: {email_data.subject}")
        
        email_body = f"""{email_data.user_message}

{email_data.processed_data}

Regards,
{email_data.sender}"""

        payload = {
            "to": email_data.to,
            "subject": email_data.subject,
            "body": email_body
        }
        
        status = self.send_to_n8n(payload)
        
        if status.get("status") == "success" or status.get("success") is True:
            self.log(f"Email sent successfully to {email_data.to}")
        else:
            self.log(f"Failed to send email to {email_data.to}: {status.get('error', 'Unknown error')}")
        
        return {"status": status["status"], "email": email_data.to}
        
if __name__ == "__main__":
    email_sender = EmailSender()

    email_meeting_summary = Email(
    to="aleetest16@gmail.com",
    subject="Action Items: Project Apollo Sync",
    processed_data="""
    - Deadline moved to Oct 15th.
    - Budget approved for additional cloud credits.
    - Marketing team to provide assets by Friday.
    """,
    user_message="Here is the summary from this morning's call. Please review the deadlines.",
    sender="Project Management Office")

    result = email_sender.send_email(email_meeting_summary)
    print(result)