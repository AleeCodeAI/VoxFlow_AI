from fastapi import APIRouter, HTTPException
import logging
from pydantic_schemas import (
    EmailRequest,
    EmailResponse,
    TextExtractionRequest,
    TextExtractionData,
    TextExtractionResponse,
    TranslationRequest,
    TranslationResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/send-email", response_model=EmailResponse)
async def send_email(request: EmailRequest):
    try:
        logger.info(f"Sending email to: {request.to}")

        from tools.email_sender import EmailSender, Email

        email_sender = EmailSender()

        email_data = Email(
            to=request.to,
            subject=request.subject,
            processed_data=request.processed_data,
            user_message=request.user_message,
            sender=request.sender,
        )

        email_sender.send_email(email_data)

        logger.info(f"Email sent successfully to: {request.to}")

        return EmailResponse(
            status="success",
            message=f"Email sent successfully to {request.to}",
            email=request.to,
        )

    except Exception as e:
        logger.error(f"Error sending email: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Email sending failed",
                "detail": str(e),
            },
        )


@router.post("/extract-text", response_model=TextExtractionResponse)
async def extract_text(request: TextExtractionRequest):
    logger.info("Extracting keywords and keypoints from processed data")

    from tools.text_extractor import TextExtracter, ProcessedData

    text_extracter = TextExtracter()
    processed_data = ProcessedData(processed_data=request.processed_data)

    extraction_result = text_extracter.extract(processed_data)

    return TextExtractionResponse(
        status="success",
        message="Text extraction completed successfully",
        data=TextExtractionData(
            keywords=extraction_result.keywords,
            keypoints=extraction_result.keypoints,
        ),
    )


@router.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    try:
        logger.info(f"Translating text to language: {request.language}")

        from tools.translator import Translate

        translator = Translate()

        translation_result = translator.translate(
            language=request.language, data=request.processed_data
        )

        logger.info("Translation completed")

        return TranslationResponse(
            status="success",
            message=f"Text translated successfully to {request.language}",
            translated_data=translation_result.translated_data,
        )

    except Exception as e:
        logger.error(f"Error translating text: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Translation failed",
                "detail": str(e),
            },
        )