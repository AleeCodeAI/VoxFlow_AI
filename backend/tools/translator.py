from pydantic import BaseModel
from deep_translator import GoogleTranslator
from color import Logger
import logging 

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)


# --- Schemas ---
class TranslationInput(BaseModel):
    language: str 
    processed_data: str

class TranslationOutput(BaseModel):
    translated_data: str

class Translate(Logger):
    name = "Translator"
    color = Logger.WHITE

    def translate(self, language: str, data: str) -> TranslationOutput:
        try:
            self.log("Translating the data")
            
            # 1. Define chunk size (staying safe under the 5000 limit)
            CHUNK_SIZE = 4000 
            chunks = [data[i:i + CHUNK_SIZE] for i in range(0, len(data), CHUNK_SIZE)]
            
            translator = GoogleTranslator(source='auto', target=language)
            translated_parts = []

            # 2. Translate each chunk
            for chunk in chunks:
                translated_parts.append(translator.translate(chunk))
            
            # 3. Join them back together
            full_translation = " ".join(translated_parts)
            
            self.log("✅ Translation completed")
            return TranslationOutput(translated_data=full_translation)
            
        except Exception as e:
            self.log(f"❌ something wrong happened: {str(e)}")
            return TranslationOutput(translated_data=f"Translation Error: {str(e)}")
# --- Execution ---
if __name__ == "__main__":
    translator_service = Translate()
    
    # Example: Translating to Urdu ('ur')
    result = translator_service.translate(
        language="es", 
        data="This approach combines Pydantic for data validation and the googletrans library for the actual translation logic. Using Pydantic ensures that the data entering your translation class is structured correctly."
    )
    
    print(f"Result Object: {result}")
    print(f"Translated Text: {result.translated_data}")