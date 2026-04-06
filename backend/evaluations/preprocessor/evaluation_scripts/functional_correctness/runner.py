import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from core.preprocessor.preprocessor import Preprocessor


class Runner:
    """Executes the preprocessor for a single transcription"""

    def run_preprocessor(self, transcription_obj):
        """
        Execute the preprocessor for a single transcription.
        """
        input_data = {
            "id": transcription_obj.id,
            "name": transcription_obj.name,
            "transcription": transcription_obj.transcription,
        }

        try:
            preprocessor = Preprocessor()
            preprocessor.preprocess(input_data)
            preprocessor.observability.flush()

            return preprocessor.report, True

        except Exception as e:
            return f"ERROR: {str(e)}", False
