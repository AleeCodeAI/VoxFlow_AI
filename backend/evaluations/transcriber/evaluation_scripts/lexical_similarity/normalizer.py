import jsonlines
from pydantic_schemas import NormalizedObject
from jiwer import Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip
from utils.color import Logger
from pathlib import Path


class Normalizer(Logger):
    """
    Normalizes transcription and reference texts for fair comparison.

    This class loads transcription and reference data, then applies text
    normalization (lowercase conversion, punctuation removal, whitespace
    normalization) to prepare the texts for lexical evaluation.
    """

    name = "lexical_normalizer"
    color = Logger.MAGENTA

    def __init__(self):
        """
        Initialize the Normalizer with file paths and preprocessing pipeline.

        Sets up the jiwer text normalization pipeline consisting of lowercase
        conversion, punctuation removal, multiple space removal, and stripping.
        """
        EVAL_ROOT = Path(__file__).parent.parent.parent
        self.transcriptions_path = (
            EVAL_ROOT / "evaluation_databases" / "transcriptions_data.jsonl"
        )
        self.reference_path = (
            EVAL_ROOT / "evaluation_databases" / "transcriptions_reference_data.jsonl"
        )
        self.normalizer = Compose(
            [ToLowerCase(), RemovePunctuation(), RemoveMultipleSpaces(), Strip()]
        )
        self.log("Normalizer initialized with text preprocessing pipeline")

    def load_transcriptions(self):
        """
        Load transcription data from JSONL file.

        Returns:
            list: List of transcription objects loaded from the JSONL file
        """
        self.log(f"Loading transcriptions from: {self.transcriptions_path}")
        with jsonlines.open(self.transcriptions_path) as reader:
            transcriptions = [obj for obj in reader]
        self.log(f"Successfully loaded {len(transcriptions)} transcriptions")
        return transcriptions

    def load_references(self):
        """
        Load reference transcription data from JSONL file.

        Returns:
            list: List of reference objects loaded from the JSONL file
        """
        self.log(f"Loading references from: {self.reference_path}")
        with jsonlines.open(self.reference_path) as reader:
            references = [obj for obj in reader]
        self.log(f"Successfully loaded {len(references)} references")
        return references

    def normalize(self, transcriptions, references) -> list[NormalizedObject]:
        """
        Normalize transcription-reference pairs for evaluation.

        Matches transcriptions with their corresponding references by file name,
        then applies text normalization to both texts. Unmatched transcriptions
        are skipped with a warning.

        Args:
            transcriptions (list): List of transcription objects
            references (list): List of reference objects

        Returns:
            list[NormalizedObject]: List of normalized transcription-reference pairs
        """
        self.log("Starting normalization process")
        normalized_pairs: list[NormalizedObject] = []
        reference_by_name = {reference["name"]: reference for reference in references}

        for transcription_obj in transcriptions:
            file_name = transcription_obj["name"]
            if file_name not in reference_by_name:
                self.log(f"No reference found for transcription: {file_name}")
                continue

            reference_obj = reference_by_name[file_name]
            raw_transcription = transcription_obj["transcription"]
            raw_reference = reference_obj["transcription"]

            normalized_transcription = self.normalizer(raw_transcription)
            normalized_reference = self.normalizer(raw_reference)

            normalized_pairs.append(
                NormalizedObject(
                    id=transcription_obj["id"],
                    file_name=file_name,
                    transcription=normalized_transcription,
                    reference=normalized_reference,
                )
            )

        self.log(f"Normalization complete: {len(normalized_pairs)} pairs processed")
        return normalized_pairs
