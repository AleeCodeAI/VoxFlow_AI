import logging
from datetime import datetime
from jiwer import wer, cer
from pydantic_schemas import NormalizedObject, LexicalMetrics
from utils.color import Logger
from .metrics import compute_ngram_similarity, determine_quality_label
from .normalizer import Normalizer
from .reporter import Reporter

logging.basicConfig(level=logging.INFO, format="%(message)s")


class LexicalEvaluator(Logger):
    """
    Evaluates transcription quality using lexical metrics.

    This class computes Word Error Rate (WER), Character Error Rate (CER),
    and n-gram similarity between transcriptions and references. It also
    assigns quality labels and generates comprehensive evaluation reports.
    """

    name = "lexical_evaluator"
    color = Logger.CYAN

    def __init__(self):
        """
        Initialize the LexicalEvaluator with output paths.

        Sets up output directory and file paths for storing evaluation
        results and summary reports.
        """
        self.normalizer = Normalizer()
        self.reporter = Reporter()
        self.log("Lexical evaluator initialized")

    def evaluate(
        self, normalized_pairs: list[NormalizedObject]
    ) -> list[LexicalMetrics]:
        """
        Evaluate all normalized transcription-reference pairs.

        Computes WER, CER, and n-gram similarity for each pair, assigns
        quality labels, and returns a list of evaluation results.

        Args:
            normalized_pairs (list[NormalizedObject]): List of normalized pairs to evaluate

        Returns:
            list[LexicalMetrics]: List of evaluation results with all metrics
        """
        self.log(f"Starting evaluation of {len(normalized_pairs)} transcription pairs")
        results = []

        for pair in normalized_pairs:
            transcription_text = pair.transcription
            reference_text = pair.reference

            word_error_rate = wer(reference_text, transcription_text)
            character_error_rate = cer(reference_text, transcription_text)
            ngram_score = compute_ngram_similarity(
                reference_text, transcription_text, ngram_size=2
            )
            quality_label = determine_quality_label(
                word_error_rate, character_error_rate, ngram_score
            )

            results.append(
                LexicalMetrics(
                    id=pair.id,
                    file_name=pair.file_name,
                    wer=word_error_rate,
                    cer=character_error_rate,
                    ngram=ngram_score,
                    quality_label=quality_label,
                    timestamp=datetime.now().isoformat(),
                )
            )
            self.log(
                f"Evaluated {pair.file_name}: WER={word_error_rate:.4f}, CER={character_error_rate:.4f}, N-gram={ngram_score:.4f}, Quality={quality_label}"
            )

        self.log(f"Evaluation complete: {len(results)} results generated")
        return results

    def run(self):
        """
        Run the complete lexical evaluation pipeline.

        This method runs the complete evaluation process:
        1. Loads transcriptions and references
        2. Normalizes the text data
        3. Evaluates transcription quality using lexical metrics
        4. Saves results to JSONL and generates a summary report
        """
        transcriptions = self.normalizer.load_transcriptions()
        references = self.normalizer.load_references()
        normalized_pairs = self.normalizer.normalize(transcriptions, references)

        lexical_results = self.evaluate(normalized_pairs)

        self.reporter.save_execution(lexical_results)
        self.reporter.generate_report(lexical_results)

        print(f"\nEvaluation complete! Processed {len(lexical_results)} files")
        print(f"Results saved to: {self.reporter.results_path}")
        print(f"Summary report saved to: {self.reporter.summary_path}")


if __name__ == "__main__":
    LexicalEvaluator().run()
