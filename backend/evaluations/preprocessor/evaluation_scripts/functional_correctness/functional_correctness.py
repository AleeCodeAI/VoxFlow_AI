import os
import json
import logging
from datetime import datetime
from pydantic_schemas import PreprocessorEvaluationResult, TranscriptionInput
from utils.color import Logger
from evaluations.preprocessor.evaluation_scripts.functional_correctness.metrics import (
    Metrics,
)
from evaluations.preprocessor.evaluation_scripts.functional_correctness.verifier import (
    Verifier,
)
from evaluations.preprocessor.evaluation_scripts.functional_correctness.runner import (
    Runner,
)
from evaluations.preprocessor.evaluation_scripts.functional_correctness.storage import (
    Storage,
)
from evaluations.preprocessor.evaluation_scripts.functional_correctness.reporter import (
    Reporter,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")


class EvaluationPipeline(Logger):
    """
    Runs functional correctness evaluation for the preprocessor script.
    Executes preprocessing on test data and collects metrics.
    """

    name = "PreprocessFunctionalEvaluation"
    color = Logger.WHITE

    def __init__(self):
        """
        Initialize evaluation pipeline with paths and setup output directory.
        """
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        PREPROCESSOR_DIR = os.path.join(BASE_DIR, "..", "..")
        EVALUATIONS_DIR = os.path.join(BASE_DIR, "..", "..", "..")

        self.transcriptions_path = os.path.join(
            EVALUATIONS_DIR, "test_data", "preprocessor", "transcriptions_data.jsonl"
        )
        output_dir = os.path.join(PREPROCESSOR_DIR, "evaluation_databases")
        summary_dir = os.path.join(PREPROCESSOR_DIR, "evaluation_results")

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(summary_dir, exist_ok=True)

        results_file = os.path.join(output_dir, "functional_executions.json")
        summary_file = os.path.join(summary_dir, "functional_evaluation_summary.md")

        self.metrics = Metrics()
        self.verifier = Verifier()
        self.runner = Runner()
        self.storage = Storage(results_file)
        self.reporter = Reporter(summary_file)

        self.log("Evaluation pipeline initialized")

    def load_transcriptions(self):
        """
        Load all transcription objects from JSONL file.
        """
        transcriptions = []
        with open(self.transcriptions_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    transcriptions.append(TranscriptionInput(**data))
        self.log(f"Loaded {len(transcriptions)} transcriptions")
        return transcriptions

    def evaluate_single(self, transcription_obj):
        """
        Run evaluation for a single transcription and return result.
        """
        self.log(f"Evaluating {transcription_obj.name} (ID: {transcription_obj.id})")

        report, success = self.runner.run_preprocessor(transcription_obj)

        metrics = self.metrics.parse_logs(report)

        output_exists, _ = self.verifier.verify_output_file(transcription_obj.id)

        session_integrity = success and output_exists

        result = PreprocessorEvaluationResult(
            id=transcription_obj.id,
            file_name=transcription_obj.name,
            chunk_completeness=metrics["chunk_completeness"],
            llm_retries=metrics["llm_retries"],
            output_existence=output_exists,
            session_integrity=session_integrity,
            timestamp=datetime.now(),
        )

        self.log(f"Retries: {result.llm_retries} | Success: {session_integrity}")

        return result

    def evaluate(self):
        """
        Execute the complete evaluation pipeline.
        """
        self.log("=" * 60)
        self.log("STARTING PREPROCESSOR FUNCTIONAL EVALUATION")
        self.log("=" * 60)

        transcriptions = self.load_transcriptions()

        results = []
        for transcription in transcriptions:
            result = self.evaluate_single(transcription)
            results.append(result)

        self.storage.save_results(results)
        self.reporter.generate_summary(results)

        self.log("=" * 60)
        self.log("EVALUATION COMPLETE")
        self.log("=" * 60)


if __name__ == "__main__":
    pipeline = EvaluationPipeline()
    pipeline.evaluate()
