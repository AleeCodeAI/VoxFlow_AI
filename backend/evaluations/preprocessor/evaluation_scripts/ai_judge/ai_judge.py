from pydantic_schemas import EvaluationError, Result
from utils.color import Logger
from datetime import datetime
from .evaluation_client import EvaluationClient
from .prompt_builder import PromptBuilder
from .storage import Storage
from .reporter import Reporter
import logging
import json
import os


logging.basicConfig(level=logging.INFO, format="%(message)s")


class AIJudge(Logger):
    """
    AI-powered evaluation system for assessing preprocessing quality.
    Compares original transcriptions with preprocessed versions and generates detailed metrics.
    """

    name = "AIJudge"
    color = Logger.MAGENTA

    def __init__(self):
        """Initialize the AI Judge with OpenAI client and file paths"""
        try:
            self.log("Initializing AI Judge...")
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            PREPROCESSOR_DIR = os.path.join(BASE_DIR, "..", "..")
            EVALUATIONS_DIR = os.path.join(BASE_DIR, "..", "..", "..")
            self.transcriptions_path = os.path.join(
                EVALUATIONS_DIR,
                "test_data",
                "preprocessor",
                "transcriptions_data.jsonl",
            )
            self.preprocessed_transcriptions_path = os.path.join(
                EVALUATIONS_DIR, "test_data", "preprocessor", "preprocessings.jsonl"
            )
            self.summary_path = os.path.join(
                PREPROCESSOR_DIR, "evaluation_results", "judge_evaluation_summary.md"
            )
            self.execution_path = os.path.join(
                PREPROCESSOR_DIR, "evaluation_databases", "judge_executions.jsonl"
            )

            self.client = EvaluationClient()
            self.prompt_builder = PromptBuilder()
            self.storage = Storage(self.execution_path)
            self.reporter = Reporter(self.summary_path)

            self.log(
                f"AI Judge initialized successfully with model: {self.client.model}"
            )
        except Exception as e:
            error = EvaluationError(
                error_type="InitializationError",
                message=str(e),
                timestamp=datetime.now().isoformat(),
                context={"stage": "initialization"},
            )
            self.log(f"Error initializing AI Judge: {error.message}")
            raise Exception(error.model_dump_json())

    def load(self):
        """Load transcriptions and preprocessed transcriptions from JSONL files with validation"""
        try:
            self.log("Loading transcription data...")
            transcriptions = []
            preprocessed_transcriptions = []

            with open(self.transcriptions_path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    transcription_obj = json.loads(line.strip())
                    transcriptions.append(transcription_obj)
                self.log(f"Loaded {len(transcriptions)} original transcriptions")

            with open(
                self.preprocessed_transcriptions_path, "r", encoding="utf-8"
            ) as f:
                for idx, line in enumerate(f):
                    preprocessed_obj = json.loads(line.strip())
                    preprocessed_transcriptions.append(preprocessed_obj)
                self.log(
                    f"Loaded {len(preprocessed_transcriptions)} preprocessed transcriptions"
                )

            self.log("Validating ID and name consistency...")
            mismatches = []
            for trans, prep in zip(transcriptions, preprocessed_transcriptions):
                trans_id = trans.get("id")
                prep_id = prep.get("id")
                trans_name = trans.get("name")
                prep_name = prep.get("name")

                if trans_id != prep_id or trans_name != prep_name:
                    mismatches.append(
                        {
                            "transcription": {"id": trans_id, "name": trans_name},
                            "preprocessed": {"id": prep_id, "name": prep_name},
                        }
                    )

            if mismatches:
                error = EvaluationError(
                    error_type="ValidationError",
                    message=f"Found {len(mismatches)} mismatches between transcription and preprocessed objects",
                    timestamp=datetime.now().isoformat(),
                    context={"mismatches": mismatches},
                )
                self.log(f"Validation failed: {error.message}")
                raise Exception(error.model_dump_json())

            self.log("Validation successful: All IDs and names match")
            return transcriptions, preprocessed_transcriptions

        except json.JSONDecodeError as e:
            error = EvaluationError(
                error_type="JSONDecodeError",
                message=f"Invalid JSON format: {str(e)}",
                timestamp=datetime.now().isoformat(),
                context={
                    "transcriptions_path": self.transcriptions_path,
                    "preprocessed_path": self.preprocessed_transcriptions_path,
                },
            )
            self.log(f"Error parsing JSON: {error.message}")
            raise Exception(error.model_dump_json())
        except Exception as e:
            if "ValidationError" in str(e):
                raise
            error = EvaluationError(
                error_type="DataLoadError",
                message=str(e),
                timestamp=datetime.now().isoformat(),
                context={
                    "transcriptions_path": self.transcriptions_path,
                    "preprocessed_path": self.preprocessed_transcriptions_path,
                },
            )
            self.log(f"Error loading data: {error.message}")
            raise Exception(error.model_dump_json())

    def evaluate(self):
        """Execute evaluation process for all transcription pairs"""
        try:
            self.log("Starting evaluation process...")
            transcriptions, preprocessed_transcriptions = self.load()

            if len(transcriptions) != len(preprocessed_transcriptions):
                raise ValueError(
                    f"Mismatch in data lengths: {len(transcriptions)} vs {len(preprocessed_transcriptions)}"
                )

            results = []
            total_pairs = len(transcriptions)

            for idx, (trans_obj, prep_obj) in enumerate(
                zip(transcriptions, preprocessed_transcriptions), 1
            ):
                try:
                    trans_id = trans_obj.get("id")
                    trans_name = trans_obj.get("name")
                    transcription = trans_obj.get("transcription")
                    preprocessed_transcription = prep_obj.get(
                        "preprocessed_transcription"
                    )

                    self.log(
                        f"Evaluating pair {idx}/{total_pairs} - ID: {trans_id}, File: {trans_name}"
                    )

                    messages = self.prompt_builder.build(
                        transcription, preprocessed_transcription
                    )
                    ai_result = self.client.call(messages)

                    result = Result(
                        id=trans_id,
                        file_name=trans_name,
                        meaning_preservation=ai_result.meaning_preservation,
                        information_loss=ai_result.information_loss,
                        preprocessing_quality=ai_result.preprocessing_quality,
                        hallucination=ai_result.hallucination,
                        confidence=ai_result.confidence,
                        reasoning=ai_result.reasoning,
                    )

                    write_mode = "w" if idx == 1 else "a"
                    self.storage.save_execution(result, mode=write_mode)
                    results.append(result)

                    self.log(
                        f"Evaluation {idx} completed - ID: {trans_id}, Quality: {result.preprocessing_quality}, Confidence: {result.confidence:.2f}"
                    )

                except Exception as e:
                    error = EvaluationError(
                        error_type="EvaluationError",
                        message=str(e),
                        timestamp=datetime.now().isoformat(),
                        context={
                            "pair_index": idx,
                            "total_pairs": total_pairs,
                            "id": trans_obj.get("id", "unknown"),
                        },
                    )
                    self.log(
                        f"Error evaluating pair {idx} (ID: {trans_obj.get('id', 'unknown')}): {error.message}"
                    )
                    continue

            self.log(f"Evaluation completed: {len(results)}/{total_pairs} successful")

            if results:
                self.reporter.generate_summary(results)

            return results

        except Exception as e:
            error = EvaluationError(
                error_type="EvaluationProcessError",
                message=str(e),
                timestamp=datetime.now().isoformat(),
                context={"stage": "main_evaluation"},
            )
            self.log(f"Critical error in evaluation process: {error.message}")
            raise Exception(error.model_dump_json())


if __name__ == "__main__":
    judge = AIJudge()
    result = judge.evaluate()
    print("================================================")
    print(f"Total Evaluations Completed: {len(result)}")
    print("================================================")
