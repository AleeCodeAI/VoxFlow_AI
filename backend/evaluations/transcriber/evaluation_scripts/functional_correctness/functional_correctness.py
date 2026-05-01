import sys
import uuid
import logging
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from databases.transcriber_eval_query_repository import TranscriberEvalQueryRepository
from core.transcriber.transcriber import Transcriber
from utils.color import Logger
from pydantic_schemas import (
    TranscriptionEvaluationResult,
    EvaluationSummary,
    ErrorMessage,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")

EVAL_DIR = Path(__file__).parent.parent.parent
BACKEND_ROOT = EVAL_DIR.parent.parent

RESULTS_FILE = EVAL_DIR / "evaluation_databases" / "functional_evaluation_results.jsonl"
SUMMARY_FILE = EVAL_DIR / "evaluation_results" / "functional_evaluation_summary.md"
DB_FILE = BACKEND_ROOT / "databases" / "transcriptions.jsonl"
TEST_DATA_DIR = EVAL_DIR.parent / "test_data" / "transcriber"


class TranscriptionFunctionalEvaluator(Logger):
    name = "TranscriptionFunctionalEvaluator"
    color = Logger.YELLOW

    def __init__(self):
        self.transcriber = Transcriber()
        self.output_check = TranscriberEvalQueryRepository()

    def load_test_files(self):
        valid_files = [
            (f, True) for f in (TEST_DATA_DIR / "valids").glob("*") if f.is_file()
        ]
        invalid_files = [
            (f, False) for f in (TEST_DATA_DIR / "invalids").glob("*") if f.is_file()
        ]

        self.log(
            f"Loaded {len(valid_files)} valid and {len(invalid_files)} invalid test files"
        )
        return valid_files + invalid_files

    def check_input_validation(self, expected_valid, errors):
        has_format_error = any("Invalid file format" in e.error_message for e in errors)

        if not expected_valid:
            return has_format_error
        return not has_format_error
    
    def check_output_saved(self, result_id):
        if not result_id:
            return False
        return self.output_check.exists(result_id)

    def process_single_file(self, file_path, expected_valid):
        self.log(f"Testing {file_path.name} (expected_valid={expected_valid})")

        task_id = str(uuid.uuid4())
        errors = []
        result = None
        report = None

        try:
            result, report = self.transcriber.transcribe(str(file_path))
            self.log(f"Transcription completed for {file_path.name}")
        except Exception as exc:
            errors.append(
                ErrorMessage(
                    id=task_id,
                    file_name=file_path.name,
                    error_message=str(exc),
                    timestamp=datetime.now(),
                )
            )

        all_chunks_processed = (
            report.chunks_completed == report.total_chunks
            if report
            else expected_valid is False
        )

        eval_result = TranscriptionEvaluationResult(
            id=result.id if result else task_id,
            file_name=file_path.name,
            expected_valid=expected_valid,
            input_validation_passed=self.check_input_validation(expected_valid, errors),
            transcription_completed=result is not None,
            output_saved=self.check_output_saved(result.id if result else None),
            all_chunks_processed=all_chunks_processed,
            retries=report.retries if report else [],
            total_time_ms=report.total_time_ms if report else 0,
            errors=errors,
        )

        if eval_result.success:
            if eval_result.is_expected_rejection:
                self.log(
                    f"Result for {file_path.name}: EXPECTED REJECTION (correctly rejected invalid file)"
                )
            else:
                self.log(f"Result for {file_path.name}: SUCCESS")
        else:
            self.log(f"Result for {file_path.name}: UNEXPECTED FAILURE")

        return eval_result

    def generate_summary(self, results):
        total = len(results)
        valid_results = [r for r in results if r.expected_valid]
        invalid_results = [r for r in results if not r.expected_valid]
        valid_count = len(valid_results)
        invalid_count = len(invalid_results)

        overall_successes = sum(1 for r in results if r.success)
        valid_successes = sum(1 for r in valid_results if r.success)
        invalid_rejections = sum(1 for r in invalid_results if r.success)
        input_validation_correct = sum(1 for r in results if r.input_validation_passed)
        completed = sum(1 for r in valid_results if r.transcription_completed)
        saved = sum(1 for r in valid_results if r.output_saved)
        chunks_ok = sum(1 for r in valid_results if r.all_chunks_processed)

        total_retries = sum(r.retry_count for r in results)
        total_time = sum(r.total_time_ms for r in valid_results)
        unexpected_failures = sum(1 for r in results if r.is_unexpected_failure)
        expected_rejections = sum(1 for r in results if r.is_expected_rejection)
        total_errors = sum(len(r.errors) for r in results)

        summary = EvaluationSummary(
            total_files=total,
            valid_files_count=valid_count,
            invalid_files_count=invalid_count,
            overall_success_rate=overall_successes / total if total > 0 else 0,
            valid_files_success_rate=valid_successes / valid_count
            if valid_count > 0
            else 0,
            invalid_files_rejection_rate=invalid_rejections / invalid_count
            if invalid_count > 0
            else 0,
            input_validation_accuracy=input_validation_correct / total
            if total > 0
            else 0,
            completion_rate=completed / valid_count if valid_count > 0 else 0,
            output_save_rate=saved / valid_count if valid_count > 0 else 0,
            chunk_processing_rate=chunks_ok / valid_count if valid_count > 0 else 0,
            average_retries=total_retries / total if total > 0 else 0,
            average_time_ms=total_time / valid_count if valid_count > 0 else 0,
            unexpected_failures=unexpected_failures,
            expected_rejections=expected_rejections,
            total_errors=total_errors,
            timestamp=datetime.now(),
        )

        self.log(
            f"Summary: {overall_successes}/{total} overall success, {unexpected_failures} unexpected failures, {expected_rejections} expected rejections"
        )
        return summary

    def save_results(self, results):
        self.log(f"Saving {len(results)} results to {RESULTS_FILE}")

        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            for result in results:
                f.write(result.model_dump_json() + "\n")

        self.log(f"Results saved to {RESULTS_FILE}")

    def save_summary(self, summary):
        self.log(f"Generating summary report at {SUMMARY_FILE}")

        total = summary.total_files
        valid_n = summary.valid_files_count
        invalid_n = summary.invalid_files_count

        markdown = f"""# Transcription Functional Evaluation Summary

**Generated:** {summary.timestamp.strftime("%Y-%m-%d %H:%M:%S")}

## Overview

- **Total Files Tested:** {total}
- **Valid Files:** {valid_n}
- **Invalid Files:** {invalid_n}

## Performance Metrics

### Overall Results

| Metric | Count | Percentage |
|--------|-------|------------|
| Overall Success Rate | {int(summary.overall_success_rate * total)}/{total} | {summary.overall_success_rate * 100:.2f}% |
| Expected Rejections (Invalid Files) | {summary.expected_rejections}/{invalid_n} | {summary.invalid_files_rejection_rate * 100:.2f}% |
| Unexpected Failures | {summary.unexpected_failures}/{total} | {(summary.unexpected_failures / total * 100) if total > 0 else 0:.2f}% |

### Valid Files Performance

| Component | Count | Percentage |
|-----------|-------|------------|
| Valid Files Success Rate | {int(summary.valid_files_success_rate * valid_n)}/{valid_n} | {summary.valid_files_success_rate * 100:.2f}% |
| Input Validation Accuracy | {int(summary.input_validation_accuracy * total)}/{total} | {summary.input_validation_accuracy * 100:.2f}% |
| Transcription Completion | {int(summary.completion_rate * valid_n)}/{valid_n} | {summary.completion_rate * 100:.2f}% |
| Output Save Success | {int(summary.output_save_rate * valid_n)}/{valid_n} | {summary.output_save_rate * 100:.2f}% |
| Chunk Processing Success | {int(summary.chunk_processing_rate * valid_n)}/{valid_n} | {summary.chunk_processing_rate * 100:.2f}% |

### Invalid Files Performance

| Component | Count | Percentage |
|-----------|-------|------------|
| Correctly Rejected | {summary.expected_rejections}/{invalid_n} | {summary.invalid_files_rejection_rate * 100:.2f}% |
| Incorrectly Accepted | {invalid_n - summary.expected_rejections}/{invalid_n} | {((invalid_n - summary.expected_rejections) / invalid_n * 100) if invalid_n > 0 else 0:.2f}% |

## Error Analysis

- **Total Errors:** {summary.total_errors}
- **Average Retries per File:** {summary.average_retries:.2f}
- **Average Execution Time:** {summary.average_time_ms:.0f}ms

## Status

{"✓ All tests passed successfully" if summary.unexpected_failures == 0 else f"✗ {summary.unexpected_failures} unexpected failure(s) detected"}
"""

        with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
            f.write(markdown)

        self.log(f"Summary report saved to {SUMMARY_FILE}")

    def evaluate(self):
        self.log("Starting transcription functional evaluation")

        test_files = self.load_test_files()

        if not test_files:
            self.log("No test files found")
            return

        results = []
        for file_path, expected_valid in test_files:
            result = self.process_single_file(file_path, expected_valid)
            results.append(result)

        self.save_results(results)

        summary = self.generate_summary(results)
        self.save_summary(summary)

        self.log("Evaluation complete")


if __name__ == "__main__":
    evaluator = TranscriptionFunctionalEvaluator()
    evaluator.evaluate()
