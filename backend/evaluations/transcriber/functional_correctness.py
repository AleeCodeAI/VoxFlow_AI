import sys
import re
import uuid
import logging
from pathlib import Path
from datetime import datetime
from io import StringIO
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).parent.parent.parent))

from core.transcriber import Transcriber
from color import Logger

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

class ErrorMessage(BaseModel):
    """Represents an error encountered during transcription"""
    id: str = Field(description="Unique identifier for the transcription task")
    file_name: str = Field(description="Name of the audio file")
    error_message: str = Field(description="Detailed error message")
    timestamp: datetime = Field(description="Timestamp of when the error occurred")


class TranscriptionEvaluationResult(BaseModel):
    """Evaluation result for a single test file"""
    id: str = Field(description="Unique identifier for the transcription task")
    file_name: str = Field(description="Name of the audio file")
    expected_valid: bool = Field(description="Whether the file was expected to be valid")
    input_validation_passed: bool = Field(description="Whether input validation behaved correctly")
    transcription_completed: bool = Field(description="Whether transcription completed successfully")
    output_saved: bool = Field(description="Whether output was saved to database")
    all_chunks_processed: bool = Field(description="Whether all audio chunks were processed")
    retry_count: int = Field(description="Number of retries attempted")
    errors: list[ErrorMessage] = Field(description="List of errors encountered")
    
    @property
    def success(self):
        """
        Overall success indicator
        For invalid files: success means correctly rejected
        For valid files: success means fully processed
        """
        if not self.expected_valid:
            return self.input_validation_passed
        
        return (self.input_validation_passed and 
                self.transcription_completed and 
                self.output_saved and 
                self.all_chunks_processed)
    
    @property
    def is_expected_rejection(self):
        """Invalid file that was correctly rejected"""
        return not self.expected_valid and self.input_validation_passed
    
    @property
    def is_unexpected_failure(self):
        """
        Something went wrong that shouldn't have
        Either invalid file accepted OR valid file failed processing
        """
        if not self.expected_valid:
            return not self.input_validation_passed
        else:
            return not self.success


class EvaluationSummary(BaseModel):
    """Aggregate metrics across all test cases"""
    total_files: int
    valid_files_count: int
    invalid_files_count: int
    
    overall_success_rate: float
    valid_files_success_rate: float
    invalid_files_rejection_rate: float
    
    input_validation_accuracy: float
    completion_rate: float
    output_save_rate: float
    chunk_processing_rate: float
    average_retries: float
    
    unexpected_failures: int
    expected_rejections: int
    total_errors: int
    timestamp: datetime


class TranscriptionFunctionalEvaluator(Logger):
    """
    Evaluates transcriber functionality by running test files and analyzing results
    """
    name = "TranscriptionFunctionalEvaluator"
    color = Logger.YELLOW

    def __init__(self):
        self.transcriber = Transcriber()
        self.base_path = Path(__file__).parent.parent / "test_data" / "transcriber"
        self.results_file = Path(__file__).parent / "functional_evaluation_results.jsonl"
        self.summary_file = Path(__file__).parent / "functional_evaluation_summary.md"
        self.db_path = Path(r"D:\Projects\audio_preprocessor\backend\databases\transcriptions.jsonl")
        self.captured_logs = ""

    def load_test_files(self):
        """
        Loads all test audio files from valid and invalid directories
        Returns list of tuples: (file_path, expected_valid)
        """
        valid_dir = self.base_path / "valids"
        invalid_dir = self.base_path / "invalids"
        
        valid_files = [(file_path, True) for file_path in valid_dir.glob("*") if file_path.is_file()]
        invalid_files = [(file_path, False) for file_path in invalid_dir.glob("*") if file_path.is_file()]
        
        self.log(f"Loaded {len(valid_files)} valid and {len(invalid_files)} invalid test files")
        return valid_files + invalid_files

    def check_input_validation(self, expected_valid, error_messages):
        """
        Validates that the transcriber correctly accepted or rejected the input file
        For invalid files: returns True if format error was found (correctly rejected)
        For valid files: returns True if no format errors occurred (correctly accepted)
        """
        has_format_error = any("Invalid file format" in err.error_message for err in error_messages)
        
        if not expected_valid:
            return has_format_error
        return not has_format_error

    def check_output_saved(self, transcription_id):
        """
        Verifies that transcription was saved to the database
        """
        if not transcription_id or not self.db_path.exists():
            return False
        
        with open(self.db_path, 'r', encoding='utf-8') as db_file:
            return any(transcription_id in line for line in db_file)

    def check_chunk_processing(self):
        """
        Verifies all audio chunks were successfully processed by comparing split vs completed counts
        """
        split_match = re.search(r"Audio split into (\d+) valid chunks", self.captured_logs)
        if not split_match:
            return False
        
        expected_chunks = int(split_match.group(1))
        completed_chunks = len(re.findall(r"Completed chunk \d+/\d+", self.captured_logs))
        
        return expected_chunks == completed_chunks

    def count_retries(self):
        """
        Counts retry attempts from captured logs
        """
        return len(re.findall(r"Retry \d+/\d+", self.captured_logs))

    def extract_errors_from_logs(self, task_id, file_name):
        """
        Extracts error information from captured logs
        """
        errors = []
        error_patterns = [
            r"Retry \d+/\d+.*",
            r"Error in chunk \d+.*",
            r"Failed to.*",
            r"WARNING:.*"
        ]
        
        for pattern in error_patterns:
            for match in re.findall(pattern, self.captured_logs):
                errors.append(ErrorMessage(
                    id=task_id,
                    file_name=file_name,
                    error_message=match.strip(),
                    timestamp=datetime.now()
                ))
        
        return errors

    def process_single_file(self, file_path, expected_valid):
        """
        Processes a single test file and returns evaluation result
        """
        self.log(f"Testing {file_path.name} (expected_valid={expected_valid})")
        
        log_stream = StringIO()
        log_handler = logging.StreamHandler(log_stream)
        logging.getLogger().addHandler(log_handler)
        
        task_id = str(uuid.uuid4())
        errors = []
        transcription = None

        try:
            transcription = self.transcriber.transcribe(str(file_path))
            self.log(f"Transcription completed for {file_path.name}")
        except Exception as exc:
            error_msg = f"CRITICAL: {str(exc)}"
            self.log(f"Transcription failed: {error_msg}")
            errors.append(ErrorMessage(
                id=task_id,
                file_name=file_path.name,
                error_message=error_msg,
                timestamp=datetime.now()
            ))

        self.captured_logs = log_stream.getvalue()
        logging.getLogger().removeHandler(log_handler)

        errors.extend(self.extract_errors_from_logs(task_id, file_path.name))

        result = TranscriptionEvaluationResult(
            id=task_id,
            file_name=file_path.name,
            expected_valid=expected_valid,
            input_validation_passed=self.check_input_validation(expected_valid, errors),
            transcription_completed=transcription is not None,
            output_saved=self.check_output_saved(transcription.id if transcription else None),
            all_chunks_processed=self.check_chunk_processing() if expected_valid else True,
            retry_count=self.count_retries(),
            errors=errors
        )
        
        if result.success:
            if result.is_expected_rejection:
                self.log(f"Result for {file_path.name}: EXPECTED REJECTION (correctly rejected invalid file)")
            else:
                self.log(f"Result for {file_path.name}: SUCCESS")
        else:
            self.log(f"Result for {file_path.name}: UNEXPECTED FAILURE")
        
        return result

    def generate_summary(self, results):
        """
        Generates aggregate metrics from evaluation results
        """
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
        unexpected_failures = sum(1 for r in results if r.is_unexpected_failure)
        expected_rejections = sum(1 for r in results if r.is_expected_rejection)
        total_errors = sum(len(r.errors) for r in results)
        
        summary = EvaluationSummary(
            total_files=total,
            valid_files_count=valid_count,
            invalid_files_count=invalid_count,
            overall_success_rate=overall_successes / total if total > 0 else 0,
            valid_files_success_rate=valid_successes / valid_count if valid_count > 0 else 0,
            invalid_files_rejection_rate=invalid_rejections / invalid_count if invalid_count > 0 else 0,
            input_validation_accuracy=input_validation_correct / total if total > 0 else 0,
            completion_rate=completed / valid_count if valid_count > 0 else 0,
            output_save_rate=saved / valid_count if valid_count > 0 else 0,
            chunk_processing_rate=chunks_ok / valid_count if valid_count > 0 else 0,
            average_retries=total_retries / total if total > 0 else 0,
            unexpected_failures=unexpected_failures,
            expected_rejections=expected_rejections,
            total_errors=total_errors,
            timestamp=datetime.now()
        )
        
        self.log(f"Summary: {overall_successes}/{total} overall success, {unexpected_failures} unexpected failures, {expected_rejections} expected rejections")
        return summary

    def save_results(self, results):
        """
        Saves evaluation results to JSONL file, overwriting previous data
        """
        self.log(f"Saving {len(results)} results to {self.results_file}")
        
        with open(self.results_file, "w", encoding="utf-8") as output_file:
            for result in results:
                output_file.write(result.model_dump_json() + "\n")
        
        self.log(f"Results saved to {self.results_file}")

    def save_summary(self, summary):
        """
        Saves evaluation summary as a formatted markdown file
        """
        self.log(f"Generating summary report at {self.summary_file}")
        
        markdown_content = f"""# Transcription Evaluation Summary

**Generated:** {summary.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

## Overview

- **Total Files Tested:** {summary.total_files}
- **Valid Files:** {summary.valid_files_count}
- **Invalid Files:** {summary.invalid_files_count}

## Performance Metrics

### Overall Results

| Metric | Count | Percentage |
|--------|-------|------------|
| Overall Success Rate | {int(summary.overall_success_rate * summary.total_files)}/{summary.total_files} | {summary.overall_success_rate * 100:.2f}% |
| Expected Rejections (Invalid Files) | {summary.expected_rejections}/{summary.invalid_files_count} | {summary.invalid_files_rejection_rate * 100:.2f}% |
| Unexpected Failures | {summary.unexpected_failures}/{summary.total_files} | {(summary.unexpected_failures / summary.total_files * 100) if summary.total_files > 0 else 0:.2f}% |

### Valid Files Performance

| Component | Count | Percentage |
|-----------|-------|------------|
| Valid Files Success Rate | {int(summary.valid_files_success_rate * summary.valid_files_count)}/{summary.valid_files_count} | {summary.valid_files_success_rate * 100:.2f}% |
| Input Validation Accuracy | {int(summary.input_validation_accuracy * summary.total_files)}/{summary.total_files} | {summary.input_validation_accuracy * 100:.2f}% |
| Transcription Completion | {int(summary.completion_rate * summary.valid_files_count)}/{summary.valid_files_count} | {summary.completion_rate * 100:.2f}% |
| Output Save Success | {int(summary.output_save_rate * summary.valid_files_count)}/{summary.valid_files_count} | {summary.output_save_rate * 100:.2f}% |
| Chunk Processing Success | {int(summary.chunk_processing_rate * summary.valid_files_count)}/{summary.valid_files_count} | {summary.chunk_processing_rate * 100:.2f}% |

### Invalid Files Performance

| Component | Count | Percentage |
|-----------|-------|------------|
| Correctly Rejected | {summary.expected_rejections}/{summary.invalid_files_count} | {summary.invalid_files_rejection_rate * 100:.2f}% |
| Incorrectly Accepted | {summary.invalid_files_count - summary.expected_rejections}/{summary.invalid_files_count} | {((summary.invalid_files_count - summary.expected_rejections) / summary.invalid_files_count * 100) if summary.invalid_files_count > 0 else 0:.2f}% |

## Error Analysis

- **Total Errors:** {summary.total_errors}
- **Average Retries per File:** {summary.average_retries:.2f}

## Status

{'✓ All tests passed successfully' if summary.unexpected_failures == 0 else f'✗ {summary.unexpected_failures} unexpected failure(s) detected'}
"""
        
        with open(self.summary_file, "w", encoding="utf-8") as summary_file:
            summary_file.write(markdown_content)
        
        self.log(f"Summary report saved to {self.summary_file}")

    def evaluate(self):
        """
        Runs full evaluation pipeline on all test files
        """
        self.log("Starting transcription evaluation")
        
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