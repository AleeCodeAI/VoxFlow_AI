import jsonlines
from datetime import datetime
from pydantic_schemas import LexicalMetrics
from utils.color import Logger
from pathlib import Path


class Reporter(Logger):
    """
    Handles output for lexical evaluation results.

    This class saves evaluation results to a JSONL file and generates
    a comprehensive Markdown summary report.
    """

    name = "lexical_evaluator"
    color = Logger.CYAN

    def __init__(self):
        """
        Initialize the Reporter with output paths.

        Sets up output directory and file paths for storing evaluation
        results and summary reports.
        """

        EVAL_ROOT = Path(__file__).parent.parent.parent
        self.output_directory = EVAL_ROOT / "evaluation_databases"
        self.results_path = self.output_directory / "lexical_evaluations_result.jsonl"
        self.summary_path = (
            EVAL_ROOT / "evaluation_results" / "lexical_evaluation_summary.md"
        )
        self.log("Lexical evaluator initialized")

    def save_execution(self, results: list[LexicalMetrics]):
        """
        Save evaluation results to a JSONL file.

        Creates the output directory if it doesn't exist and writes all
        evaluation results to a JSONL file for further analysis.

        Args:
            results (list[LexicalMetrics]): List of evaluation results to save
        """
        self.log(f"Saving results to: {self.results_path}")
        self.output_directory.mkdir(parents=True, exist_ok=True)
        with jsonlines.open(self.results_path, mode="w") as writer:
            for result in results:
                writer.write(result.dict())
        self.log(f"Successfully saved {len(results)} evaluation results")

    def generate_report(self, results: list[LexicalMetrics]):
        """
        Generate a comprehensive Markdown summary report.

        Creates a detailed report including average metrics, performance insights,
        best/worst performing files, and a complete table of all results sorted
        by WER. The report is saved as a Markdown file.

        Args:
            results (list[LexicalMetrics]): List of evaluation results to summarize
        """
        self.log(f"Generating summary report at: {self.summary_path}")

        if not results:
            self.log("No results to generate report")
            return

        total_count = len(results)
        average_wer = sum(result.wer for result in results) / total_count
        average_cer = sum(result.cer for result in results) / total_count
        average_ngram = sum(result.ngram for result in results) / total_count

        min_wer = min(result.wer for result in results)
        max_wer = max(result.wer for result in results)
        min_cer = min(result.cer for result in results)
        max_cer = max(result.cer for result in results)
        min_ngram = min(result.ngram for result in results)
        max_ngram = max(result.ngram for result in results)

        best_wer_file = min(results, key=lambda x: x.wer).file_name
        worst_wer_file = max(results, key=lambda x: x.wer).file_name

        sorted_results = sorted(results, key=lambda x: x.wer)

        report_content = f"""# Lexical Evaluation Summary Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Total Evaluations:** {total_count}

## Average Metrics

| Metric | Average | Min | Max |
|--------|---------|-----|-----|
| Word Error Rate (WER) | {average_wer:.4f} | {min_wer:.4f} | {max_wer:.4f} |
| Character Error Rate (CER) | {average_cer:.4f} | {min_cer:.4f} | {max_cer:.4f} |
| N-gram Similarity | {average_ngram:.4f} | {min_ngram:.4f} | {max_ngram:.4f} |

## Performance Insights

- **Best Performing File (Lowest WER):** {best_wer_file} (WER: {min_wer:.4f})
- **Worst Performing File (Highest WER):** {worst_wer_file} (WER: {max_wer:.4f})
- **WER Range:** {max_wer - min_wer:.4f}
- **CER Range:** {max_cer - min_cer:.4f}

## Detailed Results

| File Name | WER | CER | N-gram | Quality | Timestamp |
|-----------|-----|-----|--------|--------|-----------|
"""

        for result in sorted_results:
            report_content += f"| {result.file_name} | {result.wer:.4f} | {result.cer:.4f} | {result.ngram:.4f} | {result.quality_label} | {result.timestamp} |\n"

        with open(self.summary_path, "w", encoding="utf-8") as file:
            file.write(report_content)

        self.log(f"Summary report generated successfully with {total_count} entries")
