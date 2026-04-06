from datetime import datetime
from evaluations.preprocessor.evaluation_scripts.functional_correctness.metrics import (
    Metrics,
)


class Reporter:
    """Generates markdown summary report from functional correctness evaluation results"""

    def __init__(self, summary_file: str):
        self.summary_file = summary_file
        self.metrics = Metrics()

    def generate_summary(self, results):
        """
        Generate markdown summary with statistics and insights.
        """
        import statistics

        total = len(results)
        successful = sum(1 for r in results if r.session_integrity)
        failed = total - successful

        avg_retries = statistics.mean([r.llm_retries for r in results])

        chunk_complete_count = sum(1 for r in results if r.chunk_completeness)
        output_exists_count = sum(1 for r in results if r.output_existence)

        summary = f"""# Preprocessor Evaluation Summary

    **Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    ---

    ## Overview

    | Metric | Value |
    |--------|-------|
    | Total Executions | {total} |
    | Successful | {successful} |
    | Failed | {failed} |
    | Success Rate | {(successful / total * 100):.2f}% |

    ---

    ## Performance Metrics

    | Metric | Average |
    |--------|---------|
    | LLM Retries | {avg_retries:.2f} |
    | Chunk Completeness Rate | {(chunk_complete_count / total * 100):.2f}% |
    | Output Existence Rate | {(output_exists_count / total * 100):.2f}% |

    ---

    ## Detailed Results

    | File Name | ID | Retries | Complete | Output | Status |
    |-----------|-----|---------|----------|--------|--------|
    """

        for result in results:
            status = "✅ Pass" if result.session_integrity else "❌ Fail"
            complete = "✓" if result.chunk_completeness else "✗"
            output = "✓" if result.output_existence else "✗"

            summary += f"| {result.file_name} | {result.id[:8]}... | {result.llm_retries} | {complete} | {output} | {status} |\n"

        summary += f"""
    ---

    ## Key Insights

    - **Average LLM Retries:** {avg_retries:.2f} retries per execution
    - **Reliability:** {(chunk_complete_count / total * 100):.1f}% of executions completed all chunks
    - **Data Persistence:** {(output_exists_count / total * 100):.1f}% of outputs were successfully saved

    ---

    ## Failure Analysis

    """

        failures = [r for r in results if not r.session_integrity]
        if failures:
            for failure in failures:
                summary += f"- **{failure.file_name}** (ID: {failure.id[:8]}...): "
                if not failure.output_existence:
                    summary += "Output not saved. "
                if not failure.chunk_completeness:
                    summary += "Incomplete chunk processing. "
                summary += f"Retries: {failure.llm_retries}\n"
        else:
            summary += "No failures detected. All executions completed successfully.\n"

        with open(self.summary_file, "w", encoding="utf-8") as f:
            f.write(summary)
