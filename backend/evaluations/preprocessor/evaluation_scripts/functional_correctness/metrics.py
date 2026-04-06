class Metrics:
    """Handles metric calculations for functional correctness evaluation"""

    def parse_logs(self, report):
        if report is None:
            return {
                'llm_retries': 0,
                'chunk_completeness': False,
            }
        
        chunk_completeness = report.chunks_processed == report.chunk_count if report.chunk_count > 0 else True

        return {
            'llm_retries': report.llm_retries,
            'chunk_completeness': chunk_completeness,
        }
