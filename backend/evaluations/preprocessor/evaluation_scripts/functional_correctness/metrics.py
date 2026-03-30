class Metrics:
    """Handles metric calculations for functional correctness evaluation"""

    def parse_logs(self, logs):
        """
        Extract metrics from preprocessor execution logs.
        """
        llm_retries = 0
        chunk_count = 0
        chunks_processed = 0

        for line in logs.split('\n'):
            if 'Attempt' in line and 'failed' in line:
                llm_retries += 1
            if 'Split transcription into' in line:
                chunk_count = int(line.split('into')[1].split('chunks')[0].strip())
            if 'Processing chunk' in line:
                chunks_processed += 1

        chunk_completeness = chunks_processed == chunk_count if chunk_count > 0 else True

        return {
            'llm_retries': llm_retries,
            'chunk_completeness': chunk_completeness,
        }