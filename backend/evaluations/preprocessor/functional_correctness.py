import os
import json
import sys
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field
import statistics
from color import Logger
import logging

sys.path.append(str(Path(__file__).parent.parent.parent))
from core.preprocessor import Preprocessor

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

class PreprocessorEvaluationResult(BaseModel):
    """
    Stores evaluation metrics for a single preprocessing execution.
    """
    id: str = Field(description="Unique identifier for the evaluation result")
    file_name: str = Field(description="Name of the file being evaluated")
    chunk_completeness: bool = Field(description="Whether all chunks were processed completely")
    llm_retries: int = Field(description="Number of retries made by the LLM during processing")
    output_existence: bool = Field(description="Indicates if the output file exists after processing")
    session_integrity: bool = Field(description="Indicates if the session data remains intact after processing")
    content_quality: float = Field(description="Quality score of the processed content")
    timestamp: datetime = Field(description="Timestamp of when the evaluation was performed")

class TranscriptionInput(BaseModel):
    """
    Represents a single transcription from the input JSONL file.
    """
    id: str
    name: str
    transcription: str
    timestamp: str

class EvaluationPipeline(Logger):
    """
    Runs functional correctness evaluation for the preprocessor script.
    Executes preprocessing on test data and collects metrics.
    """
    
    name = "PreprocessFunctionalEvaluation"
    color = Logger.WHITE
    
    def __init__(self, transcriptions_path, preprocessor_script_path, output_dir):
        """
        Initialize evaluation pipeline with paths and setup output directory.
        """
        self.transcriptions_path = transcriptions_path
        self.preprocessor_script_path = preprocessor_script_path
        self.output_dir = output_dir or str(Path(__file__).parent)
        self.results_file = os.path.join(self.output_dir, "execution_results.json")
        self.summary_file = os.path.join(self.output_dir, "evaluation_summary.md")
        os.makedirs(self.output_dir, exist_ok=True)
        self.log("Evaluation pipeline initialized")
        
    def load_transcriptions(self):
        """
        Load all transcription objects from JSONL file.
        """
        transcriptions = []
        with open(self.transcriptions_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    transcriptions.append(TranscriptionInput(**data))
        self.log(f"Loaded {len(transcriptions)} transcriptions")
        return transcriptions
    
    def parse_logs(self, logs):
        """
        Extract metrics from preprocessor execution logs.
        """
        llm_retries = 0
        chunk_count = 0
        chunks_processed = 0
        output_saved = False
        
        for line in logs.split('\n'):
            if 'Attempt' in line and 'failed' in line:
                llm_retries += 1
            if 'Split transcription into' in line:
                chunk_count = int(line.split('into')[1].split('chunks')[0].strip())
            if 'Processing chunk' in line:
                chunks_processed += 1
            if 'saved to' in line:
                output_saved = True
        
        chunk_completeness = chunks_processed == chunk_count if chunk_count > 0 else True
        
        return {
            'llm_retries': llm_retries,
            'chunk_completeness': chunk_completeness,
            'output_existence': output_saved
        }
    
    def get_quality_label(self, quality_score):
        """
        Return quality label based on content quality score.
        """
        if quality_score >= 1.10:
            return "BAD (Hallucination)"
        elif quality_score >= 1.00: 
            return "BAD (Stagnant)"
        elif quality_score >= 0.75:
            return "GOLDEN"
        elif quality_score >= 0.50:
            return "OKAY (Aggressive)"
        elif quality_score > 0.00:
            return "BAD (Info Loss)"
        else:
            return "CRITICAL (Empty)"
    
    def calculate_content_quality(self, original_text, preprocessed_text):
        """
        Calculate quality score based on text transformation metrics.
        """
        if not preprocessed_text:
            return 0.0
        
        original_len = len(original_text)
        preprocessed_len = len(preprocessed_text)
        
        if original_len == 0:
            return 0.0
        
        compression_ratio = preprocessed_len / original_len
        
        filler_words = ['um', 'uh', 'like', 'you know', 'basically', 'actually']
        filler_count = sum(preprocessed_text.lower().count(word) for word in filler_words)
        
        quality_score = min(1.0, compression_ratio) * (1 - min(0.5, filler_count / 100))
        
        return round(quality_score, 3)
    
    def verify_output_file(self, session_id):
        """
        Check if preprocessed output exists in the database file.
        """
        db_path = r"D:\Projects\audio_preprocessor\backend\databases\preprocessings.jsonl"
        
        if not os.path.exists(db_path):
            return False, None
        
        with open(db_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if data.get('id') == session_id:
                        return True, data.get('preprocessed_transcription', '')
        
        return False, None
    
    def run_preprocessor(self, transcription_obj):
        """
        Execute the preprocessor for a single transcription.
        """
        input_data = {
            "id": transcription_obj.id,
            "name": transcription_obj.name,
            "transcription": transcription_obj.transcription
        }
        
        try:
            preprocessor = Preprocessor()
            result = preprocessor.preprocess(input_data)
            preprocessor.langfuse.flush()
            
            return f"Processing completed for {transcription_obj.name}", True
        
        except Exception as e:
            return f"ERROR: {str(e)}", False
    
    def evaluate_single(self, transcription_obj):
        """
        Run evaluation for a single transcription and return result.
        """
        self.log(f"Evaluating {transcription_obj.name} (ID: {transcription_obj.id})")
        
        logs, success = self.run_preprocessor(transcription_obj)
        
        metrics = self.parse_logs(logs)
        
        output_exists, preprocessed_text = self.verify_output_file(transcription_obj.id)
        
        content_quality = 0.0
        if preprocessed_text:
            content_quality = self.calculate_content_quality(
                transcription_obj.transcription,
                preprocessed_text
            )
        
        session_integrity = success and output_exists
        
        result = PreprocessorEvaluationResult(
            id=transcription_obj.id,
            file_name=transcription_obj.name,
            chunk_completeness=metrics['chunk_completeness'],
            llm_retries=metrics['llm_retries'],
            output_existence=output_exists,
            session_integrity=session_integrity,
            content_quality=content_quality,
            timestamp=datetime.now()
        )
        
        self.log(f"Retries: {result.llm_retries} | Quality: {result.content_quality} | Success: {session_integrity}")
        
        return result
    
    def save_results(self, results):
        """
        Save all evaluation results to JSON file.
        """
        results_data = [result.model_dump(mode='json') for result in results]
        
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        self.log(f"Saved execution results to {self.results_file}")
    
    def generate_summary(self, results):
        """
        Generate markdown summary with statistics and insights.
        """
        total = len(results)
        successful = sum(1 for r in results if r.session_integrity)
        failed = total - successful
        
        avg_retries = statistics.mean([r.llm_retries for r in results])
        avg_quality = statistics.mean([r.content_quality for r in results])
        
        chunk_complete_count = sum(1 for r in results if r.chunk_completeness)
        output_exists_count = sum(1 for r in results if r.output_existence)
        
        summary = f"""# Preprocessor Evaluation Summary

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Overview

| Metric | Value |
|--------|-------|
| Total Executions | {total} |
| Successful | {successful} |
| Failed | {failed} |
| Success Rate | {(successful/total*100):.2f}% |

---

## Performance Metrics

| Metric | Average |
|--------|---------|
| LLM Retries | {avg_retries:.2f} |
| Content Quality Score | {avg_quality:.3f} |
| Chunk Completeness Rate | {(chunk_complete_count/total*100):.2f}% |
| Output Existence Rate | {(output_exists_count/total*100):.2f}% |

---

## Detailed Results

| File Name | ID | Retries | Quality | Label | Complete | Output | Status |
|-----------|-----|---------|---------|-------|----------|--------|--------|
"""
        
        for result in results:
            status = "✅ Pass" if result.session_integrity else "❌ Fail"
            complete = "✓" if result.chunk_completeness else "✗"
            output = "✓" if result.output_existence else "✗"
            quality_label = self.get_quality_label(result.content_quality)
            
            summary += f"| {result.file_name} | {result.id[:8]}... | {result.llm_retries} | {result.content_quality:.3f} | {quality_label} | {complete} | {output} | {status} |\n"
        
        summary += f"""
---

## Key Insights

- **Average LLM Retries:** {avg_retries:.2f} retries per execution
- **Quality Assessment:** Average content quality score is {avg_quality:.3f}
- **Reliability:** {(chunk_complete_count/total*100):.1f}% of executions completed all chunks
- **Data Persistence:** {(output_exists_count/total*100):.1f}% of outputs were successfully saved

---

## Quality Score Guide

| Value | Status | Meaning |
|-------|--------|---------|
| 1.10+ | **BAD** | Hallucination. The LLM added more than 10% new info that wasn't there. |
| 1.00 | **BAD** | Stagnant. The script ran but changed nothing (no cleaning happened). |
| 0.75 – 0.99 | **GOLDEN** | Ideal. This is where most high-quality cleanups land. |
| 0.50 – 0.74 | **OKAY** | Aggressive. Valid if the person was extremely repetitive or "wordy." |
| 0.10 – 0.49 | **BAD** | Information Loss. The LLM summarized or deleted whole sentences. |
| 0.00 | **CRITICAL** | Empty. Total failure. No text returned. |

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
        
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        self.log(f"Saved evaluation summary to {self.summary_file}")
    
    def run(self):
        """
        Execute the complete evaluation pipeline.
        """
        self.log("="*60)
        self.log("STARTING PREPROCESSOR FUNCTIONAL EVALUATION")
        self.log("="*60)
        
        transcriptions = self.load_transcriptions()
        
        results = []
        for transcription in transcriptions:
            result = self.evaluate_single(transcription)
            results.append(result)
        
        self.save_results(results)
        self.generate_summary(results)
        
        self.log("="*60)
        self.log("EVALUATION COMPLETE")
        self.log("="*60)
        self.log(f"Results: {self.results_file}")
        self.log(f"Summary: {self.summary_file}")

if __name__ == "__main__":
    pipeline = EvaluationPipeline(
        transcriptions_path=r"D:\Projects\audio_preprocessor\backend\evaluations\test_data\preprocessor\transcriptions_data.jsonl",
        preprocessor_script_path=r"D:\Projects\audio_preprocessor\backend\core\preprocessor.py",
        output_dir=None
    )
    
    pipeline.run()