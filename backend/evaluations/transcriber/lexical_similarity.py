import jsonlines
from datetime import datetime
from pydantic import BaseModel, Field
from jiwer import Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip, wer, cer
from color import Logger
from pathlib import Path
import logging 

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

class NormalizedObject(BaseModel):
    id: str = Field(description="Unique identifier for the transcription")
    file_name: str = Field(description="Name of the audio file")
    transcription: str = Field(description="Normalized transcription text")
    reference: str = Field(description="Normalized reference text")


class LexicalMetrics(BaseModel):
    id: str = Field(description="Unique identifier for the transcription evaluation")
    file_name: str = Field(description="Name of the audio file evaluated")
    wer: float = Field(description="Word Error Rate")
    cer: float = Field(description="Character Error Rate")
    ngram: float = Field(description="N-gram Similarity Score")
    quality_label: str = Field(description="Quality Label: OK / Acceptable / Bad")
    timestamp: str = Field(description="Timestamp of the evaluation")


class Normalizer(Logger):
    name = "lexical_normalizer"
    color = Logger.MAGENTA

    def __init__(self):
        self.transcriptions_path = r"D:\Projects\audio_preprocessor\backend\evaluations\transcriber\transcriptions_data.jsonl"
        self.reference_path = r"D:\Projects\audio_preprocessor\backend\evaluations\transcriber\transcriptions_reference_data.jsonl"
        self.normalizer = Compose([
            ToLowerCase(),
            RemovePunctuation(),
            RemoveMultipleSpaces(),
            Strip()
        ])
        self.log("Normalizer initialized with text preprocessing pipeline")

    def load_transcriptions(self):
        self.log(f"Loading transcriptions from: {self.transcriptions_path}")
        with jsonlines.open(self.transcriptions_path) as reader:
            transcriptions = [obj for obj in reader]
        self.log(f"Successfully loaded {len(transcriptions)} transcriptions")
        return transcriptions

    def load_references(self):
        self.log(f"Loading references from: {self.reference_path}")
        with jsonlines.open(self.reference_path) as reader:
            references = [obj for obj in reader]
        self.log(f"Successfully loaded {len(references)} references")
        return references

    def normalize(self, transcriptions, references) -> list[NormalizedObject]:
        self.log("Starting normalization process")
        normalized_pairs: list[NormalizedObject] = []
        reference_by_name = {reference["name"]: reference for reference in references}
        
        for transcription_obj in transcriptions:
            file_name = transcription_obj["name"]
            if file_name not in reference_by_name:
                self.log(f"No reference found for transcription: {file_name}")
                continue
            
            reference_obj = reference_by_name[file_name]
            raw_transcription = transcription_obj["transcription"]
            raw_reference = reference_obj["transcription"]
            
            normalized_transcription = self.normalizer(raw_transcription)
            normalized_reference = self.normalizer(raw_reference)
            
            normalized_pairs.append(
                NormalizedObject(
                    id=transcription_obj["id"],
                    file_name=file_name,
                    transcription=normalized_transcription,
                    reference=normalized_reference
                )
            )
        
        self.log(f"Normalization complete: {len(normalized_pairs)} pairs processed")
        return normalized_pairs


class LexicalEvaluator(Logger):
    name = "lexical_evaluator"
    color = Logger.CYAN

    def __init__(self):
        self.output_directory = Path(r"D:\Projects\audio_preprocessor\backend\evaluations\transcriber")
        self.results_path = self.output_directory / "lexical_evaluations_result.jsonl"
        self.summary_path = self.output_directory / "lexical_evaluation_summary.md"
        self.log("Lexical evaluator initialized")

    def compute_ngram_similarity(self, reference: str, hypothesis: str, ngram_size=2):
        reference_tokens = reference.split()
        hypothesis_tokens = hypothesis.split()
        
        if len(reference_tokens) < ngram_size or len(hypothesis_tokens) < ngram_size:
            return 0.0
        
        reference_ngrams = set(tuple(reference_tokens[i:i+ngram_size]) for i in range(len(reference_tokens)-ngram_size+1))
        hypothesis_ngrams = set(tuple(hypothesis_tokens[i:i+ngram_size]) for i in range(len(hypothesis_tokens)-ngram_size+1))
        
        intersection = reference_ngrams.intersection(hypothesis_ngrams)
        union = reference_ngrams.union(hypothesis_ngrams)
        
        return len(intersection)/len(union) if union else 0.0

    def determine_quality_label(self, wer_score, cer_score, ngram_score):
        if wer_score <= 0.10 and cer_score <= 0.05 and ngram_score >= 0.80:
            return "OK"
        elif wer_score <= 0.25 and cer_score <= 0.15 and ngram_score >= 0.60:
            return "Acceptable"
        else:
            return "Bad"

    def evaluate(self, normalized_pairs: list[NormalizedObject]) -> list[LexicalMetrics]:
        self.log(f"Starting evaluation of {len(normalized_pairs)} transcription pairs")
        results = []
        
        for pair in normalized_pairs:
            transcription_text = pair.transcription
            reference_text = pair.reference
            
            word_error_rate = wer(reference_text, transcription_text)
            character_error_rate = cer(reference_text, transcription_text)
            ngram_score = self.compute_ngram_similarity(reference_text, transcription_text, ngram_size=2)
            quality_label = self.determine_quality_label(word_error_rate, character_error_rate, ngram_score)
            
            results.append(
                LexicalMetrics(
                    id=pair.id,
                    file_name=pair.file_name,
                    wer=word_error_rate,
                    cer=character_error_rate,
                    ngram=ngram_score,
                    quality_label=quality_label,
                    timestamp=datetime.now().isoformat()
                )
            )
            self.log(f"Evaluated {pair.file_name}: WER={word_error_rate:.4f}, CER={character_error_rate:.4f}, N-gram={ngram_score:.4f}, Quality={quality_label}")
        
        self.log(f"Evaluation complete: {len(results)} results generated")
        return results

    def save_execution(self, results: list[LexicalMetrics]):
        self.log(f"Saving results to: {self.results_path}")
        self.output_directory.mkdir(parents=True, exist_ok=True)
        with jsonlines.open(self.results_path, mode='w') as writer:
            for result in results:
                writer.write(result.dict())
        self.log(f"Successfully saved {len(results)} evaluation results")

    def generate_report(self, results: list[LexicalMetrics]):
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
        
        with open(self.summary_path, 'w', encoding='utf-8') as file:
            file.write(report_content)
        
        self.log(f"Summary report generated successfully with {total_count} entries")


if __name__ == "__main__":
    normalizer = Normalizer()
    transcriptions = normalizer.load_transcriptions()
    references = normalizer.load_references()
    normalized_pairs = normalizer.normalize(transcriptions, references)
    
    evaluator = LexicalEvaluator()
    lexical_results = evaluator.evaluate(normalized_pairs)
    evaluator.save_execution(lexical_results)
    evaluator.generate_report(lexical_results)
    
    print(f"\nEvaluation complete! Processed {len(lexical_results)} files")
    print(f"Results saved to: {evaluator.results_path}")
    print(f"Summary report saved to: {evaluator.summary_path}")
