from openai import OpenAI 
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from datetime import datetime
from color import Logger
import os 
import logging 
import json
from typing import List, Dict
from collections import Counter

load_dotenv(override=True)
api_key = os.getenv("OPENROUTER_API_KEY")
base_url = os.getenv("OPENROUTER_URL")
gpt = os.getenv("GPT_MODEL")

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)


class EvaluationError(BaseModel):
    """Custom exception model for evaluation errors"""
    error_type: str = Field(description="Type of error that occurred")
    message: str = Field(description="Detailed error message")
    timestamp: str = Field(description="When the error occurred")
    context: dict = Field(default_factory=dict, description="Additional context about the error")


class Result(BaseModel):
    """Complete evaluation result with metadata"""
    id: str = Field(description="Unique identifier for the evaluation result")
    file_name: str = Field(description="Name of the file being evaluated")
    meaning_preservation: str = Field(description="Score for meaning preservation: HIGH | MODERATE | LOW")
    information_loss: str = Field(description="The amount of information lost during preprocessing: HIGH | MODERATE | LOW")
    preprocessing_quality: str = Field(description="How well the preprocessing was done: GOLDEN | ACCEPTABLE | POOR")
    hallucination: str = Field(description="How much AI hallucinated while preprocessing: HIGH | MODERATE | LOW")
    confidence: float = Field(description="Confidence level of AI in the Output")
    reasoning: str = Field(description="Detailed reasoning behind the values given")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class AIResult(BaseModel):
    """AI evaluation result without metadata"""
    meaning_preservation: str = Field(description="Score for meaning preservation: HIGH | MODERATE | LOW")
    information_loss: str = Field(description="The amount of information lost during preprocessing: HIGH | MODERATE | LOW")
    preprocessing_quality: str = Field(description="How well the preprocessing was done: GOLDEN | ACCEPTABLE | POOR")
    hallucination: str = Field(description="How much AI hallucinated while preprocessing: HIGH | MODERATE | LOW")
    confidence: float = Field(description="Confidence level of AI in the Output")
    reasoning: str = Field(description="Detailed reasoning behind the values given")


SYSTEM_PROMPT = """ 
PERSONA:
You are an expert AI judge specializing in evaluating the quality of AI-generated text preprocessing.

TASK:
Your task is to assess the preprocessing based on specific criteria and provide a detailed evaluation on the following metrics:
1. meaning_preservation (HIGH | MODERATE | LOW)
2. information_loss (HIGH | MODERATE | LOW)
3. preprocessing_quality (GOLDEN | ACCEPTABLE | POOR)
4. hallucination (HIGH | MODERATE | LOW)
5. confidence (0.0 to 1.0)
6. reasoning (text explaining your judgments)

CONSTRAINTS:
1. Use only the VALID values for each metric as mentioned above.
2. Reason **before assigning any score**.
3. Provide detailed reasoning for each score in the "reasoning" field.
4. You must output a valid JSON object only, without any extra text or commentary.

OUTPUT FORMAT:
{
  "meaning_preservation": "...",
  "information_loss": "...",
  "preprocessing_quality": "...",
  "hallucination": "...",
  "confidence": ...,
  "reasoning": "..."
}

REASONING STEPS:
Step 1: Analyze the original and preprocessed transcriptions thoroughly. Evaluate how much of the meaning is preserved. Then mark meaning_preservation as HIGH (very well preserved), MODERATE (some meaning lost), or LOW (significant meaning lost).

Step 2: Evaluate information_loss by comparing the original transcription with the preprocessed data. Mark it as HIGH (a lot of information lost), MODERATE (some information lost), or LOW (minimal information lost).

Step 3: Assess the overall quality of preprocessing based on clarity, coherence, and relevance. Mark preprocessing_quality as GOLDEN (excellent), ACCEPTABLE (good), or POOR (subpar).

Step 4: Evaluate hallucination by checking for any fabricated or incorrect information in the preprocessed data. Mark hallucination as HIGH (frequent hallucinations), MODERATE (some hallucinations), or LOW (rare or none).

Step 5: Provide a confidence score between 0.0 and 1.0 indicating your confidence in the above evaluations.

Step 6: Combine all your reasoning into the "reasoning" field, justifying each metric assignment.

EXAMPLE:

Original transcription: "I didn't go to the store yesterday, but I went today."
Preprocessed transcription: "I went to the store today."

{
  "meaning_preservation": "MODERATE",
  "information_loss": "MODERATE",
  "preprocessing_quality": "ACCEPTABLE",
  "hallucination": "LOW",
  "confidence": 0.95,
  "reasoning": "The preprocessed text removed the temporal clause 'didn't go yesterday'. This reduces meaning preservation to MODERATE, with moderate information loss. Preprocessing quality is ACCEPTABLE as main content is preserved. No hallucinations are present. Overall confidence is high."
}
"""

USER_PROMPT = """ 
Here is the original transcription:
{transcription}

And, here is the preprocessed transcription:
{preprocessed_transcription}

Now, please evaluate them
"""


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
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self.system_prompt = SYSTEM_PROMPT
            self.user_prompt = USER_PROMPT
            self.model = gpt
            self.transcriptions_path = r"D:\Projects\audio_preprocessor\backend\evaluations\test_data\preprocessor\transcriptions_data.jsonl"
            self.preprocessed_transcriptions_path = r"D:\Projects\audio_preprocessor\backend\evaluations\test_data\preprocessor\preprocessings.jsonl"
            self.summary_path = r"D:\Projects\audio_preprocessor\backend\evaluations\preprocessor\judge_evaluation_summary.md"
            self.execution_path = r"D:\Projects\audio_preprocessor\backend\evaluations\preprocessor\judge_executions.jsonl"
            self.log(f"AI Judge initialized successfully with model: {self.model}")
        except Exception as e:
            error = EvaluationError(
                error_type="InitializationError",
                message=str(e),
                timestamp=datetime.now().isoformat(),
                context={"api_key_present": bool(api_key), "base_url": base_url}
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

            with open(self.preprocessed_transcriptions_path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    preprocessed_obj = json.loads(line.strip())
                    preprocessed_transcriptions.append(preprocessed_obj)
                self.log(f"Loaded {len(preprocessed_transcriptions)} preprocessed transcriptions")

            self.log("Validating ID and name consistency...")
            mismatches = []
            for trans, prep in zip(transcriptions, preprocessed_transcriptions):
                trans_id = trans.get("id")
                prep_id = prep.get("id")
                trans_name = trans.get("name")
                prep_name = prep.get("name")
                
                if trans_id != prep_id or trans_name != prep_name:
                    mismatches.append({
                        "transcription": {"id": trans_id, "name": trans_name},
                        "preprocessed": {"id": prep_id, "name": prep_name}
                    })
            
            if mismatches:
                error = EvaluationError(
                    error_type="ValidationError",
                    message=f"Found {len(mismatches)} mismatches between transcription and preprocessed objects",
                    timestamp=datetime.now().isoformat(),
                    context={"mismatches": mismatches}
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
                    "preprocessed_path": self.preprocessed_transcriptions_path
                }
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
                    "preprocessed_path": self.preprocessed_transcriptions_path
                }
            )
            self.log(f"Error loading data: {error.message}")
            raise Exception(error.model_dump_json())

    def make_messages(self, transcription: str, preprocessed_transcription: str) -> List[Dict]:
        """Create message array for API call with system and user prompts"""
        system_message = {"role": "system", "content": self.system_prompt}
        user_content = self.user_prompt.format(
            transcription=transcription,
            preprocessed_transcription=preprocessed_transcription
        )
        user_message = {"role": "user", "content": user_content}
        return [system_message, user_message]
    
    def save_execution(self, result: Result, mode: str = "a"):
        """Save individual evaluation result to execution log file"""
        try:
            with open(self.execution_path, mode, encoding="utf-8") as f:
                f.write(result.model_dump_json() + "\n")
            self.log(f"Saved execution result for: {result.id}")
        except Exception as e:
            error = EvaluationError(
                error_type="ExecutionSaveError",
                message=str(e),
                timestamp=datetime.now().isoformat(),
                context={"result_id": result.id, "file_path": self.execution_path}
            )
            self.log(f"Error saving execution: {error.message}")
            raise Exception(error.model_dump_json())
    
    def generate_summary(self, results: List[Result]):
        """Generate markdown summary with statistics and average metrics"""
        try:
            self.log("Generating evaluation summary...")
            
            total = len(results)
            successful = sum(1 for r in results if r.preprocessing_quality != "POOR")
            failed = total - successful
            
            success_rate = (successful / total * 100) if total > 0 else 0
            failure_rate = (failed / total * 100) if total > 0 else 0
            
            meaning_counts = Counter(r.meaning_preservation for r in results)
            info_loss_counts = Counter(r.information_loss for r in results)
            quality_counts = Counter(r.preprocessing_quality for r in results)
            hallucination_counts = Counter(r.hallucination for r in results)
            
            avg_confidence = sum(r.confidence for r in results) / total if total > 0 else 0
            
            metric_weights = {
                "meaning_preservation": {"HIGH": 3, "MODERATE": 2, "LOW": 1},
                "information_loss": {"LOW": 3, "MODERATE": 2, "HIGH": 1},
                "preprocessing_quality": {"GOLDEN": 3, "ACCEPTABLE": 2, "POOR": 1},
                "hallucination": {"LOW": 3, "MODERATE": 2, "HIGH": 1}
            }
            
            meaning_score = sum(metric_weights["meaning_preservation"][r.meaning_preservation] for r in results) / (total * 3) if total > 0 else 0
            info_loss_score = sum(metric_weights["information_loss"][r.information_loss] for r in results) / (total * 3) if total > 0 else 0
            quality_score = sum(metric_weights["preprocessing_quality"][r.preprocessing_quality] for r in results) / (total * 3) if total > 0 else 0
            hallucination_score = sum(metric_weights["hallucination"][r.hallucination] for r in results) / (total * 3) if total > 0 else 0
            
            markdown = f"""# AI Judge Evaluation Summary

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Overview
- **Total Evaluations:** {total}
- **Successful:** {successful} ({success_rate:.2f}%)
- **Failed:** {failed} ({failure_rate:.2f}%)

---

## Metric Distributions

### Meaning Preservation
| Level | Count | Percentage |
|-------|-------|------------|
| HIGH | {meaning_counts.get('HIGH', 0)} | {(meaning_counts.get('HIGH', 0) / total * 100):.2f}% |
| MODERATE | {meaning_counts.get('MODERATE', 0)} | {(meaning_counts.get('MODERATE', 0) / total * 100):.2f}% |
| LOW | {meaning_counts.get('LOW', 0)} | {(meaning_counts.get('LOW', 0) / total * 100):.2f}% |

**Average Score:** {meaning_score:.2f}/1.0

### Information Loss
| Level | Count | Percentage |
|-------|-------|------------|
| LOW | {info_loss_counts.get('LOW', 0)} | {(info_loss_counts.get('LOW', 0) / total * 100):.2f}% |
| MODERATE | {info_loss_counts.get('MODERATE', 0)} | {(info_loss_counts.get('MODERATE', 0) / total * 100):.2f}% |
| HIGH | {info_loss_counts.get('HIGH', 0)} | {(info_loss_counts.get('HIGH', 0) / total * 100):.2f}% |

**Average Score:** {info_loss_score:.2f}/1.0

### Preprocessing Quality
| Level | Count | Percentage |
|-------|-------|------------|
| GOLDEN | {quality_counts.get('GOLDEN', 0)} | {(quality_counts.get('GOLDEN', 0) / total * 100):.2f}% |
| ACCEPTABLE | {quality_counts.get('ACCEPTABLE', 0)} | {(quality_counts.get('ACCEPTABLE', 0) / total * 100):.2f}% |
| POOR | {quality_counts.get('POOR', 0)} | {(quality_counts.get('POOR', 0) / total * 100):.2f}% |

**Average Score:** {quality_score:.2f}/1.0

### Hallucination
| Level | Count | Percentage |
|-------|-------|------------|
| LOW | {hallucination_counts.get('LOW', 0)} | {(hallucination_counts.get('LOW', 0) / total * 100):.2f}% |
| MODERATE | {hallucination_counts.get('MODERATE', 0)} | {(hallucination_counts.get('MODERATE', 0) / total * 100):.2f}% |
| HIGH | {hallucination_counts.get('HIGH', 0)} | {(hallucination_counts.get('HIGH', 0) / total * 100):.2f}% |

**Average Score:** {hallucination_score:.2f}/1.0

---

## Overall Performance

- **Average Confidence:** {avg_confidence:.4f}
- **Overall Quality Score:** {(meaning_score + info_loss_score + quality_score + hallucination_score) / 4:.2f}/1.0

---

## Detailed Results

| ID | File | Meaning | Info Loss | Quality | Hallucination | Confidence |
|----|------|---------|-----------|---------|---------------|------------|
"""
            for r in results:
                markdown += f"| {r.id} | {r.file_name} | {r.meaning_preservation} | {r.information_loss} | {r.preprocessing_quality} | {r.hallucination} | {r.confidence:.2f} |\n"
            
            with open(self.summary_path, "w", encoding="utf-8") as f:
                f.write(markdown)
            
            self.log(f"Summary saved to: {self.summary_path}")
            self.log(f"Success Rate: {success_rate:.2f}% | Overall Score: {(meaning_score + info_loss_score + quality_score + hallucination_score) / 4:.2f}/1.0")
            
        except Exception as e:
            error = EvaluationError(
                error_type="SummaryGenerationError",
                message=str(e),
                timestamp=datetime.now().isoformat(),
                context={"total_results": len(results), "summary_path": self.summary_path}
            )
            self.log(f"Error generating summary: {error.message}")
            raise Exception(error.model_dump_json())
    
    def evaluate(self):
        """Execute evaluation process for all transcription pairs"""
        try:
            self.log("Starting evaluation process...")
            transcriptions, preprocessed_transcriptions = self.load()
            
            if len(transcriptions) != len(preprocessed_transcriptions):
                raise ValueError(f"Mismatch in data lengths: {len(transcriptions)} vs {len(preprocessed_transcriptions)}")
            
            results = []
            total_pairs = len(transcriptions)
            
            for idx, (trans_obj, prep_obj) in enumerate(zip(transcriptions, preprocessed_transcriptions), 1):
                try:
                    trans_id = trans_obj.get("id")
                    trans_name = trans_obj.get("name")
                    transcription = trans_obj.get("transcription")
                    preprocessed_transcription = prep_obj.get("preprocessed_transcription")
                    
                    self.log(f"Evaluating pair {idx}/{total_pairs} - ID: {trans_id}, File: {trans_name}")
                    
                    messages = self.make_messages(transcription, preprocessed_transcription)
                    
                    response = self.client.chat.completions.parse(
                        model=self.model,
                        messages=messages,
                        temperature=0.2,
                        top_p=0.93,
                        response_format=AIResult
                    )
                    
                    ai_result = response.choices[0].message.parsed
                    
                    result = Result(
                        id=trans_id,
                        file_name=trans_name,
                        meaning_preservation=ai_result.meaning_preservation,
                        information_loss=ai_result.information_loss,
                        preprocessing_quality=ai_result.preprocessing_quality,
                        hallucination=ai_result.hallucination,
                        confidence=ai_result.confidence,
                        reasoning=ai_result.reasoning
                    )
                    
                    write_mode = "w" if idx == 1 else "a"
                    self.save_execution(result, mode=write_mode)
                    results.append(result)
                    
                    self.log(f"Evaluation {idx} completed - ID: {trans_id}, Quality: {result.preprocessing_quality}, Confidence: {result.confidence:.2f}")
                    
                except Exception as e:
                    error = EvaluationError(
                        error_type="EvaluationError",
                        message=str(e),
                        timestamp=datetime.now().isoformat(),
                        context={"pair_index": idx, "total_pairs": total_pairs, "id": trans_obj.get("id", "unknown")}
                    )
                    self.log(f"Error evaluating pair {idx} (ID: {trans_obj.get('id', 'unknown')}): {error.message}")
                    continue
            
            self.log(f"Evaluation completed: {len(results)}/{total_pairs} successful")
            
            if results:
                self.generate_summary(results)
            
            return results
            
        except Exception as e:
            error = EvaluationError(
                error_type="EvaluationProcessError",
                message=str(e),
                timestamp=datetime.now().isoformat(),
                context={"stage": "main_evaluation"}
            )
            self.log(f"Critical error in evaluation process: {error.message}")
            raise Exception(error.model_dump_json())
        
if __name__ == "__main__":
    judge = AIJudge()
    result = judge.evaluate()
    print("================================================")
    print(f"Total Evaluations Completed: {len(result)}")
    print("================================================")