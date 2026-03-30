from pydantic_schemas import Result, EvaluationError
from datetime import datetime
from collections import Counter
from typing import List


class Reporter:
    """Generates markdown summary report from evaluation results"""

    def __init__(self, summary_path: str):
        self.summary_path = summary_path

    def generate_summary(self, results: List[Result]):
        try:
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

        except Exception as e:
            error = EvaluationError(
                error_type="SummaryGenerationError",
                message=str(e),
                timestamp=datetime.now().isoformat(),
                context={"total_results": len(results), "summary_path": self.summary_path}
            )
            raise Exception(error.model_dump_json())