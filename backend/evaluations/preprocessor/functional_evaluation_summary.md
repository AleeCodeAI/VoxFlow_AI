# Preprocessor Evaluation Summary

**Generated:** 2026-01-24 13:07:03

---

## Overview

| Metric | Value |
|--------|-------|
| Total Executions | 7 |
| Successful | 7 |
| Failed | 0 |
| Success Rate | 100.00% |

---

## Performance Metrics

| Metric | Average |
|--------|---------|
| LLM Retries | 0.00 |
| Content Quality Score | 0.790 |
| Chunk Completeness Rate | 100.00% |
| Output Existence Rate | 100.00% |

---

## Detailed Results

| File Name | ID | Retries | Quality | Label | Complete | Output | Status |
|-----------|-----|---------|---------|-------|----------|--------|--------|
| test1.m4a | e6288c87... | 0 | 0.702 | OKAY | ✓ | ✓ | ✅ Pass |
| test2.mp3 | f1a4e95c... | 0 | 0.976 | GOLDEN | ✓ | ✓ | ✅ Pass |
| test3.wav | af5c589d... | 0 | 0.733 | OKAY | ✓ | ✓ | ✅ Pass |
| test4.flac | 59c43a50... | 0 | 0.824 | GOLDEN | ✓ | ✓ | ✅ Pass |
| test5.opus | 52ca95c7... | 0 | 0.887 | GOLDEN | ✓ | ✓ | ✅ Pass |
| test6.aac | dc6adcad... | 0 | 0.700 | OKAY | ✓ | ✓ | ✅ Pass |
| test7.wma | 2be28f93... | 0 | 0.710 | OKAY | ✓ | ✓ | ✅ Pass |

---

## Key Insights

- **Average LLM Retries:** 0.00 retries per execution
- **Quality Assessment:** Average content quality score is 0.790
- **Reliability:** 100.0% of executions completed all chunks
- **Data Persistence:** 100.0% of outputs were successfully saved

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

No failures detected. All executions completed successfully.
