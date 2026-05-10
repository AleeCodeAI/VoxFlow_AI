# Preprocessor Functional Evaluation

## Overview

The **Preprocessor Functional Evaluation** pipeline verifies that the preprocessor system executes correctly on a set of test transcriptions. It ensures that the preprocessing pipeline processes all chunks, produces expected outputs, and maintains session integrity.

The evaluation does **not** assess the quality of the transcriptions themselves; it is purely a functional correctness check of the preprocessing workflow.

---

## Structure

1. **Evaluation Pipeline**
   The main pipeline coordinates the evaluation:

   * Loads test transcription data.
   * Executes the preprocessor pipeline for each transcription.
   * Collects and records evaluation metrics.
   * Saves results and generates a summary report.

2. **Core Components**

   * **Runner:** Executes the preprocessor on individual transcription objects.
   * **Metrics:** Extracts simple functional metrics from the preprocessing report.
   * **Verifier:** Confirms that preprocessed output files exist.
   * **Storage:** Saves individual evaluation results.
   * **Reporter:** Generates a summary report of all evaluations.

3. **Evaluation Flow**

   1. Load test transcriptions from JSONL file.
   2. For each transcription:

      * Run preprocessor pipeline.
      * Capture metrics (`chunk_completeness`, `llm_retries`).
      * Verify output file existence.
      * Determine session integrity (success if preprocessor completes and output exists).
   3. Store results and generate summary report.

---

## Metrics

The evaluation records the following metrics for each transcription:

| Metric                 | Description                                                                           | Success Definition                                             |
| ---------------------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| **Chunk Completeness** | Indicates whether all chunks of the transcription were processed by the preprocessor. | `True` if all chunks were processed.                           |
| **LLM Retries**        | Number of retries required by the preprocessor’s LLM component.                       | Fewer retries indicate smoother execution; 0 retries is ideal. |
| **Output Existence**   | Whether the preprocessor successfully produced an output file for the transcription.  | `True` if output file exists.                                  |
| **Session Integrity**  | Overall functional success for the transcription pipeline.                            | `True` if preprocessor ran successfully and output exists.     |

> **Note:** Execution time is not tracked in this evaluation. The focus is strictly on pipeline functionality and output correctness.

---

## Success Criteria

A transcription evaluation is considered **successful** if:

1. All chunks are processed (`chunk_completeness = True`).
2. Preprocessed output exists (`output_existence = True`).
3. The pipeline session completes without critical errors (`session_integrity = True`).

The evaluation summary reports the overall success rate across all test transcriptions.

---

## Output

* **Execution Results:** JSON file containing per-transcription metrics.
* **Summary Report:** Markdown file summarizing all metrics and overall functional success rates.

## Summary

The Preprocessor Functional Evaluation ensures that the preprocessing pipeline:

* Processes all transcription chunks correctly.
* Produces the expected output files.
* Completes pipeline execution without errors.
* Records key functional metrics for verification.

The results provide a clear view of the preprocessor’s operational reliability and highlight any failed or incomplete runs, enabling developers to quickly identify and fix pipeline issues.