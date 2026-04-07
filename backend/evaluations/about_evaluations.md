# **VoxFlow AI Evaluation Pipeline Documentation**

This document summarizes the evaluation strategy for the **VoxFlow AI audio preprocessing project**, covering both **transcriber** and **preprocessor** modules. Each module uses dedicated evaluations to ensure **robustness, correctness, and semantic quality**.

---

## **Project Structure**

Evaluation assets are organized under `evaluations/` to separate testing logic from core application code.

```
d:\Projects\audio_preprocessor\backend\evaluations\
|   about_evaluations.md
|   runner.py
|   
+---preprocessor
|   |   __init__.py
|   |   
|   +---evaluation_databases
|   |       functional_executions.json
|   |       judge_executions.jsonl
|   |       
|   +---evaluation_results
|   |       functional_evaluation_summary.md
|   |       judge_evaluation_summary.md
|   |
|   +---evaluation_scripts
|   |   +---ai_judge
|   |   |   |   about_evals.md
|   |   |   |   ai_judge.py
|   |   |   |   evaluation_client.py
|   |   |   |   prompt_builder.py
|   |   |   |   reporter.py
|   |   |   |   storage.py
|   |   |   |   __init__.py
|   |   |
|   |   \---functional_correctness
|   |       |   about_evals.md
|   |       |   functional_correctness.py
|   |       |   metrics.py
|   |       |   reporter.py
|   |       |   runner.py
|   |       |   storage.py
|   |       |   verifier.py
|   |       |   __init__.py
|   |
|   \---__pycache__
|
+---test_data
|   +---preprocessor
|   |       preprocessings.jsonl
|   |       transcriptions_data.jsonl
|   |
|   \---transcriber
|       +---invalids
|       |       test10.docx
|       |       test9.pdf
|       |
|       \---valids
|               test1.m4a
|               test2.mp3
|               test3.wav
|               test4.flac
|               test5.opus
|               test6.aac
|               test7.wma
|               test8.mp3
|
+---transcriber
|   |   __init__.py
|   |
|   +---evaluation_databases
|   |       functional_evaluation_results.jsonl
|   |       lexical_evaluations_result.jsonl
|   |       transcriptions_data.jsonl
|   |       transcriptions_reference_data.jsonl
|   |
|   +---evaluation_results
|   |       functional_evaluation_summary.md
|   |       lexical_evaluation_summary.md
|   |
|   +---evaluation_scripts
|   |   +---functional_correctness
|   |   |   |   about_evals.md
|   |   |   |   functional_correctness.py
|   |   |
|   |   \---lexical_similarity
|   |       |   about_evals.md
|   |       |   lexical_similarity.py
|   |       |   metrics.py
|   |       |   normalizer.py
|   |       |   reporter.py
|   |
|   \---__pycache__
|
\---__pycache__
```

> **Note:** Each `about_evals.md` contains detailed explanations of **evaluation purpose, metrics, and success criteria** for its specific script.

---

## **Test Data Overview**

* **Preprocessor test data:**

  * `transcriptions_data.jsonl` → Raw transcription inputs.
  * `preprocessings.jsonl` → Expected preprocessing outputs, edge cases, and standardized chunks.

* **Transcriber test data:**

  * `valids/` → Supported audio files for functional and lexical evaluation.
  * `invalids/` → Unsupported files to ensure graceful error handling.
  * `transcriptions_reference_data.jsonl` → Reference transcripts for lexical evaluation.

---

## **1. Transcriber Evaluation**
### **1.1 Functional Correctness**
* **Purpose:** Ensure the transcriber runs without errors and produces output for all valid audio files.
* **Metrics & Success:**
    * **Output existence** → A valid transcription is returned.
    * **Session integrity** → Processing completes without crashes.
    * **File-level correctness** → Each transcription file processed correctly.
    * **Error handling** → Invalid files are gracefully rejected.
* **Reference:** transcriber/evaluation_scripts/functional_correctness/about_evals.md
### **1.2 Lexical Similarity**
* **Purpose:** Quantitatively measure how close the transcription matches the reference text.
* **Metrics & Success:**
    * Word overlap ratio
    * Character-level accuracy
    * Overall similarity score (e.g., BLEU or custom metric)
    * Handling of edge cases → Punctuation, capitalization, minor formatting.
* **Reference:** transcriber/evaluation_scripts/lexical_similarity/about_evals.md
---

## **2. Preprocessor Evaluation**

### **2.1 Functional Correctness**

* **Purpose:** Confirm pipeline executes correctly and generates valid outputs.
* **Metrics & Success:**

  * **Chunk completeness:** All expected chunks processed.
  * **LLM retries:** Number of retries during preprocessing.
  * **Output existence:** Preprocessed file generated.
  * **Session integrity:** All processing steps completed without errors.
* **Reference:** `preprocessor/evaluation_scripts/functional_correctness/about_evals.md`

### **2.2 AI-as-Judge Semantic Evaluation**

* **Purpose:** Evaluate semantic quality of preprocessed text using LLM reasoning.
* **Metrics & Success:**

  * **Meaning Preservation** → Retains original intent.
  * **Information Loss** → Minimal content dropped.
  * **Preprocessing Quality** → GOLDEN / ACCEPTABLE / POOR.
  * **Hallucination** → Fabricated content is LOW / MODERATE / HIGH.
  * **Confidence** → 0.0–1.0.
* **Reference:** `preprocessor/evaluation_scripts/ai_judge/about_evals.md`

---

## **3. Evaluation Execution**

* **Runner script:** `runner.py` provides a **GUI interface** for dynamically selecting which evaluation(s) to run.
* **Purpose:** Simplifies execution, allows running any combination of the four evaluations, and automatically generates results and summaries.
* **Outputs:** Stored in `evaluation_databases` and summarized in `evaluation_results`.

---

## **4. Evaluation Metrics and Success Criteria Summary**

| Module       | Evaluation Method      | Key Metrics                                                                              | Success Definition                                     |
| ------------ | ---------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| Transcriber  | Functional Correctness | Output existence, Session integrity                                                      | True if all audio files processed successfully         |
| Transcriber  | Lexical Similarity     | Word overlap, Similarity score                                                           | High similarity with reference transcript              |
| Preprocessor | Functional Correctness | Chunk completeness, LLM retries, Output existence, Session integrity                     | True if all chunks processed, outputs exist, no errors |
| Preprocessor | AI-as-Judge            | Meaning Preservation, Information Loss, Preprocessing Quality, Hallucination, Confidence | GOLDEN/LOW/ACCEPTABLE labels with confidence ≥0.9      |

---

## **5. Evaluation Pipeline Flow Diagram**

```text
                        +-------------------+
                        |   runner.py GUI   |
                        |  (choose evals)   |
                        +--------+----------+
                                 |
+----------------------+----------------------+----------------------+----------------------+
          |                      |                      |                      |
+----------------+      +----------------+     +----------------+     +----------------+
| Transcriber    |      | Transcriber    |     | Preprocessor   |     | Preprocessor   |
| Functional     |      | Lexical        |     | Functional     |     | AI-as-Judge    |
| Correctness    |      | Similarity     |     | Correctness    |     | Semantic Eval  |
+--------+-------+      +--------+-------+     +--------+-------+     +--------+-------+
         |                       |                      |                      |
         v                       v                      v                      v
+----------------+      +----------------+     +----------------+     +----------------+
| Functional     |      | Lexical        |     | Functional     |     | Judge Results  |
| Results DB     |      | Results DB     |     | Results DB     |     | .jsonl         |
| .jsonl/.json   |      | .jsonl         |     | .json/.jsonl   |     +----------------+
+----------------+      +----------------+     +----------------+     
```

**Explanation:**

1. User selects evaluation(s) via GUI.
2. Evaluations run independently.
3. Metrics and success are collected.
4. Results are stored and summarized.

---

## **6. Summary of Results (Example)**

| Module       | Evaluation Method      | Key Benefit             | Result (example)                               |
| ------------ | ---------------------- | ----------------------- | ---------------------------------------------- |
| Transcriber  | Functional Correctness | System reliability      | 100% success                                   |
| Transcriber  | Lexical Similarity     | Textual accuracy        | High similarity across all files               |
| Preprocessor | Functional Correctness | Operational correctness | 100% success                                   |
| Preprocessor | AI-as-Judge            | Semantic quality        | All GOLDEN labels, high confidence (~0.95–1.0) |

---

## **7. Conclusion**

The evaluation pipeline demonstrates a **robust, reliable, and interpretable** audio preprocessing system. By combining **functional correctness tests** with **AI-as-judge reasoning**, VoxFlow AI ensures both **technical stability** and **high-quality semantic output**.

---

This version now:

* Includes **all about_evals.md references**.
* Clearly defines **purpose, metrics, and success** for each evaluation.
* Adds **runner.py GUI explanation**.
* Integrates the **evaluation flow diagram**.
* Removes implementation details while keeping it **comprehensive and professional**.


