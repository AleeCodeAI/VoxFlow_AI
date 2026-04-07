# Lexical Transcription Evaluation — Documentation Summary

## Purpose

This evaluation measures the **textual quality** of transcriptions by comparing them against reference transcripts using standard lexical metrics.

It answers the question:

**“How accurate is the transcription text compared to the ground truth?”**

This evaluation focuses on **transcription quality**, not system reliability.

---

## Scope

This evaluation validates:

* Word-level accuracy
* Character-level accuracy
* Phrase-level similarity using n-grams
* Overall transcription quality classification
* Structured reporting of results

This evaluation assumes that the transcription pipeline has already worked correctly and produced outputs.

---

## Evaluation Pipeline

The complete pipeline executed by `LexicalEvaluator.run()` is:

1. Load transcriptions
2. Load reference transcripts
3. Normalize both texts
4. Pair transcription with its reference
5. Compute lexical metrics for each pair
6. Assign a quality label
7. Save results and generate a summary report

---

## Data Preparation

Before evaluation, both transcription and reference texts are passed through a **Normalizer** which ensures:

* Consistent casing
* Removal of noise and punctuation inconsistencies
* Fair metric computation

The evaluator operates on a list of:

```
NormalizedObject(
    id,
    file_name,
    transcription,
    reference
)
```

---

## Metrics Used

For each transcription–reference pair, three metrics are computed.

### Word Error Rate (WER)

Computed using `jiwer.wer`.

Measures the proportion of word-level edits (insertions, deletions, substitutions) required to convert the transcription into the reference.

* Range: `0 → ∞`
* Ideal: `0`

---

### Character Error Rate (CER)

Computed using `jiwer.cer`.

Measures character-level edits between transcription and reference.

* More sensitive than WER
* Ideal: `0`

---

### N-gram Similarity (Jaccard Similarity on Bigrams)

Computed using `compute_ngram_similarity`.

* Splits texts into word tokens
* Forms bigrams (2-grams)
* Computes Jaccard similarity between sets of bigrams

Formula:

```
|intersection of ngrams| / |union of ngrams|
```

* Range: `0 → 1`
* Ideal: `1`

This metric captures **phrase-level similarity**, which WER/CER do not capture well.

---

## Quality Label Assignment

Each result is assigned a quality label using predefined thresholds.

| Condition                                   | Quality    |
| ------------------------------------------- | ---------- |
| WER ≤ 0.10 AND CER ≤ 0.05 AND N-gram ≥ 0.80 | OK         |
| WER ≤ 0.25 AND CER ≤ 0.15 AND N-gram ≥ 0.60 | Acceptable |
| Otherwise                                   | Bad        |

This converts numeric metrics into an easily interpretable classification.

---

## Definition of Success

For each file, the evaluator produces a `LexicalMetrics` object containing:

* WER
* CER
* N-gram similarity
* Quality label
* Timestamp

There is no pass/fail at the system level. Instead, quality is **graded** per file.

---

## Output Artifacts

### Detailed Results (JSONL)

Saved using the `Reporter.save_execution()` method.

Contains per-file lexical metrics for further analysis.

### Summary Report (Markdown)

Generated using `Reporter.generate_report()`.

Provides aggregated insights across all evaluated files.

---

## What This Evaluation Guarantees

This evaluation ensures:

* Accurate measurement of transcription quality
* Fair comparison via normalization
* Multi-level similarity measurement (word, character, phrase)
* Interpretable quality grading
* Structured and reproducible reporting

---

## What This Evaluation Does Not Cover

This evaluation does not assess:

* Input validation
* Error handling
* Chunk processing
* Database persistence
* System reliability

Those aspects are covered by the **functional evaluation**.

---

## Summary

The lexical evaluation provides a quantitative and qualitative assessment of transcription accuracy by combining WER, CER, and n-gram similarity, converting them into meaningful quality labels, and producing structured evaluation reports.
