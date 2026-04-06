def compute_ngram_similarity(reference: str, hypothesis: str, ngram_size=2):
    """
    Compute n-gram similarity score between reference and hypothesis texts.

    Uses Jaccard similarity on n-grams (default bigrams) to measure the
    overlap between reference and hypothesis texts. Higher scores indicate
    greater similarity.

    Args:
        reference (str): The reference text
        hypothesis (str): The hypothesis (transcription) text
        ngram_size (int, optional): Size of n-grams to use. Defaults to 2

    Returns:
        float: N-gram similarity score between 0 and 1, where 1 is identical
    """
    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()

    if len(reference_tokens) < ngram_size or len(hypothesis_tokens) < ngram_size:
        return 0.0

    reference_ngrams = set(
        tuple(reference_tokens[i : i + ngram_size])
        for i in range(len(reference_tokens) - ngram_size + 1)
    )
    hypothesis_ngrams = set(
        tuple(hypothesis_tokens[i : i + ngram_size])
        for i in range(len(hypothesis_tokens) - ngram_size + 1)
    )

    intersection = reference_ngrams.intersection(hypothesis_ngrams)
    union = reference_ngrams.union(hypothesis_ngrams)

    return len(intersection) / len(union) if union else 0.0


def determine_quality_label(wer_score, cer_score, ngram_score):
    """
    Assign a quality label based on metric thresholds.

    Categorizes transcription quality as "OK", "Acceptable", or "Bad"
    based on WER, CER, and n-gram similarity thresholds.

    Args:
        wer_score (float): Word Error Rate
        cer_score (float): Character Error Rate
        ngram_score (float): N-gram similarity score

    Returns:
        str: Quality label - "OK", "Acceptable", or "Bad"
    """
    if wer_score <= 0.10 and cer_score <= 0.05 and ngram_score >= 0.80:
        return "OK"
    elif wer_score <= 0.25 and cer_score <= 0.15 and ngram_score >= 0.60:
        return "Acceptable"
    else:
        return "Bad"
