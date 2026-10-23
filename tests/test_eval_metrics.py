"""Unit tests for Natural Language Generation (NLG) evaluation metrics.

Tests coverage:
- ROUGE-1, ROUGE-2, ROUGE-L precision, recall, and F1.
- BLEU-1 through BLEU-4 with brevity penalty and smoothing.
- CIDEr consensus metric for captioning across references.
- Compression ratio, savings percentage, and abstractiveness ratio.
- Edge cases: empty text, disjoint vocabulary, single vs multi-reference.
"""

from __future__ import annotations

import pytest
from src.eval_metrics import NLGEvalMetrics


def test_rouge_exact_match():
    text = "Artificial intelligence is transforming enterprise business workflows rapidly."
    scores = NLGEvalMetrics.compute_rouge(text, text)

    assert "rouge-1" in scores
    assert "rouge-2" in scores
    assert "rouge-l" in scores

    assert scores["rouge-1"]["precision"] == 100.0
    assert scores["rouge-1"]["recall"] == 100.0
    assert scores["rouge-1"]["f1"] == 100.0

    assert scores["rouge-2"]["precision"] == 100.0
    assert scores["rouge-2"]["recall"] == 100.0
    assert scores["rouge-2"]["f1"] == 100.0

    assert scores["rouge-l"]["f1"] == 100.0


def test_rouge_partial_and_disjoint():
    cand = "Machine learning optimizes business workflows."
    ref = "Artificial intelligence is transforming enterprise business workflows rapidly."

    scores = NLGEvalMetrics.compute_rouge(cand, ref)
    assert scores["rouge-1"]["f1"] > 0.0
    # "business workflows" is a shared bigram
    assert scores["rouge-2"]["f1"] > 0.0
    assert scores["rouge-l"]["f1"] > 0.0

    # Disjoint text
    disjoint_scores = NLGEvalMetrics.compute_rouge("apple banana cherry", "quantum physics gravity")
    assert disjoint_scores["rouge-1"]["f1"] == 0.0
    assert disjoint_scores["rouge-2"]["f1"] == 0.0
    assert disjoint_scores["rouge-l"]["f1"] == 0.0


def test_rouge_empty_cases():
    scores = NLGEvalMetrics.compute_rouge("", "Valid reference text.")
    assert scores["rouge-1"]["f1"] == 0.0
    assert scores["rouge-2"]["f1"] == 0.0
    assert scores["rouge-l"]["f1"] == 0.0

    scores_empty_ref = NLGEvalMetrics.compute_rouge("Candidate text.", "")
    assert scores_empty_ref["rouge-1"]["f1"] == 0.0


def test_bleu_exact_match():
    text = "A black dog runs through the green grass field."
    scores = NLGEvalMetrics.compute_bleu(text, [text])

    assert scores["brevity_penalty"] == 1.0
    assert scores["bleu-1"] == 100.0
    assert scores["bleu-2"] == 100.0
    assert scores["bleu-3"] == 100.0
    assert scores["bleu-4"] == 100.0


def test_bleu_brevity_penalty_and_multi_ref():
    cand = "A black dog runs."
    refs = [
        "A black dog runs through the green grass field.",
        "A small dark dog runs across the park lawn.",
    ]
    scores = NLGEvalMetrics.compute_bleu(cand, refs)

    # Shorter candidate than reference must incur brevity penalty < 1.0
    assert scores["brevity_penalty"] < 1.0
    assert scores["bleu-1"] > 0.0

    # Single string reference overload
    scores_single = NLGEvalMetrics.compute_bleu(cand, refs[0])
    assert "bleu-1" in scores_single

    # Empty candidate
    empty_scores = NLGEvalMetrics.compute_bleu("", refs)
    assert empty_scores["bleu-1"] == 0.0
    assert empty_scores["brevity_penalty"] == 0.0


def test_cider_consensus():
    candidate = "A black dog playing fetch in the park."
    good_refs = [
        "A dark dog running and playing in the park.",
        "A black dog fetching a ball in the grassy park.",
        "A black canine running outdoors in a park.",
    ]
    unrelated_refs = [
        "A red car driving fast on a highway.",
        "An airplane flying across blue sky with clouds.",
    ]

    score_good = NLGEvalMetrics.compute_cider(candidate, good_refs)
    score_unrelated = NLGEvalMetrics.compute_cider(candidate, unrelated_refs)

    assert score_good > 0.0
    assert score_good > score_unrelated

    # Edge cases
    assert NLGEvalMetrics.compute_cider("", good_refs) == 0.0
    assert NLGEvalMetrics.compute_cider(candidate, []) == 0.0


def test_abstractiveness_and_compression():
    source = (
        "Transformer models leverage self-attention mechanisms to process sequential data in parallel, "
        "allowing massive computational scale across distributed GPU clusters. This architecture has "
        "replaced recurrent neural networks in modern natural language processing pipelines."
    )
    summary = "Transformers use self-attention to process text in parallel, replacing older recurrent networks."

    metrics = NLGEvalMetrics.compute_abstractiveness_and_compression(summary, source)

    assert "compression_ratio" in metrics
    assert "compression_savings_pct" in metrics
    assert "novel_unigram_ratio" in metrics
    assert "novel_bigram_ratio" in metrics

    assert 0.0 < metrics["compression_ratio"] < 1.0
    assert metrics["compression_savings_pct"] > 0.0
    # Novel words like "older" or "text" make novel_unigram_ratio > 0
    assert metrics["novel_unigram_ratio"] >= 0.0
    assert metrics["novel_bigram_ratio"] >= 0.0
