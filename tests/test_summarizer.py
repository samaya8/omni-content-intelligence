"""Tests for TextSummarizer engine."""

import pytest
from src.summarizer import TextSummarizer


SAMPLE_DOCUMENT = (
    "Artificial intelligence and deep learning architectures have revolutionized automated natural language processing. "
    "Transformer models utilize self-attention mechanisms to compute contextual representations across input sequences. "
    "These neural architectures enable state-of-the-art results on machine translation, text classification, and summarization. "
    "Furthermore, multi-head attention allows the model to jointly attend to information from different representation subspaces. "
    "Pretrained models like BERT, BART, and T5 can be fine-tuned efficiently on specific downstream benchmark datasets. "
    "However, large parameter counts require substantial computational resources and memory for inference and training. "
    "Recent research focuses on model distillation, pruning, and quantization to deploy lightweight architectures onto edge devices. "
    "These advancements demonstrate the practical feasibility of deploying intelligent natural language systems at scale."
)


def test_split_sentences():
    summarizer = TextSummarizer()
    sentences = summarizer.split_sentences(SAMPLE_DOCUMENT)
    assert len(sentences) == 8
    assert sentences[0].startswith("Artificial intelligence")
    assert sentences[-1].endswith("at scale.")


def test_textrank_scoring():
    summarizer = TextSummarizer()
    sentences = summarizer.split_sentences(SAMPLE_DOCUMENT)
    scores = summarizer._compute_textrank_scores(sentences)
    assert len(scores) == len(sentences)
    assert all(s > 0 for s in scores)
    # The top scoring sentence should contain prominent keywords like "transformer", "attention", or "models"
    top_idx = scores.index(max(scores))
    assert any(k in sentences[top_idx].lower() for k in ["transformer", "models", "attention", "architectures"])


def test_extractive_summarize_lengths():
    summarizer = TextSummarizer()
    # Short
    res_short = summarizer.summarize(SAMPLE_DOCUMENT, strategy="extractive", length="short")
    assert len(res_short["sentences"]) == 2
    assert res_short["compression_ratio"] < 0.6

    # Medium
    res_med = summarizer.summarize(SAMPLE_DOCUMENT, strategy="extractive", length="medium")
    assert len(res_med["sentences"]) == 4

    # Long
    res_long = summarizer.summarize(SAMPLE_DOCUMENT, strategy="extractive", length="long")
    assert len(res_long["sentences"]) == 6


def test_extractive_format_styles():
    summarizer = TextSummarizer()
    # Bullet points
    res_bullets = summarizer.summarize_extractive(SAMPLE_DOCUMENT, num_sentences=3, format_style="bullet_points")
    assert res_bullets["summary"].startswith("• ")
    assert res_bullets["summary"].count("• ") == 3

    # Dense
    res_dense = summarizer.summarize_extractive(SAMPLE_DOCUMENT, num_sentences=3, format_style="dense")
    assert "**Executive Takeaway**" in res_dense["summary"]
    assert "**Key Findings**" in res_dense["summary"]

    # Executive
    res_exec = summarizer.summarize_extractive(SAMPLE_DOCUMENT, num_sentences=3, format_style="executive")
    assert "• " not in res_exec["summary"]
    assert len(res_exec["summary"]) > 50


def test_abstractive_fallback():
    summarizer = TextSummarizer()
    res = summarizer.summarize(SAMPLE_DOCUMENT, strategy="abstractive", length="short")
    assert res["strategy"] == "abstractive_neural_fallback"
    assert "In essence" in res["summary"] or "Overall" in res["summary"] or "Crucially" in res["summary"]
    assert res["compression_ratio"] < 1.0


def test_abstractive_with_mock_hf():
    mock_summary = "Transformers achieve state-of-the-art NLP performance using self-attention mechanisms."
    mock_pipeline = lambda text, **kwargs: [{"summary_text": mock_summary}]

    summarizer = TextSummarizer(hf_pipeline=mock_pipeline)
    res = summarizer.summarize(SAMPLE_DOCUMENT, strategy="abstractive")
    assert res["strategy"] == "abstractive_hf"
    assert res["summary"] == mock_summary


def test_summarizer_empty_and_single():
    summarizer = TextSummarizer()
    empty_res = summarizer.summarize("")
    assert empty_res["summary"] == ""

    single_doc = "This is a single standalone sentence about machine learning."
    single_res = summarizer.summarize(single_doc)
    assert single_res["summary"] == single_doc


def test_mmr_redundancy_penalty():
    summarizer = TextSummarizer()
    # Text with duplicate content sentences
    redundant_doc = (
        "Transformers use self-attention to process text sequences. "
        "Transformers use self-attention mechanisms to process input sequences. "
        "Convolutional networks process visual features across spatial grids. "
        "Recurrent networks maintain hidden states sequentially over time."
    )
    res = summarizer.summarize_extractive(redundant_doc, num_sentences=2)
    # MMR should pick sentence 1 and sentence 3 or 4, NOT sentence 1 and sentence 2
    indices = res["sentence_indices"]
    assert not (0 in indices and 1 in indices)
