"""Text normalization, caption preprocessing, and evaluation helpers."""

import re
from typing import List


def clean_summary_text(text: str) -> str:
    """Clean and normalize raw text for summarization models."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Normalize whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def format_caption_sequence(caption: str, start_token: str = "startseq", end_token: str = "endseq") -> str:
    """Wrap raw caption with start and end sequence tokens."""
    cleaned = clean_summary_text(caption)
    # Remove punctuation for vocabulary consistency
    cleaned = re.sub(r"[^\w\s]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return f"{start_token} {cleaned} {end_token}"


def calculate_token_overlap(reference: str, candidate: str) -> float:
    """Compute basic unigram Jaccard similarity as a lightweight reference score."""
    ref_tokens = set(re.findall(r"\b\w+\b", reference.lower()))
    cand_tokens = set(re.findall(r"\b\w+\b", candidate.lower()))
    if not ref_tokens or not cand_tokens:
        return 0.0
    intersection = ref_tokens.intersection(cand_tokens)
    union = ref_tokens.union(cand_tokens)
    return len(intersection) / len(union)
