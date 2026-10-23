from src.text_utils import (
    calculate_token_overlap,
    clean_summary_text,
    format_caption_sequence,
)


def test_clean_summary_text():
    raw = "Check this out: <p>Hello world!</p> Visit https://example.com/test now."
    cleaned = clean_summary_text(raw)
    assert "<p>" not in cleaned
    assert "https://" not in cleaned
    assert "hello world!" in cleaned


def test_format_caption_sequence():
    caption = "A cute brown dog is jumping over a hurdle."
    formatted = format_caption_sequence(caption)
    assert formatted.startswith("startseq")
    assert formatted.endswith("endseq")
    assert "a cute brown dog is jumping over a hurdle" in formatted


def test_calculate_token_overlap():
    ref = "the quick brown fox jumps"
    cand = "the quick brown fox"
    score = calculate_token_overlap(ref, cand)
    assert 0.75 <= score <= 1.0

    score_disjoint = calculate_token_overlap("cat dog", "apple banana")
    assert score_disjoint == 0.0
