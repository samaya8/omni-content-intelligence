"""Tests for MultimodalAttentionVisualizer."""

import pytest
from src.attention import MultimodalAttentionVisualizer


def test_generate_spatial_attention_map():
    viz = MultimodalAttentionVisualizer()
    grid = viz.generate_spatial_attention_map("dog", grid_size=8, center_x=0.5, center_y=0.5)

    assert len(grid) == 8
    assert len(grid[0]) == 8
    total_prob = sum(sum(row) for row in grid)
    assert 0.99 <= total_prob <= 1.01

    # Center cell (3, 3) should have a higher weight than corner (0, 0)
    assert grid[3][3] > grid[0][0]


def test_generate_caption_attention_series():
    viz = MultimodalAttentionVisualizer()
    caption = "A golden dog runs outside."
    res = viz.generate_caption_attention_series(caption, grid_size=6)

    assert len(res["tokens"]) == 5
    assert len(res["attention_maps"]) == 5
    assert len(res["global_map"]) == 6
    assert len(res["global_map"][0]) == 6

    # Verify global map sums to ~1.0
    global_sum = sum(sum(row) for row in res["global_map"])
    assert 0.99 <= global_sum <= 1.01


def test_compute_text_salience():
    viz = MultimodalAttentionVisualizer()
    sentences = [
        "Machine learning models analyze complex data patterns.",
        "Deep transformers utilize multi-head self-attention.",
        "A short note.",
    ]
    raw_scores = [0.25, 0.95, 0.10]
    salience = viz.compute_text_salience(sentences, raw_scores)

    assert len(salience) == 3
    # Sentence 2 must have the highest weight (1.0)
    assert salience[1]["salience_weight"] == 1.0
    # Sentence 3 must have the lowest weight (0.0)
    assert salience[2]["salience_weight"] == 0.0
    assert 0.0 <= salience[0]["salience_weight"] <= 1.0


def test_render_salience_html():
    viz = MultimodalAttentionVisualizer()
    salience = [
        {"sentence": "Key finding <1>.", "salience_weight": 0.9},
        {"sentence": "Minor detail.", "salience_weight": 0.1},
    ]
    html_out = viz.render_salience_html(salience)

    assert "<span" in html_out
    assert "background-color" in html_out
    # HTML escaping check
    assert "&lt;1&gt;" in html_out
    assert "<1>" not in html_out


def test_compute_alignment_matrix():
    viz = MultimodalAttentionVisualizer()
    sources = [
        "Transformers utilize self-attention mechanisms for sequences.",
        "Convolutional networks process visual image representations.",
    ]
    summaries = [
        "Self-attention mechanisms are central to transformers.",
    ]

    matrix = viz.compute_alignment_matrix(sources, summaries)
    assert len(matrix) == 1  # 1 summary sentence
    assert len(matrix[0]) == 2  # 2 source sentences

    # Summary aligns much higher with source 0 than source 1
    assert matrix[0][0] > matrix[0][1]


def test_empty_inputs():
    viz = MultimodalAttentionVisualizer()
    empty_series = viz.generate_caption_attention_series("")
    assert empty_series["tokens"] == []

    empty_salience = viz.compute_text_salience([])
    assert empty_salience == []

    empty_html = viz.render_salience_html([])
    assert "No content" in empty_html

    empty_matrix = viz.compute_alignment_matrix([], [])
    assert empty_matrix == []
