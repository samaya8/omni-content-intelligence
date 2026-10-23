"""Tests for ImageCaptioner engine."""

import io
from pathlib import Path
import pytest
from src.captioner import ImageCaptioner


def test_captioner_basic(tmp_path: Path):
    test_img = tmp_path / "sample.jpg"
    test_img.write_bytes(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00")

    captioner = ImageCaptioner()
    res = captioner.generate_caption(test_img, beam_size=3)

    assert isinstance(res["caption"], str)
    assert len(res["caption"]) > 10
    assert 0.0 < res["confidence"] <= 1.0
    assert "detected_entities" in res
    assert "primary_object" in res["detected_entities"]
    assert len(res["candidate_beams"]) == 3


def test_captioner_beam_search():
    captioner = ImageCaptioner()
    # Greedy (k=1)
    res_greedy = captioner.generate_caption("mock_image_a.png", beam_size=1)
    assert res_greedy["decoding_method"] == "greedy"
    assert len(res_greedy["candidate_beams"]) == 1

    # Beam (k=4)
    res_beam = captioner.generate_caption("mock_image_a.png", beam_size=4)
    assert "beam_search" in res_beam["decoding_method"]
    assert len(res_beam["candidate_beams"]) == 4
    # Scores must be ordered descending
    scores = [b["score"] for b in res_beam["candidate_beams"]]
    assert scores == sorted(scores, reverse=True)


def test_captioner_detected_entities():
    captioner = ImageCaptioner()
    res = captioner.generate_caption("mountain_view.jpg")
    entities = res["detected_entities"]
    assert "primary_object" in entities
    assert "attribute" in entities
    assert "action" in entities
    assert "environment" in entities
    # The primary object must appear in the caption
    assert entities["primary_object"] in res["caption"].lower()


def test_captioner_bytes_input():
    captioner = ImageCaptioner()
    img_bytes = b"fake_png_header_and_data_bytes_12345"
    res = captioner.generate_caption(img_bytes)
    assert res["caption"].startswith("A ") or res["caption"].startswith("Close-up") or res["caption"].startswith("View")


def test_captioner_with_mock_hf():
    mock_caption = "A golden retriever sitting in a sunny grassy park."
    mock_hf = lambda img: [{"generated_text": mock_caption}]

    captioner = ImageCaptioner(hf_pipeline=mock_hf)
    res = captioner.generate_caption("dog.jpg", use_hf=True)
    assert res["decoding_method"] == "huggingface_transformer"
    assert res["caption"] == mock_caption
    assert res["confidence"] == 0.92


def test_captioner_reproducibility():
    captioner = ImageCaptioner()
    res1 = captioner.generate_caption("image_constant_test.png")
    res2 = captioner.generate_caption("image_constant_test.png")
    assert res1["caption"] == res2["caption"]
    assert res1["image_hash"] == res2["image_hash"]
